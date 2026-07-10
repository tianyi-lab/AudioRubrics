# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
)

try:
    import swanlab

    _swanlab_available = True
except ImportError:
    _swanlab_available = False


try:
    from transformers import Qwen2_5OmniModel
    from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
except ImportError:
    print("Qwen2_5OmniModel not found, please update transformers")


import base64
import copy
import io

import requests
from qwen_omni_utils import process_mm_info, process_vision_info
from trl.data_utils import is_conversational, maybe_apply_chat_template
from trl.models import prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

from omni_r1.debug_utils import wait_for_debugger_if_rank0
from omni_r1.load_datasets.avqa import SYSTEM_PROMPT, STAGE2_USER_TEMPLATE

# Regex to extract <description>...</description> content from stage-1 completions
DESCRIPTION_RE = re.compile(r"<description>\s*(.*?)\s*</description>", re.DOTALL | re.IGNORECASE)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

# unwrap_model_for_generation = partial(unwrap_model_for_generation,
#                                       gather_deepspeed3_params = True
#                                       )

USE_VLLM = os.getenv("USE_VLLM") == "1"
VLLM_SERVER_PATH = os.getenv("VLLM_SERVER_PATH")
MODEL = None
if USE_VLLM:
    base_url = copy.deepcopy(VLLM_SERVER_PATH)
    if base_url.endswith("/generate"):
        base_url = base_url.replace("/generate", "")
    models_url = f"{base_url}/v1/models"

    response = requests.get(models_url, timeout=10)
    if response.status_code == 200:
        data = response.json()
        models = [model["id"] for model in data.get("data", [])]
        if models:
            print(f"Available models: {models}")
            MODEL = models[0]
        else:
            raise ValueError("No models found in response")

    else:
        raise ValueError(
            f"Failed to get models: {response.status_code}, {response.text}"
        )


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args=None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        train_dataloader=None,  # 新增参数
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Store custom dataloader
        self.custom_train_dataloader = train_dataloader

        # Initialize default args if needed
        args = self._init_default_args(model, args)

        # Initialize models (main model and reference model)
        model, model_id, model_init_kwargs = self._init_models(
            model, args, attn_implementation
        )

        # Initialize processing class
        processing_class, pad_token_id = self._init_processing_class(
            processing_class, model_id, max_pixels, min_pixels
        )

        # Initialize reward functions and their processing classes
        self._init_reward_functions(
            reward_funcs, reward_processing_classes, model_init_kwargs
        )

        # Initialize training configuration
        self._init_training_config(args, script_args, pad_token_id)

        # Initialize metrics and model configurations
        self._init_model_configurations(model, model_id)

        # Initialize parent Trainer class
        self._init_parent_trainer(
            model,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            callbacks,
            optimizers,
        )

        # Initialize swanlab if specified
        self._init_swanlab(args)

        # Finalize initialization (prepare models for training)
        self._finalize_initialization()

    def _init_default_args(self, model, args):
        """Initialize default GRPOConfig if not provided"""
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        return args

    def _init_models(self, model, args, attn_implementation):
        """Initialize main model and reference model"""
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

        if isinstance(model, str):
            model_id = model
            model, model_init_kwargs = self._load_model_from_string(
                model, model_id, model_init_kwargs, args
            )
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # Freeze modality encoders. Controlled per-encoder via env vars so the
        # downstream LM can optionally backprop into the audio encoder.
        # Defaults: vision frozen (we don't use images on AVQA),
        #           audio NOT frozen (training audio encoder is essential for
        #           real audio-understanding gains).
        freeze_visual = os.getenv("FREEZE_VISUAL", "true").lower() == "true"
        freeze_audio  = os.getenv("FREEZE_AUDIO",  "false").lower() == "true"
        # Optional: freeze the audio encoder backbone but keep the
        # audio_tower.proj (audio→LM hidden projector) trainable.
        freeze_audio_backbone_only = os.getenv("FREEZE_AUDIO_BACKBONE_ONLY", "false").lower() == "true"
        if hasattr(model, "visual") and freeze_visual:
            for param in model.visual.parameters():
                param.requires_grad = False
        if "Omni" in model_id and hasattr(model, "audio_tower"):
            if freeze_audio:
                for param in model.audio_tower.parameters():
                    param.requires_grad = False
            elif freeze_audio_backbone_only:
                for param in model.audio_tower.parameters():
                    param.requires_grad = False
                if hasattr(model.audio_tower, "proj"):
                    for param in model.audio_tower.proj.parameters():
                        param.requires_grad = True
        if "Omni" in model_id:
            n_audio_total = sum(p.numel() for p in model.audio_tower.parameters())
            n_audio_trainable = sum(p.numel() for p in model.audio_tower.parameters() if p.requires_grad)
            print(f"[encoder] visual frozen={freeze_visual}  audio frozen={freeze_audio}  "
                  f"backbone_only={freeze_audio_backbone_only}  "
                  f"(audio_tower trainable: {n_audio_trainable:,} / {n_audio_total:,})")

        # Initialize reference model
        self._init_reference_model(model_id, model_init_kwargs, args)

        return model, model_id, model_init_kwargs

    def _load_model_from_string(self, model, model_id, model_init_kwargs, args):
        """Load model from string path/name"""
        # Setup torch_dtype
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if (
            isinstance(torch_dtype, torch.dtype)
            or torch_dtype == "auto"
            or torch_dtype is None
        ):
            model_init_kwargs["torch_dtype"] = (
                torch.bfloat16 if args.bf16 else torch.float32
            )
        elif isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
            model_init_kwargs["torch_dtype"] = torch_dtype
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )

        # Disable caching if gradient checkpointing is enabled
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )

        # Load model based on type
        if "Qwen2-VL" in model_id:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model, **model_init_kwargs
            )
            self.model_type = "Qwen2-VL"
        elif "Qwen2.5-VL" in model_id:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model, **model_init_kwargs
            )
            self.model_type = "Qwen2.5-VL"
        elif "Aria" in model_id:
            model_init_kwargs.pop("use_cache")
            model = AriaForConditionalGeneration.from_pretrained(
                model, **model_init_kwargs
            )
            self.model_type = "Aria"
        elif "Omni" in model_id or os.getenv('USE_THINKER') == '1':
            model_init_kwargs.pop("use_cache")
            if os.getenv('USE_THINKER') == '1':
                model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2_5OmniModel.from_pretrained(model, **model_init_kwargs)
                model = model.thinker  # only the thinker part is used for GRPO
            self.model_type = "Qwen2.5-Omni"
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model, **model_init_kwargs
            )
            self.model_type = "Qwen2.5-VL"

        return model, model_init_kwargs

    def _init_reference_model(self, model_id, model_init_kwargs, args):
        """Initialize reference model if needed"""
        if args.beta > 0:
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Omni" in model_id or os.getenv('USE_THINKER') == '1':
                if os.getenv('USE_THINKER') == '1':
                    self.ref_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                else:
                    self.ref_model = Qwen2_5OmniModel.from_pretrained(
                        model_id, **model_init_kwargs
                    )
                    self.ref_model = self.ref_model.thinker
            else:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
        else:
            self.ref_model = None

    def _init_processing_class(
        self, processing_class, model_id, max_pixels, min_pixels
    ):
        """Initialize processing class (tokenizer/processor)"""
        if processing_class is None:
            if (
                "Qwen2-VL" in model_id
                or "Qwen2.5-VL" in model_id
                or "Aria" in model_id
                or True
            ):
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model_id, padding_side="left"
                )
                pad_token_id = processing_class.pad_token_id
        else:
            pad_token_id = processing_class.tokenizer.pad_token_id

        # Disable auto-resizing
        processing_class.do_resize = False
        processing_class.image_processor.do_resize = False

        return processing_class, pad_token_id

    def _init_reward_functions(
        self, reward_funcs, reward_processing_classes, model_init_kwargs
    ):
        """Initialize reward functions and their processing classes"""
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Initialize reward processing classes
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes

    def _init_training_config(self, args, script_args, pad_token_id):
        """Initialize training configuration and generation configs"""
        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.len_control = script_args.len_control
        self.beta = args.beta
        # Generation configs
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,
            temperature=1,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )

        if (
            os.getenv("FUSE_TEMPERATURE") is not None
            and os.getenv("FUSE_TEMPERATURE") == "true"
        ):
            self.audio_generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                top_p=0.95,
                temperature=float(os.getenv("AVS_TEMPERATURE", 1)),
                num_return_sequences=self.num_generations,
                pad_token_id=pad_token_id,
            )
        else:
            self.audio_generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                top_p=0.95,
                temperature=1,
                num_return_sequences=self.num_generations,
                pad_token_id=pad_token_id,
            )

        # Audio-SR1: stage-2 text-only rollout config
        self.use_sr1 = bool(getattr(script_args, "use_sr1", False))
        self.stage2_max_new_tokens = int(
            getattr(script_args, "stage2_max_new_tokens", 128)
        )

        # VPPO: audio-perception policy optimization
        self.use_vppo = bool(getattr(script_args, "use_vppo", False))
        self.vppo_top_p = float(getattr(script_args, "vppo_top_p", 0.2))
        self.vppo_adv_scale_min = float(getattr(script_args, "vppo_adv_scale_min", 0.8))
        # Uniform random time-frame masking ratio on mel features (direct audio
        # analog of VPPO image patch blackening). 0.6-0.8 recommended.
        self.vppo_mask_ratio = float(getattr(script_args, "vppo_mask_ratio", 0.7))
        # Entropy mask (OR'd with perception mask) — matches VPPO dp_actor.py:429
        self.use_vppo_on_entropy = bool(getattr(script_args, "use_vppo_on_entropy", False))
        self.top_p_entropy_tokens = float(getattr(script_args, "top_p_entropy_tokens", 0.2))
        # Answer-region protection: force the last N valid tokens (covering
        # </think><answer>X</answer>) to always be in the perception mask so
        # the final answer letter always receives policy gradient.
        self.vppo_answer_protect_tokens = int(getattr(script_args, "vppo_answer_protect_tokens", 0))
        # Mask annealing: schedule the perception mask strength over training.
        # "none" = constant VPPO mask (current behavior).
        # "linear" / "cosine" anneal effective mask 1→0, so loss transitions from
        # VPPO (early) to plain GRPO (late). Helps late-stage reasoning→answer
        # consistency while preserving early perception focus.
        self.mask_anneal_schedule = str(getattr(script_args, "mask_anneal_schedule", "none")).lower()
        if self.mask_anneal_schedule not in ("none", "linear", "cosine", "sigmoid"):
            raise ValueError(f"Unknown mask_anneal_schedule: {self.mask_anneal_schedule}")
        self.mask_anneal_warmup_steps = int(getattr(script_args, "mask_anneal_warmup_steps", 0))
        self.mask_anneal_lambda = float(getattr(script_args, "mask_anneal_lambda", 10.0))
        self.mask_anneal_gamma_frac = float(getattr(script_args, "mask_anneal_gamma_frac", 0.5))
        if self.use_vppo:
            msg = (
                f"[VPPO] top_p={self.vppo_top_p}, "
                f"adv_scale_min={self.vppo_adv_scale_min}, "
                f"mask_ratio={self.vppo_mask_ratio}"
            )
            if self.use_vppo_on_entropy:
                msg += f", entropy_top_p={self.top_p_entropy_tokens}"
            if self.mask_anneal_schedule != "none":
                msg += f", mask_anneal={self.mask_anneal_schedule}"
            if self.vppo_answer_protect_tokens > 0:
                msg += f", answer_protect={self.vppo_answer_protect_tokens}"
            print(msg)
        self.stage2_generation_config = GenerationConfig(
            max_new_tokens=self.stage2_max_new_tokens,
            do_sample=False,
            temperature=1.0,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        if self.use_sr1:
            print(
                f"[Audio-SR1] stage-2 rollout enabled "
                f"(max_new_tokens={self.stage2_max_new_tokens})"
            )

    def _init_model_configurations(self, model, model_id):
        """Initialize model-specific configurations"""
        # Suppress FLOPs warning
        model.warnings_issued["estimate_tokens"] = True

        # Initialize metrics
        self._metrics = defaultdict(list)

        # Freeze modality encoders. Controlled per-encoder via env vars; audio
        # defaults to TRAINABLE to match VPPO-RL / PAPO (their freeze_vision_tower
        # default is false — the perception-mask methods rely on encoder
        # gradients to reshape modality representations).
        freeze_visual = os.getenv("FREEZE_VISUAL", "true").lower() == "true"
        freeze_audio  = os.getenv("FREEZE_AUDIO",  "false").lower() == "true"
        freeze_audio_backbone_only = os.getenv("FREEZE_AUDIO_BACKBONE_ONLY", "false").lower() == "true"
        if hasattr(model, "visual") and freeze_visual:
            for param in model.visual.parameters():
                param.requires_grad = False
        if "Omni" in model_id and hasattr(model, "audio_tower"):
            if freeze_audio:
                for param in model.audio_tower.parameters():
                    param.requires_grad = False
            elif freeze_audio_backbone_only:
                for param in model.audio_tower.parameters():
                    param.requires_grad = False
                if hasattr(model.audio_tower, "proj"):
                    for param in model.audio_tower.proj.parameters():
                        param.requires_grad = True

    def _init_parent_trainer(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        processing_class,
        callbacks,
        optimizers,
    ):
        """Initialize parent Trainer class"""

        def data_collator(features):
            return features

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Set model accepts loss kwargs to enable loss scaling
        self.model_accepts_loss_kwargs = False

    def _init_swanlab(self, args):
        """Initialize swanlab if specified in report_to"""
        if _swanlab_available and hasattr(args, "report_to"):
            if isinstance(args.report_to, (list, tuple)):
                should_init_swanlab = (
                    "swanlab" in args.report_to or "wandb" in args.report_to
                )
            else:
                should_init_swanlab = args.report_to in ["swanlab", "wandb"]

            if should_init_swanlab and self.accelerator.is_main_process:
                # Initialize swanlab with project name and run name
                project_name = getattr(args, "project_name", "omni-r1-training")
                run_name = getattr(args, "run_name", None)

                swanlab.init(
                    project=project_name,
                    experiment_name=run_name,
                    config=args.to_dict() if hasattr(args, "to_dict") else None,
                )

    def _finalize_initialization(self):
        """Finalize initialization by preparing models"""
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True
                )

    def get_train_dataloader(self):
        if self.custom_train_dataloader is not None:
            return self.custom_train_dataloader

        return super().get_train_dataloader()

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, return_entropy=False, **kwargs):
        """Return per-token log-probs of target tokens. If return_entropy=True, also
        return per-token predictive entropy (shape [B, L-1]).

        Computed row-by-row to avoid the full [L-1, V] softmax OOM when V ~150k.
        """
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V) drop last logit (next-token unknown)
        input_ids = input_ids[:, 1:]  # (B, L-1) drop first id (no logit for it)

        per_token_logps = []
        per_token_entropy = [] if return_entropy else None
        for logits_row, input_ids_row in zip(logits, input_ids):
            # [L-1] logsumexp reduction over vocab
            lse = torch.logsumexp(logits_row, dim=1)
            # [L-1] logp of target token = logit - lse
            gathered = logits_row.gather(1, input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(gathered - lse)
            if return_entropy:
                # H[p] = -sum_v p_v log p_v = lse - sum_v p_v * logit_v
                # log_p = logit - lse ; p = exp(log_p)
                # H = -sum(p * log_p)
                log_p = logits_row - lse.unsqueeze(1)  # [L-1, V]
                p = log_p.exp()
                ent = -(p * log_p).sum(dim=1)  # [L-1]
                per_token_entropy.append(ent)
        logps = torch.stack(per_token_logps)
        if return_entropy:
            return logps, torch.stack(per_token_entropy)
        return logps

    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data

    def _augment_audio_features(self, input_features, feature_attention_mask=None):
        """Random time-frame patch masking on mel features — direct audio analog
        of VPPO's image patch blackening.

        input_features: [B, n_mels, T] tensor (Qwen2.5-Omni log-mel spectrogram)
        Returns: augmented tensor with `mask_ratio` fraction of time frames
        zeroed out, sampled UNIFORMLY at random (not contiguous). Uniform
        sampling gave a stronger KL signal than contiguous blocks in practice.
        """
        if input_features is None:
            return None
        x = input_features.clone()
        if x.dim() != 3:
            return x
        B, _, T = x.shape

        if self.vppo_mask_ratio <= 0 or T <= 1:
            return x

        num_mask = max(1, int(T * self.vppo_mask_ratio))
        for b in range(B):
            idx = torch.randperm(T, device=x.device)[:num_mask]
            x[b, :, idx] = 0.0

        return x

    def _compute_perception_signal(
        self, model, prompt_completion_ids, prompt_inputs,
        per_token_logps, completion_mask, prompt_length,
        per_token_entropy=None,
    ):
        """VPPO: compute per-token low-variance KL between clean and audio-perturbed forward.

        Returns:
          perception_mask: [B*G, T_comp] binary mask (1 = top-p% audio-critical tokens,
            OR-merged with top-p% entropy tokens if entropy provided)
          sensitivity_score: [B*G] average KL per response (for advantage shaping)
        """
        aug_features = self._augment_audio_features(
            prompt_inputs.get("input_features"),
            prompt_inputs.get("feature_attention_mask"),
        )
        aug_prompt_inputs = dict(prompt_inputs)
        if aug_features is not None:
            aug_prompt_inputs["input_features"] = aug_features

        # Temporarily switch model to eval mode so dropout doesn't add noise
        # to the clean-vs-aug KL signal. Restore afterward.
        was_training = model.training
        if was_training:
            model.eval()
        try:
            with torch.no_grad():
                aug_logps = self._get_per_token_logps(
                    model, prompt_completion_ids.detach(), **aug_prompt_inputs,
                )
            aug_logps = aug_logps[:, prompt_length - 1:]
        except Exception as e:
            print(f"[VPPO] aug forward failed ({e}); perception mask disabled this step")
            if was_training:
                model.train()
            return None, None
        if was_training:
            model.train()

        # low-variance KL per token: exp(d) - d - 1 where d = aug - clean
        # (matches VPPO/dp_actor.py:403-405)
        log_diff = (aug_logps - per_token_logps.detach()).clamp(-20.0, 20.0)
        low_var_kl = (log_diff.exp() - log_diff - 1.0).contiguous()
        low_var_kl = torch.clamp(low_var_kl, min=0.0, max=10.0)

        # Vectorized top-p% token selection (matches VPPO/dp_actor.py:407-426)
        low_var_kl_for_sort = low_var_kl.clone()
        invalid_mask = ~completion_mask.bool()
        low_var_kl_for_sort[invalid_mask] = -torch.inf

        num_valid_tokens = completion_mask.sum(dim=1)
        k = torch.ceil(num_valid_tokens.float() * self.vppo_top_p).int()

        # Sort descending, take top-k positions per response
        sorted_vals, sorted_indices = torch.sort(
            low_var_kl_for_sort, dim=1, descending=True
        )
        range_tensor = torch.arange(
            low_var_kl_for_sort.size(1), device=low_var_kl_for_sort.device
        ).expand_as(low_var_kl_for_sort)
        rank_mask = range_tensor < k.unsqueeze(1)

        # Scatter top-k mask back to original token positions
        perception_mask_bool = torch.zeros_like(low_var_kl_for_sort, dtype=torch.bool)
        perception_mask_bool.scatter_(1, sorted_indices, rank_mask)
        perception_mask_bool = perception_mask_bool & completion_mask.bool()

        # Entropy mask: OR with perception mask (matches VPPO dp_actor.py:429)
        if self.use_vppo_on_entropy and per_token_entropy is not None:
            entropy = per_token_entropy.detach()
            entropy_for_sort = entropy.clone()
            entropy_for_sort[invalid_mask] = -torch.inf
            k_ent = torch.ceil(num_valid_tokens.float() * self.top_p_entropy_tokens).int()
            _, ent_indices = torch.sort(entropy_for_sort, dim=1, descending=True)
            ent_rank_mask = range_tensor < k_ent.unsqueeze(1)
            entropy_mask_bool = torch.zeros_like(entropy_for_sort, dtype=torch.bool)
            entropy_mask_bool.scatter_(1, ent_indices, ent_rank_mask)
            entropy_mask_bool = entropy_mask_bool & completion_mask.bool()
            perception_mask_bool = perception_mask_bool | entropy_mask_bool

        # Answer-region carve-out: force the last N valid tokens per rollout to
        # always be in the mask so the final <answer>X</answer> token always
        # receives PG (fixes the Consistency-error regression the top-p cutoff
        # otherwise introduces when the answer token's KL happens to rank low).
        if self.vppo_answer_protect_tokens > 0:
            n_protect = int(self.vppo_answer_protect_tokens)
            valid_lens = completion_mask.sum(dim=1).long()  # [B*G]
            T = perception_mask_bool.size(1)
            pos = torch.arange(T, device=perception_mask_bool.device).unsqueeze(0)
            end = valid_lens.unsqueeze(1)
            start = (end - n_protect).clamp(min=0)
            answer_region = (pos >= start) & (pos < end)
            perception_mask_bool = perception_mask_bool | answer_region

        perception_mask = perception_mask_bool.to(low_var_kl.dtype)

        # Sensitivity score (mean KL over valid tokens, per response) for TAS
        low_var_kl_masked = low_var_kl * completion_mask
        n_valid_safe = num_valid_tokens.float().clamp(min=1.0)
        sensitivity_score = (low_var_kl_masked.sum(dim=1) / n_valid_safe).detach()

        return perception_mask, sensitivity_score

    def _run_stage2_rollout(
        self,
        completion_texts: list,
        inputs: list,
        unwrapped_model,
        device,
    ) -> list:
        """Audio-SR1 stage-2 rollout: text-only answer using extracted <description>.

        For each stage-1 completion, extract <description>...</description>, build
        a text-only prompt (no audio features), and generate a short answer.
        Returns a list of stage-2 decoded texts, aligned 1-1 with completion_texts.

        The model is the SAME policy as stage-1 — gradients don't flow through
        stage-2 (we call model.generate under no_grad). Stage-2 accuracy enters
        the advantage via the reward function only.
        """
        n = len(completion_texts)
        stage2_texts = ["" for _ in range(n)]
        B = len(inputs)
        G = n // B if B > 0 else n

        # Build text-only messages, one per completion.
        messages_list = []
        for idx, comp in enumerate(completion_texts):
            # Extract <description> text. If the tag is missing or empty, the
            # stage-2 prompt will carry "(none)" — a model that skipped the
            # description can't reconstruct evidence and will score poorly.
            m = DESCRIPTION_RE.search(comp or "")
            description = m.group(1).strip() if m and m.group(1).strip() else "(none)"

            # Map rollout idx back to original sample.
            # `repeat_reward_kwargs` layout: outer num_generations, inner B
            # → rollout idx i corresponds to sample i % B.
            sample_i = idx % B if B > 0 else 0
            problem_text = inputs[sample_i].get("problem", "")

            user_text = STAGE2_USER_TEMPLATE.format(
                description=description,
                Question=problem_text,
            )
            messages_list.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "text", "text": user_text}]},
                ]
            )

        # Build text prompts.
        prompts_text = [
            maybe_apply_chat_template({"prompt": msgs}, self.processing_class)["prompt"]
            for msgs in messages_list
        ]
        prompts_text = [p[0] if isinstance(p, list) else p for p in prompts_text]

        # Tokenize (text-only — no audio/image/video).
        stage2_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=None,
            audios=None,
            videos=None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        stage2_inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in stage2_inputs.items()
        }
        # Strip any multimodal / use_audio flags (text-only).
        for k in [
            "pixel_values", "image_grid_thw",
            "pixel_values_videos", "video_grid_thw",
            "input_features", "feature_attention_mask",
            "use_audio_in_video", "second_per_grid_ts",
        ]:
            stage2_inputs.pop(k, None)

        prompt_len = stage2_inputs["input_ids"].size(1)

        try:
            with torch.inference_mode():
                out = unwrapped_model.generate(
                    **stage2_inputs,
                    generation_config=self.stage2_generation_config,
                )
            gen_ids = out[:, prompt_len:]
            decoded = self.processing_class.batch_decode(
                gen_ids, skip_special_tokens=True
            )
            for i, text in enumerate(decoded):
                stage2_texts[i] = text
        except Exception as e:
            print(f"[Audio-SR1] stage-2 generation failed: {e}")

        return stage2_texts

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def batch_reff(self, list_inputs):
        culen = [len(inputs) for inputs in list_inputs]
        flatten_inputs = []
        for inputs in list_inputs:
            flatten_inputs.extend(inputs)

        dummy_prompt = None
        # if 'Omni' in self.model_type:
        #     print("adding dummy audio")
        #     dummy_prompt = [{
        #         "prompt": [{
        #                     "role": "user",
        #                     "content": [
        #                         {
        #                             "type": "audio",
        #                             "audio": 'file:///mnt/public/dataset/omni/REFAVS/media/zvjPsMpbnac_0_10000/audio.wav',
        #                         },
        #                         {
        #                             "type": "text",
        #                             "text": 'Hello! Just say goodbye to me, please.' # 'Answer me anything you what about the picture.'
        #                         }
        #                     ]
        #         }]
        #     }]

        if dummy_prompt is not None:
            flatten_inputs.extend(dummy_prompt)

        completions, data_process_kwargs = self.reff(flatten_inputs)
        cuidx = 0
        list_completions, list_kwargs = [], []

        for i, lens in enumerate(culen):
            list_completions.append(completions[cuidx : cuidx + lens])
            list_kwargs.append(data_process_kwargs[cuidx : cuidx + lens])
            cuidx += lens

        if dummy_prompt is not None:
            assert len(list_completions) == len(list_inputs), (
                f"error, list_completions {len(list_completions)} != list_inputs {len(list_inputs)}"
            )

        return list_completions, list_kwargs

    def reff(self, inputs):
        def has_audio_modal(content_list):
            """
            判断多模态输入中是否包含 audio 类型
            :param content_list: 输入列表，例如 input_copy_i[0]['content']
            :return: True if contains audio, else False
            """
            return any(item.get("type") == "audio" for item in content_list)

        for i, input_i in enumerate(inputs):
            inputs[i]["prompt"] = self.remove_none_from_data(input_i["prompt"])

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        # 创建输入的深拷贝以避免修改原始数据
        input_copy = [copy.deepcopy(inputs[i]["prompt"]) for i in range(len(inputs))]

        image_inputs, video_inputs, data_process_kwargs = [], [], []
        audio_inputs = []
        for input_copy_i in input_copy:
            try:
                if "Omni" not in self.model_type:
                    image_input, video_input, kwarg = process_vision_info(input_copy_i)
                    audio_input = None
                    # print(f'video_inputs: {video_input[0].shape}')
                else:
                    _user_content = next(
                        (m["content"] for m in input_copy_i if m.get("role") == "user"),
                        input_copy_i[0]["content"],
                    )
                    audio_input, image_input, video_input, kwarg = process_mm_info(
                        input_copy_i,
                        use_audio_in_video=has_audio_modal(_user_content),
                    )

            except Exception as e:
                print(f"process_vision_info error, waiting debugging signal, {e}")
                wait_for_debugger_if_rank0()
                raise e

            if image_input is not None:
                image_inputs.extend(image_input)
            if video_input is not None:
                video_inputs.extend(video_input)
            if audio_input is not None:
                audio_inputs.extend(audio_input)
            data_process_kwargs.append(kwarg)

        # with audio inputs
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(
                [pt[0] for pt in prompts_text]
            ),  # deepcopy is important, otherwise the text will be changed
            images=image_inputs if len(image_inputs) > 0 else None,
            audios=audio_inputs if len(audio_inputs) > 0 else None,
            videos=video_inputs if len(video_inputs) > 0 else None,
            use_audio_in_video=has_audio_modal(next(
                (m["content"] for m in input_copy_i if m.get("role") == "user"),
                input_copy_i[0]["content"],
            )),
            return_tensors="pt",
            padding=True,
            padding_side="left",
            do_resize=False,
            add_special_tokens=False,
        )
        if len(audio_inputs) > 0:
            prompt_inputs["use_audio_in_video"] = True
        else:
            prompt_inputs["use_audio_in_video"] = False

        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids = prompt_inputs["input_ids"]

        generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=self.processing_class.tokenizer.pad_token_id,
        )
        # print(f'Rank:{torch.distributed.get_rank()}, starting ref model generation...')

        with torch.no_grad():
            # if os.getenv("SELF_GROUND") == 'true':
            #     print("Using self-grounding")
            #     with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            #         prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=generation_config)
            if self.ref_model is not None:
                with unwrap_model_for_generation(
                    self.ref_model, self.accelerator
                ) as unwrapped_model:
                    # unwrapped_model = self.accelerator.unwrap_model(self.ref_model)
                    # print(f'Rank:{torch.distributed.get_rank()}, unwrapping ref model completed.')
                    prompt_completion_ids = unwrapped_model.generate(
                        **prompt_inputs, generation_config=generation_config
                    )
                    # print(f'Rank:{torch.distributed.get_rank()}, ref model generation completed.')
            else:
                # Model doesn't have reference model, use current model directly
                with unwrap_model_for_generation(
                    self.model, self.accelerator
                ) as unwrapped_model:
                    prompt_completion_ids = unwrapped_model.generate(
                        **prompt_inputs, generation_config=generation_config
                    )

        # Ensure generated tokens are detached from inference context
        prompt_completion_ids = prompt_completion_ids.detach()
        # print("ref model generated!")
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        # Decode the generated completions
        # prompts = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        return completions, data_process_kwargs

    def server_reff(
        self, inputs, server_url="http://localhost:12428", api_key=None, timeout=300
    ):
        """
        通过VLLM服务器获取推理结果的函数

        Args:
            inputs: 输入数据列表
            server_url: VLLM服务器地址
            api_key: API密钥（可选）
            timeout: 请求超时时间（秒）

        Returns:
            completions: 生成的文本列表
            data_process_kwargs: 处理参数列表
        """

        def has_audio_modal(content_list):
            """
            判断多模态输入中是否包含 audio 类型
            """
            return any(item.get("type") == "audio" for item in content_list)

        def encode_image_to_base64(image_path):
            """
            将图片文件编码为base64字符串
            """
            try:
                if isinstance(image_path, str) and image_path.startswith("file://"):
                    image_path = image_path[7:]
                    with open(image_path, "rb") as image_file:
                        image_data = image_file.read()
                        base64_str = base64.b64encode(image_data).decode("utf-8")
                        return base64_str

                else:
                    # image_path is a PIL.Image object
                    # If image_path is a PIL.Image object, convert it to base64

                    # Convert PIL image to base64
                    buffer = io.BytesIO()
                    image_path.save(buffer, format="JPEG")
                    image_data = buffer.getvalue()
                    base64_str = base64.b64encode(image_data).decode("utf-8")
                    return base64_str

            except Exception as e:
                print(f"Error encoding image {image_path}: {e}")
                return None

        def encode_video_to_base64(video_path):
            """
            将视频文件编码为base64字符串
            """
            try:
                if video_path.startswith("file://"):
                    video_path = video_path[7:]

                with open(video_path, "rb") as video_file:
                    video_data = video_file.read()
                    base64_str = base64.b64encode(video_data).decode("utf-8")
                    return base64_str
            except Exception as e:
                print(f"Error encoding video {video_path}: {e}")
                return None

        def encode_audio_to_base64(audio_path):
            """
            将音频文件编码为base64字符串
            """
            try:
                if audio_path.startswith("file://"):
                    audio_path = audio_path[7:]

                with open(audio_path, "rb") as audio_file:
                    audio_data = audio_file.read()
                    base64_str = base64.b64encode(audio_data).decode("utf-8")
                    return base64_str
            except Exception as e:
                print(f"Error encoding audio {audio_path}: {e}")
                return None

        def create_openai_formatter():
            """
            将内部格式转换为OpenAI API格式
            """
            ptrs = [0, 0, 0]

            def _convert_to_openai_format(prompt, m_data):
                messages = []

                for message in prompt:
                    role = message.get("role", "user")
                    content = message.get("content", [])

                    if isinstance(content, str):
                        # 纯文本消息
                        messages.append({"role": role, "content": content})
                    elif isinstance(content, list):
                        # 多模态消息
                        openai_content = []

                        for item in content:
                            item_type = item.get("type", "text")

                            if item_type == "text":
                                openai_content.append(
                                    {"type": "text", "text": item.get("text", "")}
                                )
                            elif item_type == "image":
                                # image_path = item.get("image", "")
                                image = m_data[0][ptrs[0]]
                                ptrs[0] = ptrs[0] + 1
                                base64_image = encode_image_to_base64(image)
                                if base64_image:
                                    openai_content.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{base64_image}"
                                            },
                                        }
                                    )
                            else:
                                raise ValueError(
                                    "vLLM currently only supports texts and images"
                                )
                            # elif item_type == "video":
                            #     video_path = item.get("video", "")
                            #     base64_video = encode_video_to_base64(video_path)
                            #     if base64_video:
                            #         openai_content.append(
                            #             {
                            #                 "type": "video",
                            #                 "video": f"data:video/mp4;base64,{base64_video}",
                            #             }
                            #         )
                            # elif item_type == "audio":
                            #     audio_path = item.get("audio", "")
                            #     base64_audio = encode_audio_to_base64(audio_path)
                            #     if base64_audio:
                            #         openai_content.append(
                            #             {
                            #                 "type": "audio",
                            #                 "audio": f"data:audio/wav;base64,{base64_audio}",
                            #             }
                            #         )

                        messages.append({"role": role, "content": openai_content})

                return messages

            return _convert_to_openai_format

        # 处理输入数据
        for i, input_i in enumerate(inputs):
            inputs[i]["prompt"] = self.remove_none_from_data(input_i["prompt"])

        input_copy = [copy.deepcopy(input["prompt"]) for input in inputs]

        data_process_kwargs = []
        image_inputs, audio_inputs, video_inputs = [], [], []

        for input_copy_i in input_copy:
            if "Omni" not in self.model_type:
                image_input, video_input, kwarg = process_vision_info(input_copy_i)
                audio_input = None
                # print(f'video_inputs: {video_input[0].shape}')
            else:
                _user_content = next(
                    (m["content"] for m in input_copy_i if m.get("role") == "user"),
                    input_copy_i[0]["content"],
                )
                audio_input, image_input, video_input, kwarg = process_mm_info(
                    input_copy_i,
                    use_audio_in_video=has_audio_modal(_user_content),
                )

            if image_input:
                image_inputs.extend(image_input)
            if audio_input:
                audio_inputs.extend(audio_input)
            if video_input:
                video_inputs.extend(video_input)

            data_process_kwargs.append(kwarg)
        mm_data = [image_inputs, audio_inputs, video_inputs]

        converter = create_openai_formatter()
        # 创建处理参数
        # data_process_kwargs = []
        completions = []

        # 构建请求头
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # 为每个输入生成响应
        for input_data in inputs:
            try:
                prompt = input_data["prompt"]

                # 转换为OpenAI格式
                messages = converter(prompt, mm_data)

                # 构建请求数据
                request_data = {
                    "model": MODEL,  # VLLM服务器默认模型
                    "messages": messages,
                    "max_tokens": self.max_completion_length,
                    "temperature": 0.0,  # 保持与原始reff函数一致的do_sample=False
                    "top_p": 1.0,
                    "stream": False,
                }

                # 发送请求
                response = requests.post(
                    f"{server_url}/v1/chat/completions",
                    headers=headers,
                    json=request_data,
                    timeout=timeout,
                )

                if response.status_code == 200:
                    result = response.json()
                    completion_text = result["choices"][0]["message"]["content"]
                    completions.append(completion_text)
                else:
                    print(
                        f"Server request failed with status {response.status_code}: {response.text}"
                    )
                    completions.append("")  # 失败时返回空字符串

                # # 创建模拟的处理参数
                # # 这里需要根据实际的process_vision_info或process_mm_info的返回值来构造
                # if "Omni" not in self.model_type:
                #     # 模拟process_vision_info的返回
                #     kwarg = {
                #         "has_image": any(
                #             item.get("type") == "image"
                #             for message in prompt
                #             for item in (
                #                 message.get("content", [])
                #                 if isinstance(message.get("content"), list)
                #                 else []
                #             )
                #         ),
                #         "has_video": any(
                #             item.get("type") == "video"
                #             for message in prompt
                #             for item in (
                #                 message.get("content", [])
                #                 if isinstance(message.get("content"), list)
                #                 else []
                #             )
                #         ),
                #     }
                # else:
                #     # 模拟process_mm_info的返回
                #     kwarg = {
                #         "has_image": any(
                #             item.get("type") == "image"
                #             for message in prompt
                #             for item in (
                #                 message.get("content", [])
                #                 if isinstance(message.get("content"), list)
                #                 else []
                #             )
                #         ),
                #         "has_video": any(
                #             item.get("type") == "video"
                #             for message in prompt
                #             for item in (
                #                 message.get("content", [])
                #                 if isinstance(message.get("content"), list)
                #                 else []
                #             )
                #         ),
                #         "has_audio": any(
                #             item.get("type") == "audio"
                #             for message in prompt
                #             for item in (
                #                 message.get("content", [])
                #                 if isinstance(message.get("content"), list)
                #                 else []
                #             )
                #         ),
                #     }

                # data_process_kwargs.append(kwarg)

            except Exception as e:
                print(f"Error processing input: {e}")
                completions.append("")  # 出错时返回空字符串
                data_process_kwargs.append({})  # 出错时返回空字典

        return completions, data_process_kwargs

    def batch_server_reff(
        self,
        list_inputs,
        server_url=VLLM_SERVER_PATH,
        api_key=None,
        timeout=300,
    ):
        """
        使用服务器的batch_reff版本

        Args:
            list_inputs: 输入列表的列表
            server_url: VLLM服务器地址
            api_key: API密钥（可选）
            timeout: 请求超时时间（秒）

        Returns:
            list_completions: 完成文本列表的列表
            list_kwargs: 处理参数列表的列表
        """
        culen = [len(inputs) for inputs in list_inputs]
        flatten_inputs = []
        for inputs in list_inputs:
            flatten_inputs.extend(inputs)

        completions, data_process_kwargs = self.server_reff(
            flatten_inputs, server_url=server_url, api_key=api_key, timeout=timeout
        )

        cuidx = 0
        list_completions, list_kwargs = [], []

        for lens in culen:
            list_completions.append(completions[cuidx : cuidx + lens])
            list_kwargs.append(data_process_kwargs[cuidx : cuidx + lens])
            cuidx += lens

        assert len(list_completions) == len(list_inputs), (
            f"error, list_completions {len(list_completions)} != list_inputs {len(list_inputs)}"
        )

        # torch.distributed.barrier()

        return list_completions, list_kwargs

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        def interleave2repeat(tensor, batch_size, num_generations):
            # Reshape the tensor to (num_return_sequences, batch_size, sequence_length)
            outputs = tensor.view(num_generations, batch_size, -1)
            # Permute the dimensions to (batch_size, num_return_sequences, sequence_length)
            outputs = outputs.permute(1, 0, 2)
            # Reshape back to (batch_size * num_return_sequences, sequence_length)
            outputs = outputs.contiguous().view(batch_size * num_generations, -1)
            return outputs

        def repeat_reward_kwargs(prompts, inputs_list, num_generations):
            reward_kwargs = []  # {key: [] for key in inputs_list[i].keys() if key not in ["prompt", "completion"]}
            exclusion = ["prompt", "completion"]
            for _ in range(num_generations):
                for i, example in enumerate(inputs_list):
                    reward_kwarg = dict()
                    for k in example.keys():
                        if k not in exclusion:
                            reward_kwarg[k] = example[k]
                    # using append to avoid str auto-converting to char list.
                    reward_kwarg["preprocess_kwargs"] = data_process_kwargs[i]
                    reward_kwarg["prompt"] = prompts[i]
                    reward_kwargs.append(reward_kwarg)
            # list of original reward_kwargs
            return reward_kwargs

        def has_audio_modal(content_list):
            """
            判断多模态输入中是否包含 audio 类型
            :param content_list: 输入列表，例如 input_copy_i[0]['content']
            :return: True if contains audio, else False
            """
            return any(item.get("type") == "audio" for item in content_list)

        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # when batch_size > 1, temporal_reward need to filter those not belonging to video and only compare between video subgroup.
        # video_inputs_selected = [input_i if input_i['data_type'] == 'video' else None for input_i in inputs]

        wait_for_debugger_if_rank0()

        dataset_type = inputs[0]["problem_type"]

        for i, input_i in enumerate(inputs):
            inputs[i]["prompt"] = self.remove_none_from_data(input_i["prompt"])

        prompts = [x["prompt"] for x in inputs]

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]  # TODO: check omni

        # 创建输入的深拷贝以避免修改原始数据
        input_copy = [copy.deepcopy(inputs[i]["prompt"]) for i in range(len(inputs))]
        input_copy = [self.remove_none_from_data(input_i) for input_i in input_copy]

        image_inputs, video_inputs, data_process_kwargs = [], [], []
        audio_inputs = []
        for input_copy_i in input_copy:
            try:

                _user_content = next(
                    (m["content"] for m in input_copy_i if m.get("role") == "user"),
                    input_copy_i[0]["content"],
                )
                audio_input, image_input, video_input, kwarg = process_mm_info(
                    input_copy_i,
                    use_audio_in_video=has_audio_modal(_user_content),
                )

            except Exception as e:
                print(f"process_vision_info error, waiting debugging signal, {e}")
                wait_for_debugger_if_rank0()
                raise e
            if image_input is not None:
                image_inputs.extend(image_input)
            if video_input is not None:
                video_inputs.extend(video_input)
            if audio_input is not None:
                audio_inputs.extend(audio_input)
            data_process_kwargs.append(kwarg)

        print(f"当前输入情况：{data_process_kwargs}")

        # with audio inputs
        prompts_text = [
            p[0] for p in prompts_text
        ]  # [0]  # omni processor return [str]
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(
                prompts_text
            ),  # deepcopy is important, otherwise the text will be changed
            images=image_inputs if len(image_inputs) > 0 else None,
            audios=audio_inputs if len(audio_inputs) > 0 else None,
            videos=video_inputs if len(video_inputs) > 0 else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            do_resize=False,
            add_special_tokens=False,
        )
        if len(audio_inputs) > 0:
            prompt_inputs["use_audio_in_video"] = True
        else:
            prompt_inputs["use_audio_in_video"] = False

        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        # check decoded decoded_text=self.processing_class.tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
        if self.max_prompt_length is not None:
            assert prompt_ids.size(1) <= self.max_prompt_length, (
                f"Prompt length {prompt_ids.size(1)} exceeds max prompt length {self.max_prompt_length}"
            )
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if "refavs" in dataset_type:
            generation_config = self.audio_generation_config
        else:
            generation_config = self.generation_config

        # Generate completions
        device = self.accelerator.device

        with unwrap_model_for_generation(
            model, self.accelerator, gather_deepspeed3_params=True
        ) as unwrapped_model:
            gen_kwargs = dict(**prompt_inputs, generation_config=generation_config)

            prompt_completion_ids = unwrapped_model.generate(**gen_kwargs)
            prompt_completion_ids = interleave2repeat(
                prompt_completion_ids, len(inputs), self.num_generations
            )
            prompt_length = prompt_ids.size(1)

            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat(self.num_generations, 1)

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full(
                (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
            )
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
                is_eos.size(0), -1
            )
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")

        repeat_keys = [
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        ]
        for key in repeat_keys:
            if key in prompt_inputs:
                prompt_inputs[key] = prompt_inputs[key].repeat(self.num_generations, 1)
            # else:
            #     print(f"Warning: key not found in prompt_inputs: {key}, current keys: {prompt_inputs.keys()}")
        if "second_per_grid_ts" in prompt_inputs:
            del prompt_inputs["second_per_grid_ts"]

        per_token_entropy = None  # [B*G, T_comp] if VPPO+entropy enabled
        try:
            if self.use_vppo and self.use_vppo_on_entropy:
                per_token_logps, full_entropy = self._get_per_token_logps(
                    model, prompt_completion_ids, return_entropy=True, **prompt_inputs
                )
                per_token_entropy = full_entropy[:, prompt_length - 1 :].detach()
            else:
                per_token_logps = self._get_per_token_logps(
                    model, prompt_completion_ids, **prompt_inputs
                )
            original_token_logps = per_token_logps.detach().clone()
            per_token_logps = per_token_logps[:, prompt_length - 1 :]
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            raise e

        with torch.no_grad():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        prompt_completion_ids.detach(),
                        **{
                            k: v.detach() if isinstance(v, torch.Tensor) else v
                            for k, v in prompt_inputs.items()
                        },
                    )[:, prompt_length - 1 :]
                else:
                    # No reference model, use original logps
                    ref_per_token_logps = original_token_logps[:, prompt_length - 1 :]
            except Exception as e:
                print(
                    f"Error computing ref_per_token_logps: {e}. Setting output to zero."
                )
                raise e

        # Compute the KL divergence between the model and the reference model
        # Ensure ref_per_token_logps is detached to prevent gradient issues
        ref_per_token_logps = ref_per_token_logps.detach()

        # FIXME: KL clamp is not tested
        #
        # x_clamped = torch.clamp(
        #     ref_per_token_logps - per_token_logps, min=-10, max=10
        # )  # 限制 x 的范围
        # per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        #

        x_ratio = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(x_ratio) - x_ratio - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        # Keep the raw text list for stage-2 description extraction before we
        # wrap completions in conversational format.
        completion_texts = list(completions)
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        # Audio-SR1: stage-2 text-only rollout using extracted <description>.
        if self.use_sr1:
            with unwrap_model_for_generation(
                model, self.accelerator, gather_deepspeed3_params=True
            ) as s2_unwrapped:
                stage2_texts = self._run_stage2_rollout(
                    completion_texts, inputs, s2_unwrapped, device,
                )
        else:
            stage2_texts = [""] * (len(inputs) * self.num_generations)

        # Compute the rewards
        # prompts = [prompt for _ in range(self.num_generations) for prompt in prompts]
        rewards_per_func = torch.zeros(
            len(prompts) * self.num_generations, len(self.reward_funcs), device=device
        )

        batch_ref = partial(
            self.batch_reff,
        )
        
        if USE_VLLM:
            
            batch_ref = partial(
                self.batch_server_reff,
            )

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = repeat_reward_kwargs(prompts, inputs, self.num_generations)
            # Audio-SR1: attach stage-2 completion text per rollout.
            for j, kw in enumerate(reward_kwargs):
                kw["stage2_completion"] = stage2_texts[j] if j < len(stage2_texts) else ""

            output_reward_func = reward_func(
                completions=completions,
                kwargs_list=reward_kwargs,
                ref_generate=batch_ref,
            )
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        if self.len_control:
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
            #             if len(selected_indices) > 1 and len(selected_indices) < self.num_generations:
            # if len(selected_indices) > 1:
            #     selected_items = [(i, lenth_list[i]) for i in selected_indices]
            #     sorted_items = sorted(selected_items, key=lambda x: x[1], reverse=True)
            #     N = len(sorted_items)
            #     for rank, (idx, length) in enumerate(sorted_items):
            #         reward = 0.2 - 0.2 * (rank / N)
            #         rewards[idx] += reward
            #         mem_rewards[idx] = reward
            # for idx in range(len(lenth_list)):
            #     if lenth_list[idx] >= 512:
            #         rewards[idx] -= 0.5

            if len(selected_indices) > 1:
                for idx in selected_indices:
                    if 320 <= lenth_list[idx] <= 750:
                        rewards[idx] += (
                            1  # changed only on 512 morning after vl start training
                        )

        # ---- Detailed per-rollout reward log (easy to extract for plotting) ----
        G = self.num_generations
        B = rewards.size(0) // G
        rpf = rewards_per_func  # [B*G, n_funcs]
        func_names = []
        for rf in self.reward_funcs:
            if hasattr(rf, "__name__"):
                func_names.append(rf.__name__)
            elif hasattr(rf, "func"):
                func_names.append(rf.func.__name__)
            else:
                func_names.append(str(rf))
        for g in range(B):
            s, e = g * G, (g + 1) * G
            parts = []
            for fi, fn in enumerate(func_names):
                vals = rpf[s:e, fi].tolist()
                parts.append(f"{fn}=[{','.join(f'{v:.3f}' for v in vals)}]")
            total_vals = rewards[s:e].tolist()
            lens = completion_mask[s:e].sum(1).tolist()
            parts.append(f"total=[{','.join(f'{v:.3f}' for v in total_vals)}]")
            parts.append(f"len=[{','.join(str(int(l)) for l in lens)}]")
            print(f"[reward-detail] group={g} " + " ".join(parts))
        # ---- end detail log ----

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # VPPO: compute perception mask (micro) and sensitivity score (macro)
        perception_mask = None
        if self.use_vppo and prompt_inputs.get("input_features") is not None:
            perception_mask, sensitivity_score = self._compute_perception_signal(
                model, prompt_completion_ids, prompt_inputs,
                per_token_logps, completion_mask, prompt_length,
                per_token_entropy=per_token_entropy,
            )
            # Macro: scale advantages by response-level sensitivity
            if (sensitivity_score is not None and self.vppo_adv_scale_min < 1.0):
                s = sensitivity_score
                s_min = s.min()
                s_max = s.max()
                if (s_max - s_min) > 1e-8:
                    s_norm = (s - s_min) / (s_max - s_min)  # [0,1]
                else:
                    s_norm = torch.zeros_like(s)
                scale = self.vppo_adv_scale_min + s_norm * (1.0 - self.vppo_adv_scale_min)
                advantages = advantages * scale
                # Log
                self._metrics.setdefault("vppo_sens_mean", []).append(
                    s.mean().item()
                )
                self._metrics.setdefault("vppo_scale_mean", []).append(
                    scale.mean().item()
                )

        # x - x.detach() allows for preserving gradients from x
        # Policy gradient term (maximize reward)
        pg_term = -torch.exp(
            per_token_logps - per_token_logps.detach()
        ) * advantages.unsqueeze(1)
        # KL regularization term (minimize divergence from reference)
        kl_term = self.beta * per_token_kl

        if perception_mask is not None:
            # Anneal alpha: 1 → 0 over training. alpha=1 → current VPPO;
            # alpha=0 → plain GRPO (effective perception mask = 1 everywhere).
            if self.mask_anneal_schedule in ("linear", "cosine", "sigmoid"):
                max_steps = max(1, getattr(self.args, "max_steps", 1000))
                warmup = min(self.mask_anneal_warmup_steps, max_steps - 1)
                step = self.state.global_step
                if step < warmup:
                    alpha = 1.0
                else:
                    t = min(1.0, (step - warmup) / max(1, max_steps - warmup))
                    if self.mask_anneal_schedule == "linear":
                        alpha = max(0.0, 1.0 - t)
                    elif self.mask_anneal_schedule == "cosine":
                        import math
                        alpha = 0.5 * (1.0 + math.cos(math.pi * t))
                    else:  # sigmoid: alpha(t) = 1/(1+exp(lambda*(t-gamma_frac)))
                        import math
                        alpha = 1.0 / (1.0 + math.exp(self.mask_anneal_lambda * (t - self.mask_anneal_gamma_frac)))
            else:
                alpha = 1.0
            effective_perception = alpha * perception_mask + (1.0 - alpha)

            # VPPO: apply (annealed) perception mask ONLY to policy gradient,
            # keep KL on all valid tokens (masking KL would let non-critical
            # tokens drift from reference → catastrophic forgetting).
            pg_mask = completion_mask * effective_perception
            pg_loss = (
                (pg_term * pg_mask).sum(dim=1) / pg_mask.sum(dim=1).clamp(min=1)
            )
            kl_loss = (
                (kl_term * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
            )
            loss = (pg_loss + kl_loss).mean()
            self._metrics.setdefault("vppo_mask_frac", []).append(
                (perception_mask * completion_mask).sum().item()
                / completion_mask.sum().clamp(min=1).item()
            )
            self._metrics.setdefault("vppo_anneal_alpha", []).append(alpha)
        else:
            per_token_loss = pg_term + kl_term
            loss = (
                (per_token_loss * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1).clamp(min=1)
            ).mean()

        # Log the metrics
        completion_length = (
            self.accelerator.gather(completion_mask.sum(1)).float().mean().item()
        )
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                if hasattr(reward_func, "__name__"):
                    reward_func_name = reward_func.__name__
                else:
                    # For partial functions, use the underlying function's name if available
                    reward_func_name = reward_func.func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )

        gathered_rewards = self.accelerator.gather(rewards)

        num_devices = gathered_rewards.size(0) // self.num_generations
        rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices

        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices

        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)

        self._metrics["reward"].append(self.accelerator.gather(rewards).mean().item())

        self._metrics["reward_std"].append(
            self.accelerator.gather(std_grouped_rewards).mean().item()
        )

        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)
        ).mean()
        self._metrics["kl"].append(self.accelerator.gather(mean_kl).mean().item())

        return loss


    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics.items()
        }  # average the metrics
        logs = {**logs, **metrics}

        # Log to swanlab if available and initialized
        if _swanlab_available and self.accelerator.is_main_process:
            try:
                swanlab.log(logs)
            except Exception:
                # If swanlab is not initialized or fails, continue silently
                pass

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
