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
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from trl import GRPOConfig, ModelConfig, TrlParser

try:
    import swanlab

    _swanlab_available = True
except ImportError:
    _swanlab_available = False

from omni_r1.debug_utils import wait_for_debugger_if_rank0
from omni_r1.load_datasets import AVQADataset
from omni_r1.reward import accuracy_reward, format_reward, stage2_accuracy_reward, overthinking_penalty_reward
from omni_r1.rubric_reward import rubric_reward
from omni_r1.evolving_rubric_reward import evolving_rubric_reward
from omni_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer
from omni_r1.trainer.sampler import GroupSampler


@dataclass
class GRPOScriptArguments:
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'"
        },
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    datasets_json: Optional[str] = field(
        default="datasets.json",
        metadata={"help": "the path_name of the json specifying datasets' path"},
    )
    training_datasets: list[str] = field(
        default_factory=lambda: ["ReVOS"],
        metadata={"help": "the datasets used for training"},
    )
    use_prompt_forcing: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using prompt forcing for VOS dataset"},
    )
    use_multi_vos: Optional[bool] = field(
        default=False,
        metadata={"help": "whether re-using multi-obj for VOS dataset"},
    )
    use_avs_sample_number: Optional[int] = field(
        default=1600,
        metadata={
            "help": "the number of samples to use for AVS dataset, 0 means using all samples"
        },
    )
    alpha_k: Optional[float] = field(
        default=1.0,
        metadata={"help": "the weight of R_k reward"},
    )
    alpha_k_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "the weight of mask sum quality in R_k reward"},
    )
    alpha_a: Optional[float] = field(
        default=1.0,
        metadata={"help": "the weight of R_a reward"},
    )
    alpha_g: Optional[float] = field(
        default=1.0,
        metadata={"help": "the weight of R_g reward"},
    )
    use_sr1: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable Audio-SR1: two-stage rollout (stage 1 with audio, "
            "stage 2 with extracted <description> only, no audio). Requires "
            "stage2_accuracy in reward_funcs."
        },
    )
    stage2_max_new_tokens: Optional[int] = field(
        default=128,
        metadata={"help": "Max new tokens for the stage-2 (text-only) rollout"},
    )
    stage1_acc_weight: Optional[float] = field(
        default=0.9,
        metadata={"help": "Weight for stage-1 accuracy reward (Vision-SR1 default 0.9)"},
    )
    stage1_format_weight: Optional[float] = field(
        default=0.1,
        metadata={"help": "Weight for stage-1 format reward (Vision-SR1 default 0.1)"},
    )
    stage2_acc_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Weight for stage-2 accuracy reward (Vision-SR1 default 1.0)"},
    )
    # --- VPPO: Visual (here: Audio) Perception Policy Optimization ---
    use_vppo: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable VPPO: perturb audio, compute per-token KL to identify "
            "audio-critical tokens, mask gradients to only top-p%."
        },
    )
    vppo_top_p: Optional[float] = field(
        default=0.2,
        metadata={"help": "Fraction of tokens per response to keep for gradient update (top-p by KL)"},
    )
    vppo_adv_scale_min: Optional[float] = field(
        default=0.8,
        metadata={"help": "Min scaling factor for advantage shaping (VPPO macro-level). 1.0 = disabled"},
    )
    vppo_mask_ratio: Optional[float] = field(
        default=0.7,
        metadata={
            "help": "Fraction of mel time-frames to zero out in the perturbed "
            "forward (direct audio analog of VPPO image patch blackening). "
            "Recommended 0.6-0.8; higher = stronger KL signal, more aggressive."
        },
    )
    use_vppo_on_entropy: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Also include top-p% high-entropy tokens in the gradient mask "
            "(OR with perception mask). Brings decision-point / reasoning tokens "
            "back into gradient that perception mask alone would miss."
        },
    )
    top_p_entropy_tokens: Optional[float] = field(
        default=0.2,
        metadata={"help": "Fraction of tokens per response kept by entropy mask"},
    )
    vppo_answer_protect_tokens: Optional[int] = field(
        default=0,
        metadata={
            "help": "Number of last valid tokens per rollout to always include "
            "in the perception mask (protects </think><answer>X</answer>). "
            "0 = off. 15 typically covers the answer region."
        },
    )
    mask_anneal_schedule: Optional[str] = field(
        default="none",
        metadata={
            "help": "Anneal the VPPO perception mask strength from 1 to 0 over "
            "training. 'none' = constant mask (VPPO throughout). 'linear' or "
            "'cosine' = interpolate effective_mask = alpha*mask + (1-alpha)*1, "
            "so late-stage loss becomes plain GRPO."
        },
    )
    mask_anneal_warmup_steps: Optional[int] = field(
        default=0,
        metadata={
            "help": "Hold alpha=1 for the first N steps before starting to decay. "
            "Decay then runs over [warmup, max_steps]. Useful when the VPPO peak "
            "comes early (e.g. step 200) and you want full mask preserved through "
            "the peak before relaxing."
        },
    )
    mask_anneal_lambda: Optional[float] = field(
        default=10.0,
        metadata={
            "help": "Sigmoid steepness: alpha(u) = 1/(1+exp(lambda*(u - gamma))). "
            "u is post-warmup progress in [0,1]. Larger = sharper drop. Only "
            "used when mask_anneal_schedule=sigmoid."
        },
    )
    mask_anneal_gamma_frac: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Sigmoid midpoint as fraction of post-warmup phase. alpha=0.5 "
            "at this point. Only used when mask_anneal_schedule=sigmoid."
        },
    )
    rubric_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSONL of precomputed rubrics keyed by problem_id. "
                          "Required when 'rubric' is in reward_funcs."},
    )
    rubric_judge_model: Optional[str] = field(
        default="gemini-2.5-flash",
        metadata={"help": "Gemini model used by the rubric judge for binary Yes/No grading."},
    )
    rubric_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of parallel API workers per training step for rubric judging."},
    )
    rubric_weight: Optional[float] = field(
        default=0.5,
        metadata={"help": "Reward weight applied to the rubric reward output."},
    )
    overthinking_weight: Optional[float] = field(
        default=0.1,
        metadata={"help": "Reward weight for the overthinking length penalty."},
    )
    overthinking_lmax: Optional[int] = field(
        default=256,
        metadata={"help": "L_max_output for overthinking penalty: R = 1 - |t_i|/L_max (token len of <think>)."},
    )
    rubric_neutral: Optional[float] = field(
        default=0.5,
        metadata={"help": "Fallback score used when rubric is missing for a sample or judge fails."},
    )
    rubric_judge_reasoning: Optional[str] = field(
        default=None,
        metadata={"help": "reasoning_effort for the rubric judge ('low'|'medium'|'high'). "
                          "If None, auto-low for gemini-3* flash models."},
    )
    use_evolving_rubric: Optional[bool] = field(
        default=False,
        metadata={"help": "When True the 'evolving_rubric' reward func runs the per-step "
                          "adaptive rubric pipeline; otherwise it falls back to static rubric_reward."},
    )
    evolving_topk: Optional[int] = field(
        default=5,
        metadata={"help": "Top-K most discriminative rubrics retained per group (after std filter)."},
    )
    evolving_max_new: Optional[int] = field(
        default=3,
        metadata={"help": "Max number of new rubrics Gemini may generate per call."},
    )
    evolving_system_prompt_path: Optional[str] = field(
        default="data/evolving_rubric_system_prompt.md",
        metadata={"help": "Path to the rubric-generator system prompt markdown."},
    )
    evolving_uniform_weights: Optional[bool] = field(
        default=False,
        metadata={"help": "Skip Gemini Call #2 and use 1/K weights for the kept rubrics."},
    )
    evolving_call_every: Optional[int] = field(
        default=1,
        metadata={"help": "Run evolving every N steps; static rubric_reward is used in between (cost control)."},
    )


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "stage2_accuracy": stage2_accuracy_reward,
    "rubric": rubric_reward,
    "evolving_rubric": evolving_rubric_reward,
    "overthinking": overthinking_penalty_reward,
}


training_datasets = [
    "AVQA",
]


def main(script_args, training_args, model_args):
    reward_funcs = []
    for func in script_args.reward_funcs:
        if func == "accuracy":
            partial_func = partial(
                reward_funcs_registry[func],
                log_dir=training_args.output_dir,
                alpha_a=script_args.alpha_a,
                alpha_k=script_args.alpha_k,
                alpha_g=script_args.alpha_g,
                alpha_k_ratio=script_args.alpha_k_ratio,
                reward_weight=script_args.stage1_acc_weight,
            )
        elif func == "format":
            partial_func = partial(
                reward_funcs_registry[func],
                require_description=bool(script_args.use_sr1),
                reward_weight=script_args.stage1_format_weight,
            )
        elif func == "stage2_accuracy":
            partial_func = partial(
                reward_funcs_registry[func],
                reward_weight=script_args.stage2_acc_weight,
            )
        elif func == "rubric":
            partial_func = partial(
                reward_funcs_registry[func],
                rubric_path=script_args.rubric_path,
                rubric_judge_model=script_args.rubric_judge_model,
                rubric_workers=script_args.rubric_workers,
                rubric_neutral=script_args.rubric_neutral,
                rubric_judge_reasoning=script_args.rubric_judge_reasoning,
                reward_weight=script_args.rubric_weight,
            )
        elif func == "evolving_rubric":
            partial_func = partial(
                reward_funcs_registry[func],
                rubric_path=script_args.rubric_path,
                rubric_judge_model=script_args.rubric_judge_model,
                rubric_workers=script_args.rubric_workers,
                rubric_neutral=script_args.rubric_neutral,
                rubric_judge_reasoning=script_args.rubric_judge_reasoning,
                reward_weight=script_args.rubric_weight,
                use_evolving_rubric=script_args.use_evolving_rubric,
                evolving_topk=script_args.evolving_topk,
                evolving_max_new=script_args.evolving_max_new,
                evolving_system_prompt_path=script_args.evolving_system_prompt_path,
                evolving_uniform_weights=script_args.evolving_uniform_weights,
                evolving_call_every=script_args.evolving_call_every,
                output_dir=training_args.output_dir,
            )
        elif func == "overthinking":
            partial_func = partial(
                reward_funcs_registry[func],
                reward_weight=script_args.overthinking_weight,
                l_max=script_args.overthinking_lmax,
            )
        else:
            partial_func = reward_funcs_registry[func]
        reward_funcs.append(partial_func)

    with open(script_args.datasets_json, "r") as file:
        datasets = json.load(file)

    combined_dataset = []
    supplementary_dataset = []
    training_datasets = script_args.training_datasets

    for dataset_name in training_datasets:
        try:
            if dataset_name == "AVQA":
                dataset = AVQADataset(
                    dataset_path=datasets.get(dataset_name, ""),
                    use_sr1=bool(script_args.use_sr1),
                )
            else:
                print(f"Warning: Unknown dataset {dataset_name}, skipping")
                continue
        except Exception as e:
            raise f"Preparing datasets Error: {e}"
        combined_dataset.append(dataset)

    combined_dataset = ConcatDataset(combined_dataset + supplementary_dataset)

    assert combined_dataset is not None, "No valid datasets found for training."

    dataset = combined_dataset

    dataset_sources = []
    dataset_indices = {}

    cumulative_size = 0
    dataset_idx = 0

    for ds in combined_dataset.datasets:
        ds_size = len(ds)
        dataset_sources.extend([dataset_idx] * ds_size)
        dataset_indices[dataset_idx] = (cumulative_size, cumulative_size + ds_size)
        cumulative_size += ds_size
        dataset_idx += 1

    group_ids = []
    for ds in combined_dataset.datasets:
        if isinstance(ds, AVQADataset):
            group_ids.append(0)
        else:
            group_ids.append(1)

    is_distributed = True
    if is_distributed:
        num_replicas = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        num_replicas = 1
        rank = 0

    sampler = GroupSampler(
        data_sources=dataset_sources,
        group_ids=group_ids,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        seed=training_args.seed,
        distributed=is_distributed,
        num_replicas=num_replicas,
        rank=rank,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=lambda x: x,
        num_workers=min(2, training_args.dataloader_num_workers),
        pin_memory=training_args.dataloader_pin_memory,
    )

    trainer = Qwen2VLGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset,
        train_dataloader=data_loader,
        eval_dataset=None,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    if _swanlab_available:
        try:
            swanlab.finish()
        except Exception:
            pass


def save_args_to_file(
    script_args, training_args, model_args, filepath="saved_args.json"
):
    import json
    import os
    from dataclasses import asdict

    args_dict = {
        "script_args": asdict(script_args),
        "training_args": training_args.to_dict(),
        "model_args": asdict(model_args),
    }

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(args_dict, f, indent=4)

    print(f"所有参数已保存到: {filepath}")


if __name__ == "__main__":
    wait_for_debugger_if_rank0()
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    save_args_to_file(
        script_args,
        training_args,
        model_args,
        filepath=os.path.join(training_args.output_dir, "all_args.json"),
    )

    main(script_args, training_args, model_args)
