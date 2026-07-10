import fcntl
import json
import os
import re
import threading
from pathlib import Path as pth
from typing import Dict, List

from omni_r1.load_datasets import video_r1_reward

QUESTION_TYPE = {
    "VideoR1": ["multiple choice", "numerical", "OCR", "free-form", "regression"],
}


def extract_answer(text):
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


_file_locks = {}
_file_locks_lock = threading.Lock()


def output_training_log(
    completion: str,
    gt: str,
    reward: float,
    raw_input: dict,
    output_path: pth = None,
    reward_dict: dict = None,
):
    output_path.mkdir(parents=True, exist_ok=True)
    log_file_path = pth(output_path) / "info.txt"

    with _file_locks_lock:
        if str(log_file_path) not in _file_locks:
            _file_locks[str(log_file_path)] = threading.Lock()
        file_lock = _file_locks[str(log_file_path)]

    with file_lock:
        with open(log_file_path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(
                    f"------------- {raw_input['problem_type']} Accuracy reward: {reward} -------------\n"
                )
                f.write(f"Problem: {raw_input['problem']}\n")
                f.write(f"Content: {completion}\n")
                if reward_dict is not None:
                    f.write(f"Reward Details: {json.dumps(reward_dict, indent=4)}\n")
                if gt:
                    f.write(f"Solution: {gt}\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def accuracy_reward(
    completions: List[str],
    kwargs_list: List[Dict],
    ref_generate=None,
    log_dir=None,
    reward_weight: float = 1.0,
    **alphas,
):
    log_dir = pth(log_dir).resolve() / "train_logs" if log_dir else None

    rewards = []

    # dummy call to satisfy GRPO trainer's ref_generate interface
    dummy_prompt = [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello!"}],
                }
            ]
        }
    ]
    _, _ = ref_generate([dummy_prompt])

    for i, kwarg in enumerate(kwargs_list):
        question_type = kwarg["problem_type"]
        content = completions[i][0]["content"]
        sol = kwarg.get("solution", None)
        reward = 0.0
        try:
            if question_type in QUESTION_TYPE["VideoR1"]:
                output_ans = extract_answer(content)
                gt_ans = extract_answer(sol)
                reward = video_r1_reward(
                    question_type=question_type,
                    output_ans=output_ans,
                    gt_ans=gt_ans,
                    **kwarg,
                )
            else:
                print(f"Unexpected question type: {question_type}")
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0

        rewards.append(reward * reward_weight)

        if os.getenv("LOG_MODE") != "true" or log_dir is None:
            continue

        output_training_log(
            completion=content,
            gt=sol,
            reward=reward,
            raw_input=kwarg,
            output_path=log_dir / f"{kwarg.get('problem_id', i)}",
        )

    return rewards


def format_reward(
    completions,
    kwargs_list=None,
    ref_generate=None,
    require_description: bool = False,
    reward_weight: float = 1.0,
    **kwargs,
):
    """Format reward for stage-1 completion.

    When `require_description=True` (Audio-SR1 / Vision-SR1 style), the full
    response must match:
        <description>...</description>  <think>...</think>  <answer>X</answer>

    Otherwise only requires <think>...</think> plus trailing <answer>...</answer>.
    """
    if require_description:
        pattern = re.compile(
            r"^\s*<description>.*?</description>\s*"
            r"<think>.*?</think>\s*"
            r"<answer>.*?</answer>\s*$",
            re.DOTALL,
        )
        completion_contents = [c[0]["content"] for c in completions]
        rewards = []
        for content in completion_contents:
            ok = bool(pattern.fullmatch(content.strip()))
            rewards.append(reward_weight if ok else 0.0)
        return rewards

    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        c = content.strip()
        has_think = bool(re.search(r"<think>.*?</think>", c, re.DOTALL))
        ends_answer = bool(re.search(r"<answer>.*?</answer>\s*$", c, re.DOTALL))
        if has_think and ends_answer:
            rewards.append(reward_weight)
        else:
            rewards.append(0.0)
    return rewards


def stage2_accuracy_reward(
    completions,
    kwargs_list=None,
    ref_generate=None,
    reward_weight: float = 1.0,
    **kwargs,
):
    """Accuracy on the stage-2 (text-only, description-only) rollout.

    Each kwargs_list[i] must contain:
      - "stage2_completion": text generated by the model conditioned only on the
        extracted <description> (no audio features).
      - "solution": ground-truth string containing <answer>X</answer>.

    Returns 1.0 if stage-2 prediction matches ground truth, else 0.0.
    A short answer that omits <description> will produce an empty stage-2 text
    (nothing for the model to condition on), collapsing this reward to chance.
    """
    rewards = []
    if kwargs_list is None:
        return [0.0] * len(completions)
    for kw in kwargs_list:
        stage2_text = kw.get("stage2_completion", "") or ""
        gt = extract_answer(kw.get("solution", "")).strip().upper()
        pred = extract_answer(stage2_text).strip().upper()
        if not pred:
            m = re.findall(r"\b([A-D])\b", stage2_text)
            pred = m[-1].upper() if m else ""
        rewards.append(reward_weight if pred and pred == gt else 0.0)
    return rewards


# --- Overthinking length penalty -------------------------------------------
_overthink_tok = None


def _get_overthink_tokenizer():
    """Lazy-load a text tokenizer to measure reasoning length in tokens."""
    global _overthink_tok
    if _overthink_tok is None:
        from transformers import AutoTokenizer
        mp = os.environ.get("MODEL_PATH", "/tmp/Qwen2.5-Omni-7B")
        try:
            _overthink_tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        except Exception:
            _overthink_tok = False  # signal: fall back to whitespace proxy
    return _overthink_tok


def _think_token_len(content):
    """Token length of the <think>...</think> reasoning trace (whole text if no tags)."""
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    think = m.group(1) if m else content
    tok = _get_overthink_tokenizer()
    if tok:
        try:
            return len(tok.encode(think, add_special_tokens=False))
        except Exception:
            pass
    # fallback: whitespace tokens ~= 0.75 * subword tokens
    return int(len(think.split()) / 0.75)


def overthinking_penalty_reward(
    completions,
    kwargs_list=None,
    ref_generate=None,
    reward_weight: float = 1.0,
    l_max: int = 256,
    **kwargs,
):
    """Overthinking penalty: R = 1 - |t_i| / L_max_output  (linear, can go negative).

    Penalizes long reasoning traces to curb circular/verbose/hallucinatory
    elaboration. |t_i| is the token length of the <think> section; L_max=256.
    """
    contents = [c[0]["content"] for c in completions]
    rewards = []
    for content in contents:
        n = _think_token_len(content)
        rewards.append(reward_weight * (1.0 - n / float(l_max)))
    return rewards
