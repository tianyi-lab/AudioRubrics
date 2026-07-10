import json
from pathlib import Path as pth

from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert audio understanding assistant. "
    "Listen carefully and answer multiple-choice questions. "
    "Always think step by step inside <think> tags, "
    "then give the final answer letter inside <answer> tags."
)

# GRPO baseline prompt: only <think> + <answer>
QUESTION_TEMPLATE_BASELINE = (
    "Listen to the audio carefully and answer the following "
    "multiple-choice question.\n\n"
    "Question: {Question}\n\n"
    "First, reason step by step inside <think> ... </think> tags.\n"
    "Then output ONLY the letter (A, B, C, …) of the correct answer "
    "inside <answer> ... </answer> tags.\n\n"
    "Example format:\n"
    "<think>\nYour reasoning here.\n</think>\n"
    "<answer>B</answer>"
)

# Audio-SR1 prompt: <description> + <think> + <answer>
QUESTION_TEMPLATE = (
    "Listen to the audio carefully and answer the following "
    "multiple-choice question.\n\n"
    "Question: {Question}\n\n"
    "First, describe what you hear in the audio inside <description> ... </description> tags — "
    "key sounds, speech content, music, speakers, duration, etc. Be specific enough that "
    "someone who hasn't heard the audio could answer the question from your description alone.\n"
    "Then reason step by step inside <think> ... </think> tags.\n"
    "Finally, output ONLY the letter (A, B, C, …) of the correct answer "
    "inside <answer> ... </answer> tags.\n\n"
    "Example format:\n"
    "<description>\nI hear two people having a conversation in English. A man asks about food, a woman responds she wants more. Background is quiet.\n</description>\n"
    "<think>\nYour reasoning here.\n</think>\n"
    "<answer>B</answer>"
)

STAGE2_USER_TEMPLATE = (
    "Based on the following audio description, answer the multiple-choice question.\n\n"
    "Audio description:\n{description}\n\n"
    "Question: {Question}\n\n"
    "Reason briefly inside <think> ... </think> tags, then output ONLY the letter "
    "(A, B, C, …) of the correct answer inside <answer> ... </answer> tags.\n\n"
    "Example format:\n"
    "<think>\nYour reasoning here.\n</think>\n"
    "<answer>B</answer>"
)


class AVQADataset(Dataset):
    """Audio Visual Question Answering dataset loader for Omni-R1 GRPO training."""

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        max_samples: int = 0,
        use_sr1: bool = True,
    ):
        base = pth(dataset_path).resolve()

        if base.suffix == ".json":
            json_path = base
            self.data_root = base.parent
        else:
            json_path = base / f"{split}.json"
            self.data_root = base

        if not json_path.exists():
            raise FileNotFoundError(
                f"AVQADataset: metadata file not found at {json_path}. "
                "Run download_avqa.py first."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if max_samples and max_samples > 0:
            records = records[:max_samples]

        self.records = records
        self.use_sr1 = use_sr1

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.records[idx]

        audio_abs = str(self.data_root / item["audio_path"])
        question = item["question"]

        sys_prompt = SYSTEM_PROMPT
        template = QUESTION_TEMPLATE if self.use_sr1 else QUESTION_TEMPLATE_BASELINE
        user_text = template.format(Question=question)

        prompt = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": f"file://{audio_abs}"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        return {
            "prompt": prompt,
            "data_type": "audio",
            "problem_type": "multiple choice",
            "problem": question,
            "problem_id": item.get("id", f"avqa-{idx:06d}"),
            "solution": item["solution"],
            "answer": item["answer"],
            "answer_full": item.get("answer_full", item["answer"]),
            "audio_path": audio_abs,
            "duration": item.get("duration", 0.0),
        }


# ---------------------------------------------------------------------------
def avqa_reward(question_type: str, output_ans: str, gt_ans: str, **kwargs) -> float:
    return 1.0 if output_ans.strip().upper() == gt_ans.strip().upper() else 0.0
