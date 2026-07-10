#!/usr/bin/env python3
"""
Generate answers for the MMAR benchmark using the vllm OpenAI-compatible API
with Qwen2.5-Omni.

Workflow
--------
1. Start the vllm server in one terminal:
       bash start_vllm_server.sh Qwen/Qwen2.5-Omni-7B-Instruct 8000 1

2. Run this script in another terminal (no special env activation needed):
       python generate_answers_vllm.py \\
           --input   datasets/MMAR-meta.jsonl \\
           --audio_dir datasets/audio \\
           --output  outputs/MMAR_qwen25omni_preds.jsonl

   Optional flags:
       --base_url    http://localhost:8000/v1   (vllm server address)
       --model       qwen2.5-omni               (must match --served-model-name)
       --max_workers 4                          (concurrent API requests)
       --max_new_tokens 64
       --temperature 0.0
       --limit 10                               (debug: first N samples only)
"""

import argparse
import base64
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LETTERS = [chr(ord("A") + i) for i in range(26)]


# --------------------------------------------------------------------------- #
# Audio helpers                                                                #
# --------------------------------------------------------------------------- #

def wav_to_base64(audio_path: Path) -> str:
    """Read a WAV/audio file and return a base64 string of its raw bytes.

    Tries soundfile first (re-encodes to PCM-16 WAV for consistency);
    falls back to reading the raw file bytes if soundfile is not installed.
    """
    try:
        import soundfile as sf

        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)  # stereo → mono
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except ImportError:
        # soundfile not installed — send raw bytes as-is (works for PCM WAV)
        return base64.b64encode(audio_path.read_bytes()).decode("utf-8")


def audio_content_block(b64: str) -> dict:
    """Build a vllm-compatible audio_url content block (data URL format)."""
    return {
        "type": "audio_url",
        "audio_url": {
            "url": f"data:audio/wav;base64,{b64}",
        },
    }


# --------------------------------------------------------------------------- #
# Prompt builder                                                               #
# --------------------------------------------------------------------------- #

def build_messages(question: str, choices: list[str], b64: str, no_think: bool = False) -> list:
    """Build the messages list for the OpenAI chat API."""
    choice_lines = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    if no_think:
        user_text = (
            "Listen to the audio carefully and answer the following "
            "multiple-choice question.\n\n"
            f"Question: {question}\n\n"
            f"Choices:\n{choice_lines}\n\n"
            "Output ONLY the letter (A, B, C, …) of the correct answer "
            "inside <answer> ... </answer> tags. Do not reason or explain.\n\n"
            "Example format:\n"
            "<answer>B</answer>"
        )
        system_content = (
            "You are an expert audio understanding assistant. "
            "Listen carefully and answer multiple-choice questions. "
            "Give the final answer letter directly inside <answer> tags."
        )
    else:
        user_text = (
            "Listen to the audio carefully and answer the following "
            "multiple-choice question.\n\n"
            f"Question: {question}\n\n"
            f"Choices:\n{choice_lines}\n\n"
            "First, reason step by step inside <think> ... </think> tags.\n"
            "Then output ONLY the letter (A, B, C, …) of the correct answer "
            "inside <answer> ... </answer> tags.\n\n"
            "Example format:\n"
            "<think>\nYour reasoning here.\n</think>\n"
            "<answer>B</answer>"
        )
        system_content = (
            "You are an expert audio understanding assistant. "
            "Listen carefully and answer multiple-choice questions. "
            "Always think step by step inside <think> tags, "
            "then give the final answer letter inside <answer> tags."
        )
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": [
                audio_content_block(b64),
                {"type": "text", "text": user_text},
            ],
        },
    ]


# --------------------------------------------------------------------------- #
# Answer extraction                                                            #
# --------------------------------------------------------------------------- #

def extract_letter(raw: str, choices: list[str]) -> str:
    """
    Map the raw model output back to a choice string.
    First tries to parse the <answer> tag, then falls back to 5 heuristics.
    """
    cleaned = raw.strip()

    # 0) Prefer explicit <answer>X</answer> tag
    m = re.search(r"<answer>\s*([A-Fa-f])\s*</answer>", cleaned, flags=re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        idx = LETTERS.index(letter) if letter in LETTERS else -1
        if 0 <= idx < len(choices):
            return choices[idx]

    # If the model wrapped the full choice text in <answer>, try that too
    m = re.search(r"<answer>\s*(.+?)\s*</answer>", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if m:
        inner = m.group(1).strip()
        # check if inner IS one of the choices
        for c in choices:
            if c is not None and inner.lower() == c.lower():
                return c
        # check if it starts with a letter
        m2 = re.match(r"^([A-Fa-f])[.:\)\s]", inner)
        if m2:
            letter = m2.group(1).upper()
            idx = LETTERS.index(letter) if letter in LETTERS else -1
            if 0 <= idx < len(choices):
                return choices[idx]

    # 1) Bare single letter
    if re.fullmatch(r"[A-Fa-f]", cleaned):
        letter = cleaned.upper()
        idx = LETTERS.index(letter) if letter in LETTERS else -1
        if 0 <= idx < len(choices):
            return choices[idx]

    # 2) Letter at the very start followed by punctuation/space
    m = re.match(r"^([A-Fa-f])[\.:\)\s]", cleaned)
    if m:
        letter = m.group(1).upper()
        idx = LETTERS.index(letter) if letter in LETTERS else -1
        if 0 <= idx < len(choices):
            return choices[idx]

    # 3) "the answer is X" / "choose X" / "option X"
    m = re.search(
        r"(?:answer\s+is|answer:|choose|option|select)\s*['\"]?([A-Fa-f])['\"]?",
        cleaned, flags=re.IGNORECASE,
    )
    if m:
        letter = m.group(1).upper()
        idx = LETTERS.index(letter) if letter in LETTERS else -1
        if 0 <= idx < len(choices):
            return choices[idx]

    # 4) Any standalone uppercase letter
    m = re.search(r"\b([A-F])\b", cleaned)
    if m:
        letter = m.group(1)
        idx = LETTERS.index(letter) if letter in LETTERS else -1
        if 0 <= idx < len(choices):
            return choices[idx]

    # 5) Fall back to raw text (evaluation.py string_match will handle it)
    return cleaned


# --------------------------------------------------------------------------- #
# Per-sample inference                                                         #
# --------------------------------------------------------------------------- #

def run_sample(
    client,
    model: str,
    sample: dict,
    audio_dir: Path,
    max_new_tokens: int,
    temperature: float,
    no_think: bool = False,
) -> dict:
    """Run inference for one MMAR sample; returns the sample dict with results."""
    audio_path = audio_dir / Path(sample.get("audio_path", "")).name

    # Helper: build a minimal record (used for error cases)
    def _record(prediction: str, thinking: str = "") -> dict:
        # Safely handle cases where answer or choices are non-string (e.g., NaN, floats,
        # numpy types). Coerce to str for comparisons and ensure choices is a list.
        raw_answer = sample.get("answer", "")
        answer_text = "" if raw_answer is None else str(raw_answer)

        raw_choices = sample.get("choices", []) or []
        try:
            choices_list = list(raw_choices)
        except Exception:
            choices_list = [raw_choices]

        answer_letter = ""
        for i, c in enumerate(choices_list):
            try:
                if str(c).lower() == answer_text.lower():
                    answer_letter = LETTERS[i]
                    break
            except Exception:
                continue

        return {
            "id":                sample.get("id", ""),
            "question":          sample.get("question", ""),
            "choices":           choices_list,
            "answer":            answer_letter,
            "answer_text":       answer_text,
            "thinking":          thinking,
            "answer_prediction": prediction,
            "modality":          sample.get("modality", ""),
            "category":          sample.get("category", ""),
            "sub-category":      sample.get("sub-category", ""),
        }

    if not audio_path.exists():
        return _record(f"[ERROR: audio not found: {audio_path}]")

    try:
        b64 = wav_to_base64(audio_path)
    except Exception as e:
        return _record(f"[ERROR reading audio: {e}]")

    messages = build_messages(sample["question"], sample["choices"], b64, no_think=no_think)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            if not resp.choices:
                raise ValueError(f"Empty choices in response: {resp}")
            raw_text = resp.choices[0].message.content or ""
            break
        except Exception as e:
            if attempt == 2:
                return _record(f"[ERROR API: {e}]")
            time.sleep(2 ** attempt)

    predicted_letter = extract_letter(raw_text, sample["choices"])
    think_m = re.search(r"<think>\s*(.*?)\s*</think>", raw_text, flags=re.DOTALL | re.IGNORECASE)
    thinking = think_m.group(1).strip() if think_m else ""

    return _record(predicted_letter, thinking)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="MMAR inference via vllm OpenAI-compatible API"
    )
    p.add_argument("--base_url", default="http://localhost:8000/v1",
                   help="vllm server base URL")
    p.add_argument("--model", default="qwen2.5-omni",
                   help="Served model name (must match --served-model-name in vllm serve)")
    p.add_argument("--input", default="datasets/MMAR/annotation/MMAR-meta.jsonl",
                   help="Path to MMAR-meta.jsonl")
    p.add_argument("--audio_dir", default="datasets/MMAR/audio",
                   help="Directory containing WAV files")
    p.add_argument("--output", default="outputs/MMAR_qwen25omni_preds.jsonl",
                   help="Output JSONL path")
    p.add_argument("--max_workers", type=int, default=4,
                   help="Number of concurrent API requests")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Max tokens to generate (thinking + answer; use ≥512)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None,
                   help="Only process first N samples (for debugging)")
    p.add_argument("--no_think", action="store_true",
                   help="直接输出 <answer>X</answer>, 不用 <think> 推理")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai package not found.  pip install openai")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Connect to the vllm server                                          #
    # ------------------------------------------------------------------ #
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    print(f"Connecting to vllm server at {args.base_url} …")

    try:
        available = [m.id for m in client.models.list().data]
        print(f"Available models: {available}")
        if args.model not in available:
            print(f"[WARN] '{args.model}' not in {available}. "
                  f"Check --model or start the server with the correct --served-model-name.")
    except Exception as e:
        print(f"[ERROR] Cannot reach vllm server: {e}")
        print("  → Start the server first:  bash start_vllm_server.sh")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Load dataset                                                        #
    # ------------------------------------------------------------------ #
    input_path = Path(args.input)
    with open(input_path) as f:
        if str(input_path).endswith(".json"):
            samples = json.load(f)
        else:
            samples = [json.loads(l) for l in f if l.strip()]

    if args.limit:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} samples")

    # ------------------------------------------------------------------ #
    # Output file — support resume                                        #
    # ------------------------------------------------------------------ #
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "id" in rec:
                        done_ids.add(rec["id"])
                except Exception:
                    pass
        print(f"Resuming – {len(done_ids)} already done")

    todo = [s for s in samples if s.get("id") not in done_ids]
    print(f"Remaining: {len(todo)}")

    audio_dir = Path(args.audio_dir)

    # ------------------------------------------------------------------ #
    # Run inference with thread pool                                      #
    # ------------------------------------------------------------------ #
    out_f = open(output_path, "a", encoding="utf-8")
    completed = errors = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                run_sample, client, args.model, sample,
                audio_dir, args.max_new_tokens, args.temperature, args.no_think
            ): sample
            for sample in todo
        }

        for future in as_completed(futures):
            result = future.result()
            pred = result.get("answer_prediction", "")
            if str(pred).startswith("[ERROR"):
                errors += 1
                print(f"\n[WARN] {result.get('id')}: {pred}")
            else:
                completed += 1
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            total_done = completed + errors
            print(
                f"  [{total_done}/{len(todo)}] done "
                f"(ok={completed}, err={errors})",
                end="\r",
            )

    out_f.close()
    print(f"\n\nDone! {completed} predictions → {output_path}")
    if errors:
        print(f"Errors / skipped: {errors}")
    print(f"\nEvaluate:\n  python evaluation.py --input {output_path}")


if __name__ == "__main__":
    main()
