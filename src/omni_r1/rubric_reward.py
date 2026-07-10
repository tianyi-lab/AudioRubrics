"""Rubric-as-rewards (RaR) reward function for GRPO.

For each rollout (completion + audio_path), look up the precomputed 5-criterion
rubric for that prompt (keyed by problem_id), call an LLM judge to decide which
criteria are satisfied (binary), and return  S(x, y) = Σ w_k · 1[Yes_k]  ∈ [0, 1].

Usage:
    --reward_funcs accuracy format rubric \\
    --rubric_path /home/jovyan/Audio_reason/outputs/rubrics_avqa_train.jsonl \\
    --rubric_judge_model gemini-2.5-flash \\
    --rubric_workers 8

Combined with the existing accuracy + format rewards via the standard reward
registry. Reward weight is independently controllable via the existing alpha_*
script args (this function multiplies its output by `reward_weight`).
"""
from __future__ import annotations

import base64
import json
import os
import threading
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy import of OpenAI client to avoid hard dep at module load.
_client_cache: Optional[Any] = None
_rubric_store: Dict[str, Dict[str, Any]] = {}
_rubric_path_loaded: Optional[str] = None
_score_cache: Dict[str, float] = {}  # (id, hash(completion)) -> score


_client_pool = None
_client_iter = None
_client_lock = threading.Lock()


def _get_client():
    """Round-robin over one or more Gemini keys to spread quota/rate load.

    Reads GEMINI_API_KEYS (comma-separated) if set, else GEMINI_API_KEY.
    """
    global _client_pool, _client_iter
    if _client_pool is None:
        import itertools
        backend = os.environ.get("JUDGE_BACKEND", "gemini").lower()
        if backend == "trapi":
            from openai import AzureOpenAI
            tok_file = os.environ["TRAPI_TOKEN_FILE"]  # required for the azure backend
            endpoint = os.environ["TRAPI_ENDPOINT"]  # your Azure OpenAI endpoint
            apiver = os.environ.get("TRAPI_API_VERSION", "2025-04-01-preview")
            def _tok():
                return open(tok_file).read().strip()
            _client_pool = [AzureOpenAI(azure_endpoint=endpoint, azure_ad_token_provider=_tok, api_version=apiver)]
            _client_iter = itertools.cycle(_client_pool)
            print(f"[rubric] TRAPI judge backend (token file {tok_file})")
        else:
            from openai import OpenAI
            raw = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
            keys = [k.strip() for k in raw.split(",") if k.strip()] or [""]
            _client_pool = [
                OpenAI(api_key=k, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
                for k in keys
            ]
            _client_iter = itertools.cycle(_client_pool)
            print(f"[rubric] Gemini client pool: {len(_client_pool)} key(s) round-robin")
    with _client_lock:
        return next(_client_iter)


# Backend-aware create kwargs: TRAPI(gpt-audio) doesn't take temperature/reasoning_effort,
# uses max_completion_tokens; Gemini uses temperature=0 (+optional reasoning_effort).
def _create_kwargs(reasoning_effort=None):
    if os.environ.get("JUDGE_BACKEND", "gemini").lower() == "trapi":
        return {"max_completion_tokens": int(os.environ.get("TRAPI_MAX_TOKENS", "3000"))}
    kw = {"temperature": 0}
    if reasoning_effort:
        kw["reasoning_effort"] = reasoning_effort
    return kw


def _load_rubrics(path: str) -> None:
    """Load JSONL of rubrics into the global store, keyed by sample id."""
    global _rubric_store, _rubric_path_loaded
    if _rubric_path_loaded == path:
        return
    store: Dict[str, Dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            rb = r.get("rubric", {})
            if not isinstance(rb, dict) or "_error" in rb:
                continue
            rubrics = rb.get("rubrics", [])
            if not rubrics or len(rubrics) != 5:
                continue
            # Normalize weights to sum to 1.0 in case Gemini drift slightly
            total_w = sum(c.get("weight", 0.0) for c in rubrics)
            if total_w <= 0:
                continue
            norm_rubrics = [
                {"category": c.get("category", ""),
                 "criterion": c.get("criterion", ""),
                 "weight": c.get("weight", 0.0) / total_w}
                for c in rubrics
            ]
            store[r["id"]] = {
                "audio_path": r.get("audio_path", ""),
                "rubrics": norm_rubrics,
                "question_domain": rb.get("question_domain", ""),
            }
    _rubric_store = store
    _rubric_path_loaded = path
    print(f"[rubric_reward] loaded {len(store)} rubrics from {path}", flush=True)


def _audio_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _judge_prompt(question: str, response: str, rubrics: List[Dict[str, Any]]) -> str:
    rubric_text = "\n".join(
        f"R{i+1} ({c['category']}, weight {c['weight']:.2f}): {c['criterion']}"
        for i, c in enumerate(rubrics)
    )
    return f"""You are an expert audio response evaluator. Listen to the attached audio, read the question, candidate response, and 5 binary rubrics. For each rubric, decide whether the candidate response satisfies it (Yes) or not (No).

[Question]: {question}

[Candidate Response]:
{response}

[Rubrics to evaluate]:
{rubric_text}

Return strictly JSON (no code fences):
{{"R1": "Yes|No", "R2": "Yes|No", "R3": "Yes|No", "R4": "Yes|No", "R5": "Yes|No"}}"""


def _judge_one(
    sample_id: str,
    completion_text: str,
    question: str,
    audio_path: str,
    rubrics: List[Dict[str, Any]],
    model: str,
    max_retries: int = 4,
    reasoning_effort: Optional[str] = None,
) -> Optional[float]:
    """Run the judge on one (sample, completion) pair. Returns Σ w_k · 1[Yes_k] or None on failure."""
    cache_key = f"{sample_id}|{hash(completion_text)}"
    if cache_key in _score_cache:
        return _score_cache[cache_key]
    if not Path(audio_path).exists():
        return None
    try:
        b64 = _audio_b64(audio_path)
    except Exception:
        return None
    prompt = _judge_prompt(question, completion_text, rubrics)
    client = _get_client()
    extra_kwargs = _create_kwargs(reasoning_effort)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    {"type": "text", "text": prompt},
                ]}],
                timeout=120,
                **extra_kwargs,
            )
            txt = resp.choices[0].message.content.strip()
            txt = re.sub(r"^```(?:json)?\s*", "", txt)
            txt = re.sub(r"\s*```\s*$", "", txt)
            verdicts = json.loads(txt)
            score = 0.0
            for i, c in enumerate(rubrics, 1):
                v = verdicts.get(f"R{i}", "No")
                if isinstance(v, str) and v.strip().lower().startswith("y"):
                    score += c["weight"]
            _score_cache[cache_key] = score
            return score
        except Exception as e:
            msg = str(e)[:120]
            if any(k in msg for k in ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                time.sleep(min(30, 2 * (2**attempt)))
                continue
            # transient parse / network — short backoff
            time.sleep(2)
    return None


def rubric_reward(
    completions: List[List[Dict[str, str]]],
    kwargs_list: List[Dict[str, Any]],
    ref_generate=None,
    log_dir=None,
    reward_weight: float = 1.0,
    rubric_path: Optional[str] = None,
    rubric_judge_model: str = "gemini-2.5-flash",
    rubric_workers: int = 8,
    rubric_neutral: float = 0.5,
    rubric_judge_reasoning: Optional[str] = None,
    **_,
) -> List[float]:
    """Top-level reward function compatible with the trainer's registry."""
    if rubric_path is None:
        return [0.0] * len(completions)
    _load_rubrics(rubric_path)
    # Auto-pick reasoning_effort=low for gemini-3 series unless explicit override
    eff_reasoning = rubric_judge_reasoning
    if eff_reasoning is None and "gemini-3" in rubric_judge_model:
        eff_reasoning = "low"

    # Build judging tasks; samples without a rubric get the neutral fallback.
    tasks = []  # list of (i, fn_args) for those needing judging
    rewards: List[float] = [0.0] * len(completions)
    for i, kw in enumerate(kwargs_list):
        sample_id = kw.get("problem_id")
        rec = _rubric_store.get(sample_id)
        if rec is None:
            rewards[i] = rubric_neutral * reward_weight
            continue
        completion_text = completions[i][0]["content"]
        tasks.append((i, sample_id, completion_text, kw.get("problem", ""),
                      kw.get("audio_path", rec["audio_path"]), rec["rubrics"]))

    if not tasks:
        return rewards

    with ThreadPoolExecutor(max_workers=rubric_workers) as ex:
        fut_to_idx = {
            ex.submit(_judge_one, sid, ct, q, ap, rb, rubric_judge_model,
                      4, eff_reasoning): i
            for (i, sid, ct, q, ap, rb) in tasks
        }
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                s = fut.result()
            except Exception:
                s = None
            rewards[idx] = (s if s is not None else rubric_neutral) * reward_weight

    return rewards
