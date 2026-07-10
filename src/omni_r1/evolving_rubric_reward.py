"""Evolving rubric-as-rewards (RaR) reward function for GRPO.

Per-step adaptive rubric scoring:

  1. Gemini Call #1 (audio + question + 8 rollouts + 5 static rubrics):
     returns {new_rubrics:[...], judgments:{rubric_id:[Yes/No]*8}} for the
     full set (static + new).
  2. Polarity flip on negative new rubrics so "satisfied = good" uniformly.
  3. Per-rubric std across 8 rollouts → drop std==0 → keep top-K (default 5).
  4. Gemini Call #2 (audio + question + kept rubrics): returns weights summing
     to 1.0.
  5. reward[r] = Σ_kept weights[k] · 1[satisfied_k for rollout r], in [0,1].

On any failure (Gemini error, malformed JSON, all rubrics std=0, no rubrics
survive): falls back to the static `rubric_reward` for the affected prompt.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omni_r1 import rubric_reward as _rr
from omni_r1.rubric_reward import (
    _audio_b64,
    _get_client,
    _create_kwargs,
    _load_rubrics,
    rubric_reward,
)

# --- module-level state ------------------------------------------------------

_evolve_prompt_cache: Dict[str, str] = {}        # path -> contents
_step_counter: int = 0
_step_counter_lock = threading.Lock()
_log_lock = threading.Lock()
_log_fp_cache: Dict[str, Any] = {}               # output_dir -> file handle


def _get_log_fp(output_dir: str):
    if output_dir in _log_fp_cache:
        return _log_fp_cache[output_dir]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fp = open(Path(output_dir) / "evolving_rubrics.jsonl", "a", buffering=1)
    _log_fp_cache[output_dir] = fp
    return fp


def _log_evolve(output_dir: Optional[str], record: Dict[str, Any]) -> None:
    if not output_dir:
        return
    try:
        fp = _get_log_fp(output_dir)
        with _log_lock:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_evolve_system_prompt(path: str) -> Optional[str]:
    if path in _evolve_prompt_cache:
        return _evolve_prompt_cache[path]
    try:
        text = Path(path).read_text()
        _evolve_prompt_cache[path] = text
        return text
    except Exception:
        return None


# --- helpers -----------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _extract_think(text: str) -> str:
    m = _THINK_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def _strip_code_fences(txt: str) -> str:
    txt = re.sub(r"^```(?:json)?\s*", "", txt.strip())
    txt = re.sub(r"\s*```\s*$", "", txt)
    return txt


# --- Gemini Call #1: generate new rubrics + judge all rubrics on 8 rollouts --

_ANSWER_RE = re.compile(r"<answer>\s*([A-Fa-f])\s*</answer>")


def _extract_answer_letter(text: str) -> str:
    m = _ANSWER_RE.search(text or "")
    return m.group(1).upper() if m else ""


def _build_call1_messages(
    question: str,
    static_rubrics: List[Dict[str, Any]],
    completions_8: List[str],
    audio_b64: str,
    max_new: int,
    system_prompt: str,
    gt_letter: str = "",
) -> List[Dict[str, Any]]:
    static_lines = []
    for i, c in enumerate(static_rubrics, 1):
        static_lines.append(
            f"S{i} ({c.get('category','')}): {c.get('criterion','')}"
        )
    static_block = "\n".join(static_lines) if static_lines else "(none)"

    gt_norm = (gt_letter or "").strip().upper()
    response_lines = []
    correctness_summary = []
    for i, comp in enumerate(completions_8, 1):
        think = _extract_think(comp)
        if len(think) > 2000:
            think = think[:2000] + " ... [truncated]"
        ans = _extract_answer_letter(comp)
        if gt_norm and ans:
            tag = "CORRECT" if ans == gt_norm else f"WRONG (chose {ans})"
        elif gt_norm and not ans:
            tag = "WRONG (no parseable answer)"
        else:
            tag = "(GT unavailable)"
        correctness_summary.append(f"T{i}: {tag}")
        response_lines.append(f"--- T{i}  [final answer: {ans or '?'} → {tag}] ---\n{think}")
    responses_block = "\n\n".join(response_lines)
    correctness_block = "\n".join(correctness_summary)

    schema_instructions = (
        f"\n\n## Required Output (strict JSON, no code fences)\n"
        f"Generate AT MOST {max_new} new rubrics that are non-redundant with "
        f"the static rubrics S1..S{len(static_rubrics)} above. Each new "
        f"rubric must include a 'polarity' field: 'positive' (satisfied = "
        f"good) or 'negative' (satisfied = bad/flaw present).\n"
        f"Then judge ALL rubrics (static + new) against EACH of the {len(completions_8)} "
        f"responses T1..T{len(completions_8)}. Use exactly 'Yes' or 'No'.\n\n"
        f"Schema:\n"
        f"{{\"new_rubrics\": ["
        f"{{\"id\": \"N1\", \"title\": \"<short label>\", "
        f"\"description\": \"<detailed criterion>\", "
        f"\"polarity\": \"positive|negative\"}}"
        f", ...],"
        f" \"judgments\": {{"
        f"\"S1\": [\"Yes\"|\"No\", ...{len(completions_8)} values], ..., "
        f"\"S{len(static_rubrics)}\": [...], "
        f"\"N1\": [...], ...}}}}"
    )

    gt_section = (
        f"## Ground-truth final answer\n{gt_norm}\n\n"
        if gt_norm else ""
    )
    correctness_section = (
        f"## Per-rollout correctness summary\n{correctness_block}\n\n"
        f"Note: A rollout's final-answer correctness is provided as context. "
        f"Generate rubrics that distinguish *reasoning quality*, not just "
        f"correctness — e.g., a rollout may be CORRECT through guessing "
        f"and still deserve a low score; a rollout may be WRONG yet show "
        f"good acoustic grounding for the wrong-but-plausible cue.\n\n"
        if gt_norm else ""
    )
    user_text = (
        f"## Question\n{question}\n\n"
        f"{gt_section}"
        f"## Existing (static) rubrics\n{static_block}\n\n"
        f"## Candidate Responses ({len(completions_8)} rollouts of the same prompt)\n"
        f"{responses_block}\n\n"
        f"{correctness_section}"
        f"{schema_instructions}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": user_text},
        ]},
    ]


def _call1_generate_and_judge(
    messages: List[Dict[str, Any]],
    model: str,
    reasoning_effort: Optional[str],
    n_rollouts: int,
    n_static: int,
    max_retries: int = 4,
) -> Optional[Dict[str, Any]]:
    client = _get_client()
    extra_kwargs = _create_kwargs(reasoning_effort)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=180,
                **extra_kwargs,
            )
            txt = _strip_code_fences(resp.choices[0].message.content or "")
            data = json.loads(txt)
            new_rubrics = data.get("new_rubrics", [])
            judgments = data.get("judgments", {})
            if not isinstance(new_rubrics, list) or not isinstance(judgments, dict):
                raise ValueError("malformed top-level types")
            # Ensure all static rubrics judged
            for i in range(1, n_static + 1):
                key = f"S{i}"
                if key not in judgments or not isinstance(judgments[key], list) \
                        or len(judgments[key]) != n_rollouts:
                    raise ValueError(f"missing/bad judgments for {key}")
            # New rubric judgments
            for r in new_rubrics:
                rid = r.get("id")
                if rid not in judgments or len(judgments[rid]) != n_rollouts:
                    raise ValueError(f"missing/bad judgments for new rubric {rid}")
                if r.get("polarity") not in ("positive", "negative"):
                    r["polarity"] = "positive"
            return {"new_rubrics": new_rubrics, "judgments": judgments}
        except Exception as e:
            msg = str(e)[:120]
            if any(k in msg for k in ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                time.sleep(min(30, 2 * (2**attempt)))
                continue
            time.sleep(2)
    return None


# --- post-processing ---------------------------------------------------------

def _to_binary(verdict_list: List[Any]) -> List[int]:
    return [
        1 if (isinstance(v, str) and v.strip().lower().startswith("y")) else 0
        for v in verdict_list
    ]


def _polarity_normalise(
    judgments: Dict[str, List[Any]],
    new_rubrics: List[Dict[str, Any]],
    n_static: int,
) -> Dict[str, List[int]]:
    """Convert all judgments to binary; flip negative new rubrics so 1=good."""
    normalised: Dict[str, List[int]] = {}
    # Static rubrics are positive by construction
    for i in range(1, n_static + 1):
        key = f"S{i}"
        normalised[key] = _to_binary(judgments[key])
    for r in new_rubrics:
        rid = r["id"]
        bits = _to_binary(judgments[rid])
        if r.get("polarity") == "negative":
            bits = [1 - b for b in bits]
        normalised[rid] = bits
    return normalised


def _std_binary(bits: List[int]) -> float:
    n = len(bits)
    if n == 0:
        return 0.0
    mean = sum(bits) / n
    return (sum((b - mean) ** 2 for b in bits) / n) ** 0.5


def _filter_topk(
    normalised: Dict[str, List[int]],
    topk: int,
) -> Tuple[List[str], Dict[str, List[int]]]:
    stds = [(rid, _std_binary(bits)) for rid, bits in normalised.items()]
    stds = [(rid, s) for rid, s in stds if s > 0.0]
    stds.sort(key=lambda x: -x[1])
    kept = stds[:topk]
    kept_ids = [rid for rid, _ in kept]
    return kept_ids, {rid: normalised[rid] for rid in kept_ids}


# --- Gemini Call #2: assign weights ------------------------------------------

_WEIGHT_INSTRUCTIONS = (
    "You are an expert audio-reasoning evaluator. You will receive an audio "
    "clip, a question, and a list of K binary rubrics that have been chosen "
    "as the most discriminative for assessing model responses on this clip. "
    "Listen to the audio, read the question, and decide how the K rubrics "
    "should be weighted: rubrics that more directly probe whether the response "
    "is grounded in this specific audio should receive higher weight. Output "
    "exactly K positive numbers that sum to 1.0.\n\n"
    "Strict JSON output (no code fences):\n"
    "{\"weights\": [w1, w2, ..., wK]}"
)


def _build_call2_messages(
    question: str,
    audio_b64: str,
    kept_rubric_texts: List[str],
) -> List[Dict[str, Any]]:
    rubric_block = "\n".join(
        f"R{i+1}: {t}" for i, t in enumerate(kept_rubric_texts)
    )
    user_text = (
        f"## Question\n{question}\n\n"
        f"## Kept Rubrics (K = {len(kept_rubric_texts)})\n{rubric_block}\n\n"
        f"## Required Output\nReturn ONLY: "
        f"{{\"weights\": [..K floats summing to 1.0..]}}"
    )
    return [
        {"role": "system", "content": _WEIGHT_INSTRUCTIONS},
        {"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": user_text},
        ]},
    ]


def _call2_assign_weights(
    audio_b64: str,
    question: str,
    kept_rubric_texts: List[str],
    model: str,
    reasoning_effort: Optional[str],
    max_retries: int = 4,
) -> Optional[List[float]]:
    if not kept_rubric_texts:
        return None
    K = len(kept_rubric_texts)
    messages = _build_call2_messages(question, audio_b64, kept_rubric_texts)
    client = _get_client()
    extra_kwargs = _create_kwargs(reasoning_effort)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=120,
                **extra_kwargs,
            )
            txt = _strip_code_fences(resp.choices[0].message.content or "")
            data = json.loads(txt)
            weights = data.get("weights")
            if not isinstance(weights, list) or len(weights) != K:
                raise ValueError("bad weights shape")
            weights = [max(0.0, float(w)) for w in weights]
            total = sum(weights)
            if total <= 0:
                raise ValueError("non-positive weight sum")
            return [w / total for w in weights]
        except Exception as e:
            msg = str(e)[:120]
            if any(k in msg for k in ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                time.sleep(min(30, 2 * (2**attempt)))
                continue
            time.sleep(2)
    return None


def _compute_rewards(
    kept_ids: List[str],
    kept_judgments: Dict[str, List[int]],
    weights: List[float],
    n_rollouts: int,
) -> List[float]:
    rewards = [0.0] * n_rollouts
    for i in range(n_rollouts):
        score = 0.0
        for k, rid in enumerate(kept_ids):
            score += weights[k] * kept_judgments[rid][i]
        rewards[i] = score
    return rewards


# --- per-prompt orchestration ------------------------------------------------

def _evolve_one_prompt(
    sample_id: str,
    question: str,
    audio_path: str,
    static_rubrics: List[Dict[str, Any]],
    completions_8: List[str],
    model: str,
    reasoning_effort: Optional[str],
    topk: int,
    max_new: int,
    uniform_weights: bool,
    system_prompt: str,
    output_dir: Optional[str],
    step: int,
    gt_letter: str = "",
) -> Optional[List[float]]:
    """Returns 8 floats in [0,1] or None on failure."""
    if not Path(audio_path).exists():
        _log_evolve(output_dir, {
            "step": step, "sample_id": sample_id, "fallback": "audio_missing",
            "audio_path": audio_path,
        })
        return None
    try:
        audio_b64 = _audio_b64(audio_path)
    except Exception as e:
        _log_evolve(output_dir, {
            "step": step, "sample_id": sample_id, "fallback": "audio_b64_failed",
            "error": str(e)[:200],
        })
        return None

    # Call 1
    messages1 = _build_call1_messages(
        question=question,
        static_rubrics=static_rubrics,
        completions_8=completions_8,
        audio_b64=audio_b64,
        max_new=max_new,
        system_prompt=system_prompt,
        gt_letter=gt_letter,
    )
    n_static = len(static_rubrics)
    n_rollouts = len(completions_8)
    parsed = _call1_generate_and_judge(
        messages1, model, reasoning_effort, n_rollouts, n_static,
    )
    if parsed is None:
        _log_evolve(output_dir, {
            "step": step, "sample_id": sample_id, "fallback": "call1_failed",
        })
        return None

    new_rubrics = parsed["new_rubrics"]
    normalised = _polarity_normalise(parsed["judgments"], new_rubrics, n_static)

    kept_ids, kept_judgments = _filter_topk(normalised, topk)
    if not kept_ids:
        _log_evolve(output_dir, {
            "step": step, "sample_id": sample_id, "fallback": "all_std_zero",
        })
        return None

    # Build human-readable rubric texts for Call 2 + logging
    rubric_text_map: Dict[str, str] = {}
    for i, c in enumerate(static_rubrics, 1):
        rubric_text_map[f"S{i}"] = (
            f"[{c.get('category','')}] {c.get('criterion','')}"
        )
    for r in new_rubrics:
        polarity_tag = "(absence of)" if r.get("polarity") == "negative" else ""
        rubric_text_map[r["id"]] = (
            f"[{r.get('title','')}] {polarity_tag} {r.get('description','')}"
        ).strip()
    kept_texts = [rubric_text_map.get(rid, rid) for rid in kept_ids]

    # Call 2 (or uniform)
    K = len(kept_ids)
    if uniform_weights:
        weights = [1.0 / K] * K
    else:
        weights = _call2_assign_weights(
            audio_b64=audio_b64,
            question=question,
            kept_rubric_texts=kept_texts,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if weights is None:
            weights = [1.0 / K] * K  # graceful: use uniform if call 2 fails

    scores = _compute_rewards(kept_ids, kept_judgments, weights, n_rollouts)

    _log_evolve(output_dir, {
        "step": step,
        "sample_id": sample_id,
        "n_static": n_static,
        "n_new_generated": len(new_rubrics),
        "kept_ids": kept_ids,
        "kept_texts": kept_texts,
        "kept_judgments": kept_judgments,
        "weights": weights,
        "scores": scores,
    })
    return scores


# --- top-level reward function ----------------------------------------------

def evolving_rubric_reward(
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
    # evolving-specific:
    use_evolving_rubric: bool = True,
    evolving_topk: int = 5,
    evolving_max_new: int = 3,
    evolving_system_prompt_path: Optional[str] = None,
    evolving_uniform_weights: bool = False,
    evolving_call_every: int = 1,
    output_dir: Optional[str] = None,
    **_,
) -> List[float]:
    """Drop-in replacement for `rubric_reward` that uses the evolving pipeline.

    Falls back to static `rubric_reward` for any prompt that the evolving
    pipeline cannot score.
    """
    if not use_evolving_rubric or rubric_path is None:
        return rubric_reward(
            completions=completions, kwargs_list=kwargs_list,
            ref_generate=ref_generate, log_dir=log_dir,
            reward_weight=reward_weight, rubric_path=rubric_path,
            rubric_judge_model=rubric_judge_model,
            rubric_workers=rubric_workers, rubric_neutral=rubric_neutral,
            rubric_judge_reasoning=rubric_judge_reasoning,
        )

    # Step gating: every N steps run evolving, else static
    global _step_counter
    with _step_counter_lock:
        _step_counter += 1
        step = _step_counter
    if evolving_call_every > 1 and (step % evolving_call_every != 0):
        return rubric_reward(
            completions=completions, kwargs_list=kwargs_list,
            ref_generate=ref_generate, log_dir=log_dir,
            reward_weight=reward_weight, rubric_path=rubric_path,
            rubric_judge_model=rubric_judge_model,
            rubric_workers=rubric_workers, rubric_neutral=rubric_neutral,
            rubric_judge_reasoning=rubric_judge_reasoning,
        )

    _load_rubrics(rubric_path)
    system_prompt = (_load_evolve_system_prompt(evolving_system_prompt_path)
                     if evolving_system_prompt_path else None)
    if system_prompt is None:
        # Cannot run evolving without prompt — full fallback
        return rubric_reward(
            completions=completions, kwargs_list=kwargs_list,
            ref_generate=ref_generate, log_dir=log_dir,
            reward_weight=reward_weight, rubric_path=rubric_path,
            rubric_judge_model=rubric_judge_model,
            rubric_workers=rubric_workers, rubric_neutral=rubric_neutral,
            rubric_judge_reasoning=rubric_judge_reasoning,
        )

    eff_reasoning = rubric_judge_reasoning
    if eff_reasoning is None and "gemini-3" in rubric_judge_model and "flash" in rubric_judge_model:
        eff_reasoning = "low"

    # Demux generation-major batch by problem_id.
    # Layout: entries [i, i+B, i+2B, ...] are N=8 rollouts of sample i.
    n_total = len(completions)
    rewards: List[float] = [0.0] * n_total

    groups: Dict[str, List[int]] = {}
    for i, kw in enumerate(kwargs_list):
        sid = kw.get("problem_id")
        if sid is None:
            sid = f"_idx{i}"
        groups.setdefault(sid, []).append(i)

    fallback_indices: List[int] = []
    tasks = []  # (sid, indices, question, audio_path, static_rubrics, completions_8, gt_letter)
    for sid, idxs in groups.items():
        if len(idxs) != 8:
            fallback_indices.extend(idxs)
            continue
        rec = _rr._rubric_store.get(sid)
        if rec is None:
            fallback_indices.extend(idxs)
            continue
        kw0 = kwargs_list[idxs[0]]
        question = kw0.get("problem", "")
        audio_path = kw0.get("audio_path", rec.get("audio_path", ""))
        # GT letter: prefer kw["answer"] (single letter); fall back to parsing
        # kw["solution"] for "<answer>X</answer>".
        gt_letter = (kw0.get("answer", "") or "").strip()
        if not gt_letter or len(gt_letter) > 1:
            sol = kw0.get("solution", "") or ""
            m = re.search(r"<answer>\s*([A-Fa-f])\s*</answer>", sol)
            if m:
                gt_letter = m.group(1)
        completions_8 = [completions[i][0]["content"] for i in idxs]
        tasks.append((sid, idxs, question, audio_path, rec["rubrics"],
                      completions_8, gt_letter))

    # Run per-prompt evolving in parallel
    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, rubric_workers)) as ex:
            fut_to_task = {
                ex.submit(
                    _evolve_one_prompt,
                    sid, q, ap, static, comps,
                    rubric_judge_model, eff_reasoning,
                    evolving_topk, evolving_max_new,
                    evolving_uniform_weights, system_prompt,
                    output_dir, step, gt,
                ): (sid, idxs)
                for (sid, idxs, q, ap, static, comps, gt) in tasks
            }
            for fut in as_completed(fut_to_task):
                sid, idxs = fut_to_task[fut]
                try:
                    scores = fut.result()
                except Exception as e:
                    import traceback
                    _log_evolve(output_dir, {
                        "step": step, "sample_id": sid,
                        "fallback": "exception_in_evolve_one",
                        "error": str(e)[:200],
                        "traceback": traceback.format_exc()[-500:],
                    })
                    scores = None
                if scores is not None and len(scores) == len(idxs):
                    for j, gi in enumerate(idxs):
                        rewards[gi] = scores[j] * reward_weight
                else:
                    fallback_indices.extend(idxs)

    # Fallback: run static rubric_reward on the failing slice
    if fallback_indices:
        fallback_indices_sorted = sorted(set(fallback_indices))
        fb_completions = [completions[i] for i in fallback_indices_sorted]
        fb_kwargs = [kwargs_list[i] for i in fallback_indices_sorted]
        fb_rewards = rubric_reward(
            completions=fb_completions, kwargs_list=fb_kwargs,
            ref_generate=ref_generate, log_dir=log_dir,
            reward_weight=reward_weight, rubric_path=rubric_path,
            rubric_judge_model=rubric_judge_model,
            rubric_workers=rubric_workers, rubric_neutral=rubric_neutral,
            rubric_judge_reasoning=rubric_judge_reasoning,
        )
        for j, gi in enumerate(fallback_indices_sorted):
            rewards[gi] = fb_rewards[j]
        _log_evolve(output_dir, {
            "step": step, "fallback_count": len(fallback_indices_sorted),
        })

    return rewards
