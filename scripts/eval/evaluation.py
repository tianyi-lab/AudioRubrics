import argparse
import json
from pathlib import Path
import re

# Letter labels used during generation (must match generate_answers_vllm.py)
_LETTERS = ["A", "B", "C", "D", "E", "F"]


def letter_correct(prediction: str, answer: str) -> bool:
    """Direct letter comparison (case-insensitive)."""
    return bool(prediction.strip()) and prediction.strip().upper() == answer.strip().upper()


def string_match(answer, prediction, choices):
    """
    Original token-level string matching from MMAU.
    Returns True if the prediction contains all answer tokens and none
    of the tokens that are exclusive to incorrect choices.
    """
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    
    if not prediction_tokens:
        return False
    
    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    
    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)
    
    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    
    return cond1 and cond2


def extract_answer(raw_prediction: str, choices: list, answer: str) -> tuple[str, bool]:
    """
    Extract the model's intended answer from raw_prediction and check
    whether it matches the ground-truth 'answer'.

    Strategy (applied in order):
      1. If the raw prediction IS already one of the choice strings
         (case-insensitive), use it directly.
      2. If the raw prediction starts with or contains a lone letter
         (A/B/C/D/…) map it to the corresponding choice text.
      3. Fall back to the MMAU string_match heuristic.

    Returns:
        (resolved_prediction: str, is_correct: bool)
    """
    raw = (raw_prediction or "").strip()
    choices = [c for c in choices if c is not None]

    # ---- 1. Direct match against choices (case-insensitive) ---------------
    for choice in choices:
        if raw.lower() == choice.lower():
            return choice, (choice.lower() == answer.lower())

    # ---- 2. Letter extraction (A/B/C/D …) ---------------------------------
    # Patterns:  "A", "A.", "A)", "A:", "A -", "(A)", "answer: A", "The answer is B"
    letter_patterns = [
        r"(?:^|[\s\(\[])([A-Fa-f])[\.\)\]:\s]",   # letter followed by punctuation/space
        r"(?:answer is|answer:|choose|option)\s*[\'\"]?([A-Fa-f])[\'\"]?",
        r"^([A-Fa-f])$",                            # bare single letter
    ]
    for pattern in letter_patterns:
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in _LETTERS:
                idx = _LETTERS.index(letter)
                if idx < len(choices):
                    resolved = choices[idx]
                    return resolved, (resolved.lower() == answer.lower())

    # ---- 3. String-match fallback ------------------------------------------
    matched = string_match(answer, raw, choices)
    return raw, matched

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process benchmark JSON and calculate accuracy.")
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file to be evaluated')
    
    args = parser.parse_args()  
    
    if args.input.endswith('json'):
        with open(args.input, 'r') as f:
            input_data = json.load(f)
    elif args.input.endswith('jsonl'):
        with open(args.input, 'r') as f:
            input_data = [json.loads(line.strip()) for line in f if line.strip()]
    else:
        # try as jsonl by default
        with open(args.input, 'r') as f:
            input_data = [json.loads(line.strip()) for line in f if line.strip()]

    corr, total = 0, 0

    # Track metrics for different categories:
    modality_metrics = {
        'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0],
        'mix-sound-music': [0, 0], 'mix-sound-speech': [0, 0],
        'mix-music-speech': [0, 0], 'mix-sound-music-speech': [0, 0],
    }
    category_metrics = {
        'Signal Layer': [0, 0], 'Perception Layer': [0, 0],
        'Semantic Layer': [0, 0], 'Cultural Layer': [0, 0],
    }
    
    # Here is the new dict for sub-category metrics
    subcat_metrics = {}

    output_key = 'answer_prediction'  # The key that contains model output
    no_pred_count = 0
    matched_outputs = []
    new_data = []

    # for idx, sample in enumerate(tqdm(input_data)):
    for idx, sample in enumerate(input_data):
        
        # If there's no model output key, skip
        if output_key not in sample:
            no_pred_count += 1
            continue
        
        _prediction = sample[output_key]
        if not _prediction:
            _prediction = ''
            no_pred_count += 1

        # Support both new format (letter in 'answer') and old format (choice text)
        _answer = sample.get('answer', '')
        choices  = sample.get('choices', [])
        modality = sample.get('modality', 'unknown')
        if isinstance(modality, list):
            modality = modality[0] if modality else 'unknown'
        category = sample.get('category', 'unknown')
        if isinstance(category, list):
            category = category[0] if category else 'unknown'
        subcat   = sample.get('sub-category', None)
        if isinstance(subcat, list):
            subcat = subcat[0] if subcat else None

        # ------------------------------------------------------------------
        # Correctness check
        # New format: both answer and answer_prediction are letters (A/B/C/D)
        # Old format: answer is choice text → fall back to extract_answer
        # ------------------------------------------------------------------
        is_letter = bool(re.fullmatch(r'[A-Fa-f]', _answer.strip())) if _answer else False
        _prediction = str(_prediction) if not isinstance(_prediction, str) else _prediction
        if is_letter:
            # Normalize prediction to letter:
            # case 1: already a letter
            pred_norm = _prediction.strip().upper()
            if not re.fullmatch(r'[A-F]', pred_norm):
                # case 2: prediction is choice text → map back to letter
                for i, c in enumerate(choices):
                    if _prediction.strip().lower() == str(c).lower():
                        pred_norm = _LETTERS[i] if i < len(_LETTERS) else _prediction.strip()
                        break
                else:
                    # case 3: extract letter via heuristics
                    m = re.search(r'\b([A-F])\b', _prediction.strip())
                    pred_norm = m.group(1) if m else _prediction.strip().upper()
            match_result = pred_norm == _answer.strip().upper()
            resolved_pred = pred_norm
        else:
            resolved_pred, match_result = extract_answer(_prediction, choices, _answer)
        sample['resolved_prediction'] = resolved_pred

        if modality not in modality_metrics:
            modality_metrics[modality] = [0, 0]
        if category not in category_metrics:
            category_metrics[category] = [0, 0]
        if subcat is not None and subcat not in subcat_metrics:
            subcat_metrics[subcat] = [0, 0]

        if match_result:
            modality_metrics[modality][0] += 1
            category_metrics[category][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            matched_outputs.append([_answer, resolved_pred])
            corr += 1
            sample['match'] = 1
        else:
            sample['match'] = 0

        total += 1
        new_data.append(sample)
        modality_metrics[modality][1] += 1
        category_metrics[category][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1


    # Print results:
    print("*"*30)
    print("Modality-wise Accuracy:")
    for modality in sorted(modality_metrics):
        n_correct, n_total = modality_metrics[modality]
        if n_total == 0:
            continue
        acc = (n_correct / n_total) * 100
        print(f"  {modality:35s}: {acc:.2f}%  ({n_correct}/{n_total})")
    
    print("*"*30)
    print("Category-wise Accuracy:")
    for category in sorted(category_metrics):
        n_correct, n_total = category_metrics[category]
        if n_total == 0:
            continue
        acc = (n_correct / n_total) * 100
        print(f"  {category:20s}: {acc:.2f}%  ({n_correct}/{n_total})")
    
    print("*"*30)
    print("Sub-category-wise Accuracy:")
    for subcat in sorted(subcat_metrics):
        n_correct, n_total = subcat_metrics[subcat]
        if n_total == 0:
            continue
        acc = (n_correct / n_total) * 100
        print(f"  {subcat:40s}: {acc:.2f}%  ({n_correct}/{n_total})")

    print("*"*30)
    print(f"Total Accuracy: {(corr/total) * 100:.2f}%  ({corr}/{total})")
    print("*"*30)
    print(f"No prediction count: {no_pred_count}")

    # Optionally save annotated results next to the input file
    input_p = Path(args.input)
    out_path = input_p.parent / (input_p.stem + "_evaluated.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in new_data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nAnnotated results saved → {out_path}")