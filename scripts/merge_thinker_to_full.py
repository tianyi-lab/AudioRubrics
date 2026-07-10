"""
Merge trained thinker weights with original talker/token2wav weights
to produce a full Qwen2_5OmniModel compatible with vllm.
"""
import argparse
import json
import shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", required=True)
parser.add_argument("--orig_dir", default="/home/jovyan/Omni-R1/models/Qwen/Qwen2.5-Omni-3B")
parser.add_argument("--out_dir",  required=True)
args = parser.parse_args()

CKPT_DIR = Path(args.ckpt_dir)
ORIG_DIR = Path(args.orig_dir)
OUT_DIR  = Path(args.out_dir)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Copy non-weight files from original (config, tokenizer, etc.) ──────────
for f in ORIG_DIR.iterdir():
    if f.is_dir():
        continue
    if f.suffix not in (".safetensors",) and f.name != "model.safetensors.index.json":
        dst = OUT_DIR / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            print(f"Copied {f.name}")

# ── 2. Load original index to know which orig files hold talker/token2wav ──────
with open(ORIG_DIR / "model.safetensors.index.json") as f:
    orig_index = json.load(f)
orig_weight_map = orig_index["weight_map"]

# Keys that are NOT thinker → keep from original
non_thinker_keys = {k: v for k, v in orig_weight_map.items()
                    if not k.startswith("thinker.")}
# Which original shard files contain non-thinker keys
non_thinker_shards = sorted(set(non_thinker_keys.values()))
print(f"Non-thinker shards: {non_thinker_shards}")

# ── 3. Load ckpt-200 index ────────────────────────────────────────────────────
with open(CKPT_DIR / "model.safetensors.index.json") as f:
    ckpt_index = json.load(f)
ckpt_weight_map = ckpt_index["weight_map"]
ckpt_shards = sorted(set(ckpt_weight_map.values()))
print(f"Checkpoint shards: {ckpt_shards}")

# ── 4. Write new thinker shards (add thinker. prefix) ─────────────────────────
# Map old shard name → new shard name
ckpt_shard_rename = {}
for i, shard in enumerate(ckpt_shards, 1):
    new_name = f"thinker-{shard}"  # e.g. thinker-model-00001-of-00002.safetensors
    ckpt_shard_rename[shard] = new_name

for shard, new_name in ckpt_shard_rename.items():
    out_path = OUT_DIR / new_name
    if out_path.exists():
        print(f"Skip (exists): {new_name}")
        continue
    print(f"Rewriting {shard} → {new_name} (adding thinker. prefix)...")
    tensors = load_file(CKPT_DIR / shard)
    new_tensors = {f"thinker.{k}": v for k, v in tensors.items()}
    save_file(new_tensors, out_path)
    print(f"  Saved {new_name} ({out_path.stat().st_size / 1e9:.2f} GB)")

# ── 5. Symlink original non-thinker shards ────────────────────────────────────
for shard in non_thinker_shards:
    dst = OUT_DIR / shard
    if not dst.exists():
        dst.symlink_to(ORIG_DIR / shard)
        print(f"Symlinked {shard}")

# ── 6. Build new index.json ───────────────────────────────────────────────────
new_weight_map = {}
# thinker keys → renamed ckpt shards
for key, shard in ckpt_weight_map.items():
    new_weight_map[f"thinker.{key}"] = ckpt_shard_rename[shard]
# non-thinker keys → original shards (symlinked)
for key, shard in non_thinker_keys.items():
    new_weight_map[key] = shard

new_index = {
    "metadata": orig_index.get("metadata", {}),
    "weight_map": new_weight_map,
}
with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
    json.dump(new_index, f, indent=2)
print(f"Written index.json with {len(new_weight_map)} keys")

print("\nDone! Merged model at:", OUT_DIR)