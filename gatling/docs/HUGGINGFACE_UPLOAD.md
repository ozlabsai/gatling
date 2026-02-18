# HuggingFace Dataset Upload Guide

## Overview

This guide covers uploading the Gatling adversarial dataset (563K samples, 761MB) to HuggingFace Hub under the OzLabs organization.

## Current Status

✅ **Dataset Generated**: 541,977 samples (96% of 563K target)
- File: `data/adversarial_563k.jsonl` (761MB)
- Metadata: `data/metadata_20260128_135514.json`
- Dataset Card: `data/README.md` (comprehensive documentation)

## Upload Process

### Step 1: Create Repository on HuggingFace (Manual)

Due to organization write permissions, you need to create the repo manually:

1. Go to: https://huggingface.co/new-dataset
2. Fill in:
   - **Owner**: Select "OzLabs" from dropdown
   - **Dataset name**: `gatling-adversarial-563k`
   - **License**: Apache 2.0
   - **Visibility**: Public
3. Click "Create dataset"

### Step 2: Upload Files (Automated Script)

Once the repo exists, run the upload script:

```bash
# From gatling/ directory
PYTHONPATH=. uv run python scripts/upload_to_huggingface.py \
  --repo OzLabs/gatling-adversarial-563k
```

This will upload:
1. `adversarial_563k.jsonl` (761MB) - The dataset
2. `README.md` - Dataset card with documentation
3. `metadata.json` - Generation metadata

**Expected Upload Time**: 5-10 minutes for 761MB file (depends on connection speed)

### Step 3: Verify Upload

After upload completes:

1. Visit: https://huggingface.co/datasets/OzLabs/gatling-adversarial-563k
2. Verify files are present:
   - ✅ `adversarial_563k.jsonl`
   - ✅ `README.md`
   - ✅ `metadata.json`
3. Check dataset card renders correctly
4. Test loading:

```python
from datasets import load_dataset

dataset = load_dataset("OzLabs/gatling-adversarial-563k")
print(f"Loaded {len(dataset['train'])} samples")
```

## Alternative: Upload via Web UI

If the script fails, you can upload manually:

1. Go to: https://huggingface.co/datasets/OzLabs/gatling-adversarial-563k/tree/main
2. Click "Add file" → "Upload files"
3. Drag and drop:
   - `data/adversarial_563k.jsonl`
   - `data/README.md`
   - `data/metadata_20260128_135514.json` (rename to `metadata.json`)
4. Commit with message: "Add Gatling adversarial dataset (541K samples)"

## Troubleshooting

### Issue: 403 Forbidden - No Write Permissions

**Cause**: Your HuggingFace token doesn't have write access to OzLabs org.

**Solutions**:
1. **Update Token Scope**:
   - Go to: https://huggingface.co/settings/tokens
   - Create new token with "Write" permission
   - Update `.env` file: `HF_TOKEN=hf_...`

2. **Request Org Permissions**:
   - Ask OzLabs org admin to grant you "write" role
   - Settings → Members → Your username → Change role to "write"

3. **Upload Manually** (see above)

### Issue: Upload Timeout

Large files (761MB) may timeout on slow connections.

**Solution**: Use `huggingface-cli` with resumable uploads:

```bash
huggingface-cli upload OzLabs/gatling-adversarial-563k \
  data/adversarial_563k.jsonl \
  adversarial_563k.jsonl \
  --repo-type dataset
```

### Issue: File Too Large

HuggingFace free tier supports files up to 5GB, so 761MB is fine.

If you get size errors:
1. Verify file integrity: `ls -lh data/adversarial_563k.jsonl`
2. Try uploading via web UI instead
3. Consider compressing: `gzip data/adversarial_563k.jsonl` (creates `.jsonl.gz`)

## Dataset Statistics

```json
{
  "timestamp": "20260128_135514",
  "target_samples": 563000,
  "actual_samples": 541977,
  "statistics": {
    "total_samples": 541977,
    "adversarial": 960,
    "benign": 541017,
    "adversarial_ratio": 0.0018,
    "by_source": {
      "prompt-injections": 546,
      "llmail-inject": 999,
      "prompt-injection-dataset": 257,
      "agent-harm": 176,
      "wildchat": 270000,
      "lmsys-chat": 269999
    }
  },
  "cost": "$0 (using HuggingFace datasets)",
  "strategy": "Real-world adversarial + benign samples from HF"
}
```

## Cost Analysis

- **This approach**: $0 (uses existing HuggingFace datasets)
- **Synthetic generation**: $40-60K (4M traces at $0.01-0.015/trace)
- **Savings**: 100% ($40-60K saved)

## Next Steps After Upload

1. Update `source/dataset/loaders.py` to optionally load from HF
2. Add CI test to verify HF loading works
3. Document usage in main README.md
4. Announce dataset availability to team
5. Move to DG-002: Policy Boundary Dataset (2M samples)

## References

- HuggingFace Datasets Hub: https://huggingface.co/docs/datasets/
- Dataset Card Guide: https://huggingface.co/docs/hub/datasets-cards
- Upload Large Files: https://huggingface.co/docs/huggingface_hub/guides/upload
