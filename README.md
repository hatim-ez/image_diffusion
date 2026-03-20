# Diffusion Image Playground

Minimal-yet-opinionated text-to-image diffusion training stack geared toward experimenting with LAION-style datasets on a single workstation.

## Features
- WebDataset streaming pipeline with tokenizer training utilities and NSFW filtering scripts.
- Config-driven experiments (YAML) selecting UNet or DiT-style transformer backbones.
- Mixed precision training with EMA tracking, TensorBoard + JSONL logging.
- Scheduler utilities (linear/cosine), DDPM/DDIM samplers, classifier-free guidance.
- Offline latent encoding pipeline using Stable Diffusion's VAE for lightweight training.
- Single-node multi-GPU training support via DDP with node-aware WebDataset shard splitting.

## Getting Started

1. **Create & activate a Python ≥3.9 virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e .
   ```

2. **Download LAION metadata**  
   Provide a text file with shard URLs and run:
   ```bash
   python scripts/download_laion_subset.py --url-file urls.txt --output-dir data/raw
   ```

3. **Filter + shard into WebDataset**
   ```bash
   python scripts/prepare_webdataset.py \
     --metadata data/raw/part-00000-....parquet \
     --output-dir data/webdataset/shards \
     --shard-size 1000 \
     --image-size 256
   ```

4. **Extract captions for tokenizer training**
   ```bash
   python scripts/extract_captions.py \
     --metadata data/raw/part-00000-....parquet \
     --output data/raw/captions.txt
   ```

5. **Train SentencePiece tokenizer**
   ```bash
   python scripts/train_tokenizer.py \
     --corpus data/raw/captions.txt \
     --output-dir tokenizer
   ```

6. **(Optional) Encode latents with Stable Diffusion VAE**
   ```bash
   python scripts/encode_latents.py \
     --shards 'data/webdataset/shards/{000000..000999}.tar' \
     --output-dir data/webdataset/latents \
     --vae runwayml/stable-diffusion-v1-5
   ```

## Training

Launch training with:
```bash
python train/train.py --config configs/unet_latent256.yaml
```

Key outputs:
- `logs/`: TensorBoard events + JSONL metrics, periodic sample grids, CLIP similarity stats.
- `checkpoints/step_xxxxxxx.pt`: model, optimizer, EMA, scheduler, tokenizer hash, config snapshot.

Resume from a checkpoint by setting `training.resume_from` inside the YAML or passing via CLI (edit config).

## Sampling
```bash
python scripts/sample.py \
  --config configs/unet_latent256.yaml \
  --checkpoint checkpoints/step_0005000.pt \
  --prompts configs/validation_prompts.txt \
  --outdir samples \
  --guidance-scale 7.5
```

## Metrics
- Training automatically logs CLIP similarity between validation prompts and generated samples.
- Additional offline metrics live in `metrics/` (FID/KID, CLIP score recompute). Example:
  ```bash
  python metrics/compute_clip_score.py --images samples --prompts configs/validation_prompts.txt
  python metrics/compute_fid.py --real data/fid_ref --fake samples
  ```

## Configuration Tips
- `dataset.image_size` should match the tensor resolution produced by the loader (latent resolution when `latent_mode=true`).
- `model.vae_model` must be provided when training/validating in latent mode so that evaluation can decode latents.
- Adjust `scheduler.total_steps` to align with `training.max_steps` for correct cosine annealing length.
- Use `training.eval_batch_size` and `validation_prompts` to tailor evaluation load to hardware limits.

## Next Steps
- Extend `models/transformer.py` with DiT-style classifier-free guidance conditioning blocks.
- Add gradient checkpointing toggles inside backbones to support larger models.
- Integrate additional schedulers (EDM, flow matching) or improved samplers (Heun, PLMS).

See `MODEL_CARD.md` for dataset disclosures, ethical considerations, and deployment notes.
