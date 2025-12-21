# Model Card Template

## Model Details
- **Model**: Configurable UNet / Transformer diffusion backbones (latent or pixel mode)
- **Developers**: Diffusion Image Playground contributors
- **Version**: 0.1.0
- **License**: Same as repository (specify before release)

## Intended Use
- Research experimentation on text-to-image diffusion using filtered LAION subsets.
- Not intended for production deployment without further safety & bias evaluation.

## Training Data
- Source: Public LAION-400M subset (filtered via `scripts/prepare_webdataset.py`).
- Filtering: NSFW labels removed, broken URLs skipped, captions cleaned/lowercased, text length clamped, images resized/cropped to 256px (or encoded to latents).
- Tokenizer: SentencePiece trained on cleaned captions (77-token context).

## Evaluation Data
- `configs/validation_prompts.txt` provides 30 curated prompts spanning diverse styles.
- Additional held-out LAION samples recommended for quant metrics (FID/KID).

## Metrics
- Training-time CLIP similarity (mean/std) for validation prompts.
- Offline scripts for FID/KID (`metrics/compute_fid.py`) and CLIP score recomputation.
- Recommend tracking aesthetic score / human evaluation before release.

## Ethical Considerations
- Even after NSFW filtering, LAION captions may contain biases or harmful content. Manual audits required.
- Generated imagery must be labeled synthetic; avoid impersonation or disinformation use-cases.
- Respect dataset licenses; do not redistribute raw LAION shards if not permitted.

## Limitations
- Single-node mixed precision setup; no out-of-the-box multi-GPU scaling.
- Text encoder defaults to frozen CLIP; prompt understanding may lag SOTA.
- Latent VAE is borrowed from Stable Diffusion; quality tied to upstream weights.

## Deployment Guidance
- Use `scripts/sample.py` or integrate `models.build_model` + `diffusion` utilities for inference.
- Provide user-facing filters (NSFW, watermarking) and rate limiting.
- Log prompts & metadata for auditing; obtain consent for downstream usage.
