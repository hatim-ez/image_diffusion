from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from diffusion_image.checkpointing import load_checkpoint, save_checkpoint
from diffusion_image.config import DiffusionConfig, load_config
from diffusion_image.diffusion import DiffusionProcess, ddim_sample, ddpm_sample


ROOT = Path(__file__).resolve().parents[1]


class ZeroNoiseModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps, tokens, mask
        return torch.zeros_like(x)


class SmokeTests(unittest.TestCase):
    def test_example_configs_load(self) -> None:
        unet_cfg = load_config(ROOT / "configs/unet_latent256.yaml")
        transformer_cfg = load_config(ROOT / "configs/transformer_latent256.yaml")

        self.assertEqual(unet_cfg.model.architecture, "unet")
        self.assertTrue(unet_cfg.dataset.latent_mode)
        self.assertEqual(transformer_cfg.model.architecture, "transformer")
        self.assertEqual(transformer_cfg.diffusion.sampler, "ddim")

    def test_checkpoint_roundtrip(self) -> None:
        model = nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        ema_state = {"decay": 0.999, "shadow": {"weight": torch.zeros_like(model.weight)}}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                path=path,
                step=7,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ema_state=ema_state,
                tokenizer_hash="abc123",
                config={"seed": 42},
            )
            payload = load_checkpoint(path)

        self.assertEqual(payload["step"], 7)
        self.assertEqual(payload["tokenizer_hash"], "abc123")
        self.assertEqual(payload["config"]["seed"], 42)
        self.assertIn("model", payload)
        self.assertIn("optimizer", payload)
        self.assertIn("scheduler", payload)

    def test_diffusion_samplers_preserve_shape(self) -> None:
        model = ZeroNoiseModel()
        diffusion = DiffusionProcess(
            DiffusionConfig(schedule="linear", timesteps=8, beta_start=1e-4, beta_end=0.02, ddim_steps=4)
        )
        shape = (2, 4, 8, 8)
        tokens = torch.zeros((2, 8), dtype=torch.long)
        mask = torch.ones_like(tokens)

        ddim = ddim_sample(model, diffusion, shape, tokens, mask, num_steps=4)
        ddpm = ddpm_sample(model, diffusion, shape, tokens, mask)

        self.assertEqual(ddim.shape, shape)
        self.assertEqual(ddpm.shape, shape)

    @unittest.skipUnless(importlib.util.find_spec("sentencepiece") is not None, "sentencepiece is not installed")
    def test_tokenizer_loads_repo_model(self) -> None:
        from data.tokenizer import SentencePieceTokenizer

        tokenizer = SentencePieceTokenizer(ROOT / "tokenizer/tokenizer.model", max_length=8)
        ids = tokenizer.encode("A short prompt")

        self.assertEqual(len(ids), 8)
        self.assertEqual(tokenizer.pad_id, 0)
        self.assertEqual(tokenizer.null_prompt_id, 4)


if __name__ == "__main__":
    unittest.main()
