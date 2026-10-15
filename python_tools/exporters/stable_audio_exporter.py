"""
Exporter for Stable Audio 1.0 / Stable Audio Open VAE (Oobleck / DAC architecture).

Wraps the continuous latent convolutional variational autoencoder into the nn~
streaming format for real-time circuit bending in nn_tilde_bending.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path so direct script execution works
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn

from python_tools.exporters.base_exporter import AudioCodecWrapper, export_torchscript


class OobleckEncoderAdapter(nn.Module):
    """
    Adapts AutoencoderOobleck / DAC encoder to pure tensor I/O:
    Input: [B, C, T] -> Output: [B, latent_dim, T // ratio]
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        # Handle dict / dataclass or raw tensor
        if hasattr(out, "latent"):
            return out.latent
        if hasattr(out, "sample"):
            return out.sample
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class OobleckDecoderAdapter(nn.Module):
    """
    Adapts AutoencoderOobleck / DAC decoder to pure tensor I/O:
    Input: [B, latent_dim, T // ratio] -> Output: [B, C, T]
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        # Handle dict / dataclass or raw tensor
        if hasattr(out, "sample"):
            return out.sample
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class StableAudioVAEWrapper(AudioCodecWrapper):
    """
    nn~ wrapper for Stable Audio Open / 1.0 VAE (Stereo 44.1 kHz, 64-dim latents, 2048x ratio).
    """
    def __init__(
        self,
        autoencoder_or_vae: nn.Module,
        sr: int = 44100,
        latent_dim: int = 64,
        ratio: int = 2048,
    ):
        encoder = None
        decoder = None

        # 1. Check for diffusers AutoencoderOobleck
        if hasattr(autoencoder_or_vae, "encoder") and hasattr(autoencoder_or_vae, "decoder"):
            encoder = OobleckEncoderAdapter(autoencoder_or_vae.encoder)
            decoder = OobleckDecoderAdapter(autoencoder_or_vae.decoder)
        elif hasattr(autoencoder_or_vae, "model"):
            target = autoencoder_or_vae.model
            if hasattr(target, "encoder") and hasattr(target, "decoder"):
                encoder = OobleckEncoderAdapter(target.encoder)
                decoder = OobleckDecoderAdapter(target.decoder)

        if encoder is None or decoder is None:
            raise ValueError(
                "Could not extract encoder/decoder from provided autoencoder model."
            )

        # Enable cached_conv streaming if available (eliminates boundary clicking artifacts)
        try:
            import cached_conv as cc
            logging.info("Enabling cached_conv for seamless streaming...")
            cc.use_cached_conv(True)
        except Exception as cc_err:
            logging.info(f"Proceeding without cached_conv: {cc_err}")

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            sr=sr,
            in_channels=2,
            out_channels=2,
            latent_dim=latent_dim,
            ratio=ratio,
            test_method=False,
        )


def load_stable_audio_1_vae(
    pretrained_name_or_path: str = "stabilityai/stable-audio-open-1.0",
    download_dir: str = "models/checkpoints/stable_audio_1_vae",
) -> nn.Module:
    """
    Loads the Stable Audio 1.0 / Open VAE (AutoencoderOobleck) via diffusers
    or local directory.
    """
    try:
        from diffusers import AutoencoderOobleck
        logging.info(f"Loading AutoencoderOobleck from '{pretrained_name_or_path}' (subfolder='vae')...")
        os.makedirs(download_dir, exist_ok=True)
        vae = AutoencoderOobleck.from_pretrained(
            pretrained_name_or_path,
            subfolder="vae",
            cache_dir=download_dir,
        )
        vae.eval()
        return vae
    except ImportError:
        raise ImportError(
            "diffusers is required to export Stable Audio 1.0 / Open VAE.\n"
            "Install it via: uv pip install diffusers"
        )


def export_stable_audio(
    model_checkpoint_or_instance=None,
    output_path: str = "models/stable_audio_1_vae.ts",
    sr: int = 44100,
    latent_dim: int = 64,
    ratio: int = 2048,
    repo_id: str = "stabilityai/stable-audio-open-1.0",
    download_dir: str = "models/checkpoints/stable_audio_1_vae",
):
    """
    Exports Stable Audio 1.0 / Open VAE to TorchScript (.ts).
    """
    if model_checkpoint_or_instance is None or isinstance(model_checkpoint_or_instance, str):
        path = model_checkpoint_or_instance or repo_id
        vae = load_stable_audio_1_vae(path, download_dir=download_dir)
    else:
        vae = model_checkpoint_or_instance

    wrapper = StableAudioVAEWrapper(
        autoencoder_or_vae=vae,
        sr=sr,
        latent_dim=latent_dim,
        ratio=ratio,
    )
    return export_torchscript(wrapper, output_path, test_buffer_size=8192)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Stable Audio 1.0 / Open VAE for nn_tilde_bending")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path or HF repo id (default: stabilityai/stable-audio-open-1.0)")
    parser.add_argument("--repo-id", type=str, default="stabilityai/stable-audio-open-1.0", help="Hugging Face repo id")
    parser.add_argument("--output", type=str, default="models/stable_audio_1_vae.ts", help="Output .ts file")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate")
    parser.add_argument("--ratio", type=int, default=2048, help="Temporal compression ratio (2048 for SA 1.0/Open)")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimensionality")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    export_stable_audio(
        model_checkpoint_or_instance=args.checkpoint or args.repo_id,
        output_path=args.output,
        sr=args.sr,
        latent_dim=args.latent_dim,
        ratio=args.ratio,
        repo_id=args.repo_id,
    )
