"""
Base Exporter module for nn_tilde_bending.

Provides utility routines and tracing wrappers to convert arbitrary PyTorch audio
autoencoders (encoders and decoders) into TorchScript (.ts / .pt) modules conforming
to the nn~ streaming contract.
"""

import os
import logging
from typing import Optional, Sequence, Union
import torch
import torch.nn as nn

from python_tools.module import Module


class AudioCodecWrapper(Module):
    """
    Standard nn~ wrapper for audio autoencoders (RAVE, Stable Audio, Encodec, DAC).

    Exposes:
        - encode(x): (batch, in_channels, time) -> (batch, latent_dim, time // ratio)
        - decode(z): (batch, latent_dim, time // ratio) -> (batch, out_channels, time)
        - forward(x): (batch, in_channels, time) -> (batch, out_channels, time)
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        sr: int = 44100,
        in_channels: int = 2,
        out_channels: int = 2,
        latent_dim: int = 64,
        ratio: int = 2048,
        test_method: bool = False,
    ):
        super().__init__(sr=sr)
        self.encoder_module = encoder
        self.decoder_module = decoder
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.ratio = ratio

        # Register standard nn~ methods with temporal compression ratios
        self.register_method(
            "encode",
            in_channels=in_channels,
            in_ratio=1,
            out_channels=latent_dim,
            out_ratio=ratio,
            test_method=test_method,
        )
        self.register_method(
            "decode",
            in_channels=latent_dim,
            in_ratio=ratio,
            out_channels=out_channels,
            out_ratio=1,
            test_method=test_method,
        )
        self.register_method(
            "forward",
            in_channels=in_channels,
            in_ratio=1,
            out_channels=out_channels,
            out_ratio=1,
            test_method=test_method,
        )

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes audio to continuous latent space."""
        return self.encoder_module(x)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes continuous latents back to audio."""
        return self.decoder_module(z)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoding pass: encode -> decode."""
        z = self.encode(x)
        return self.decode(z)


def export_torchscript(
    model: Module,
    output_path: str,
    test_buffer_size: int = 8192,
) -> str:
    """
    Finalizes and scripts/traces the nn~ wrapper to a TorchScript .ts file.
    """
    model.eval()
    model.finish()

    # Remove deprecated/dynamic weight_norm hooks if present to avoid TorchScript tracing bug
    for mod in model.modules():
        try:
            torch.nn.utils.remove_weight_norm(mod)
        except (ValueError, AttributeError):
            pass

    # If submodules (encoder/decoder) have complex architectures that resist direct scripting,
    # trace them individually first so AudioCodecWrapper only sees compiled TorchScript modules.
    if hasattr(model, "encoder_module") and hasattr(model, "decoder_module"):
        dummy_audio = torch.zeros(1, model.in_channels, test_buffer_size)
        dummy_latent = torch.zeros(1, model.latent_dim, test_buffer_size // model.ratio)
        try:
            logging.info("Pre-tracing encoder and decoder submodules...")
            model.encoder_module = torch.jit.trace(model.encoder_module, dummy_audio, check_trace=False)
            model.decoder_module = torch.jit.trace(model.decoder_module, dummy_latent, check_trace=False)
            logging.info("Successfully pre-traced encoder and decoder.")
        except Exception as trace_err:
            logging.warning(f"Could not pre-trace submodules directly: {trace_err}")

    logging.info(f"Exporting model to TorchScript: {output_path}...")
    try:
        scripted = torch.jit.script(model)
    except Exception as e:
        logging.warning(f"torch.jit.script failed ({e}). Falling back to torch.jit.trace...")
        dummy_audio = torch.zeros(1, model.in_channels, test_buffer_size)
        dummy_latent = torch.zeros(1, model.latent_dim, test_buffer_size // model.ratio)

        try:
            scripted = torch.jit.trace_module(
                model,
                {
                    "forward": dummy_audio,
                    "encode": dummy_audio,
                    "decode": dummy_latent,
                },
                check_trace=False,
            )
        except Exception:
            scripted = torch.jit.trace(model, dummy_audio, check_trace=False)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    scripted.save(output_path)
    logging.info(f"Successfully exported: {output_path}")
    return output_path
