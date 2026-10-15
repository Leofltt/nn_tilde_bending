"""
Exporter for Meta Encodec and Descript Audio Codec (DAC).

Converts Encodec/DAC models into continuous latent autoencoders suitable for
real-time circuit bending and latent pad modulation in nn_tilde_bending.
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


class EncodecContinuousWrapper(AudioCodecWrapper):
    """
    Adapter for Meta Encodec in Continuous Latent Mode.
    Bypasses or fixes discrete RVQ quantization so that continuous latents (128-dim)
    can be warped and modulated on the 2D Latent Vector Pad.
    """

    def __init__(
        self,
        encodec_model: nn.Module,
        sr: int = 24000,
        channels: int = 1,
        ratio: int = 320,
        latent_dim: int = 128,
        use_quantizer: bool = False,
    ):
        class ContinuousEncoder(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.encoder = m.encoder

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [B, C, T]
                return self.encoder(x)

        class ContinuousDecoder(nn.Module):
            def __init__(self, m, quantize: bool):
                super().__init__()
                self.decoder = m.decoder
                self.quantizer = m.quantizer if quantize else None

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                # z: [B, D, T // ratio]
                if self.quantizer is not None:
                    # Optional quantizer pass
                    res = self.quantizer(z, frame_rate=24000 // 320)
                    z = res.quantized
                return self.decoder(z)

        super().__init__(
            encoder=ContinuousEncoder(encodec_model),
            decoder=ContinuousDecoder(encodec_model, use_quantizer),
            sr=sr,
            in_channels=channels,
            out_channels=channels,
            latent_dim=latent_dim,
            ratio=ratio,
            test_method=False,
        )


def export_encodec(
    model_or_pretrained: str = "encodec_24khz",
    output_path: str = "encodec_nn.ts",
    channels: int = 1,
    ratio: int = 320,
    latent_dim: int = 128,
):
    """
    Loads pretrained Encodec model from torch or checkpoint and exports to TorchScript.
    """
    try:
        from encodec import EncodecModel
        logging.info(f"Loading Encodec pretrained model: {model_or_pretrained}...")
        if model_or_pretrained == "encodec_24khz":
            model = EncodecModel.encodec_model_24khz()
            sr = 24000
        elif model_or_pretrained == "encodec_48khz":
            model = EncodecModel.encodec_model_48khz()
            sr = 48000
            channels = 2
            ratio = 640
        else:
            model = torch.load(model_or_pretrained, map_location="cpu")
            sr = getattr(model, "sample_rate", 24000)
    except ImportError:
        logging.warning("Package `encodec` not found. Expecting pre-instantiated torch model.")
        if isinstance(model_or_pretrained, nn.Module):
            model = model_or_pretrained
            sr = 24000
        else:
            raise ImportError("Please install encodec via `pip install encodec` to export.")

    model.set_target_bandwidth(6.0) if hasattr(model, "set_target_bandwidth") else None
    wrapper = EncodecContinuousWrapper(
        encodec_model=model,
        sr=sr,
        channels=channels,
        ratio=ratio,
        latent_dim=latent_dim,
    )
    return export_torchscript(wrapper, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Meta Encodec for nn_tilde_bending")
    parser.add_argument("--model", type=str, default="encodec_24khz", help="encodec_24khz or checkpoint")
    parser.add_argument("--output", type=str, default="encodec_continuous.ts", help="Output .ts file")
    parser.add_argument("--channels", type=int, default=1, help="Audio channels (1 or 2)")
    parser.add_argument("--ratio", type=int, default=320, help="Downsampling ratio")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    export_encodec(args.model, args.output, args.channels, args.ratio)
