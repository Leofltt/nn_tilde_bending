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
    to_onnx: bool = False,
    promote_for_bending: bool = True,
):
    """
    Loads pretrained Encodec model from torch or checkpoint and exports to TorchScript or ONNX.
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
    wrapper.eval()

    if to_onnx:
        # Export to ONNX directly
        dummy_audio = torch.zeros(1, channels, 2048)
        base, ext = os.path.splitext(output_path)
        if ext != ".onnx":
            output_path = f"{base}.onnx"

        raw_onnx = f"{base}_raw.onnx" if promote_for_bending else output_path
        os.makedirs(os.path.dirname(os.path.abspath(raw_onnx)), exist_ok=True)

        logging.info(f"Exporting Encodec wrapper to standard ONNX: {raw_onnx}...")
        torch.onnx.export(
            wrapper,
            (dummy_audio,),
            raw_onnx,
            input_names=["audio_in"],
            output_names=["audio_out"],
            dynamic_axes={
                "audio_in": {0: "batch", 2: "samples"},
                "audio_out": {0: "batch", 2: "samples"},
            },
            opset_version=17,
        )

        if promote_for_bending:
            logging.info(f"Promoting Encodec parameters for NetBender real-time bending: {output_path}...")
            from python_tools.onnx_bend_prep import promote_initializers_to_inputs
            promote_initializers_to_inputs(raw_onnx, output_path, verbose=True)
            if raw_onnx != output_path and os.path.exists(raw_onnx):
                os.remove(raw_onnx)

        logging.info(f"Successfully exported Encodec ONNX: {output_path}")
        return output_path

    return export_torchscript(wrapper, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Meta Encodec for nn_tilde_bending")
    parser.add_argument("--model", type=str, default="encodec_24khz", help="encodec_24khz or checkpoint")
    parser.add_argument("--output", type=str, default="encodec_continuous.ts", help="Output .ts or .onnx file")
    parser.add_argument("--channels", type=int, default=1, help="Audio channels (1 or 2)")
    parser.add_argument("--ratio", type=int, default=320, help="Downsampling ratio")
    parser.add_argument("--onnx", action="store_true", help="Export as ONNX instead of TorchScript")
    parser.add_argument("--no-bending", action="store_true", help="Don't promote initializers to inputs in ONNX mode")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    export_encodec(
        model_or_pretrained=args.model,
        output_path=args.output,
        channels=args.channels,
        ratio=args.ratio,
        to_onnx=args.onnx,
        promote_for_bending=not args.no_bending,
    )
