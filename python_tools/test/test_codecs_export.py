"""
Self-contained unit and smoke test for python_tools exporters and nn~ metadata contract.
Validates:
1. Base AudioCodecWrapper export to TorchScript
2. Stable Audio VAE synthetic architecture export (Stereo 44.1 kHz, 64 latents, 2048 ratio)
3. Encodec continuous architecture export (Mono 24 kHz, 128 latents, 320 ratio)
4. Parameter buffers (encode_params, decode_params, forward_params)
5. Method signatures and shape round-trips
"""

import os
import shutil
import tempfile
import torch
import torch.nn as nn

from python_tools.exporters.base_exporter import AudioCodecWrapper, export_torchscript
from python_tools.exporters.stable_audio_exporter import StableAudioVAEWrapper
from python_tools.exporters.encodec_exporter import EncodecContinuousWrapper


class DummyConvNeXtSnakeEncoder(nn.Module):
    """Simulates a Stable Audio Open ConvNeXt/Snake encoder structure."""
    def __init__(self, in_channels=2, latent_dim=64, ratio=2048):
        super().__init__()
        # Conv kernel
        self.conv_in = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        # Snake-style activation parameter
        self.snake_alpha = nn.Parameter(torch.ones(32, 1))
        # Norm parameter
        self.norm = nn.GroupNorm(1, 32)
        # Downsample to ratio
        self.downsample = nn.AdaptiveAvgPool1d(1)
        self.proj_out = nn.Linear(32, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.conv_in(x)
        # Snake activation: x + (1/alpha) * sin^2(alpha * x)
        h = h + (1.0 / (self.snake_alpha.abs() + 1e-4)) * torch.sin(self.snake_alpha * h) ** 2
        h = self.norm(h)
        # Downsample along time dimension: T -> T // 2048
        target_t = max(1, x.size(-1) // 2048)
        h = nn.functional.adaptive_avg_pool1d(h, target_t)
        h = h.transpose(1, 2)
        h = self.proj_out(h).transpose(1, 2)
        return h


class DummyConvNeXtSnakeDecoder(nn.Module):
    """Simulates a Stable Audio Open ConvNeXt/Snake decoder structure."""
    def __init__(self, latent_dim=64, out_channels=2, ratio=2048):
        super().__init__()
        self.ratio = ratio
        self.proj_in = nn.Linear(latent_dim, 32)
        self.snake_alpha = nn.Parameter(torch.ones(32, 1))
        self.conv_out = nn.Conv1d(32, out_channels, kernel_size=7, padding=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim, T_latent]
        h = z.transpose(1, 2)
        h = self.proj_in(h).transpose(1, 2)
        # Upsample along time: T_latent -> T_latent * ratio
        h = nn.functional.interpolate(h, scale_factor=float(self.ratio), mode="nearest")
        h = h + (1.0 / (self.snake_alpha.abs() + 1e-4)) * torch.sin(self.snake_alpha * h) ** 2
        out = self.conv_out(h)
        return out


def test_stable_audio_export():
    print("\n--- [Test 1] Testing Stable Audio / SAME-S VAE Exporter ---")
    encoder = DummyConvNeXtSnakeEncoder(in_channels=2, latent_dim=64, ratio=2048)
    decoder = DummyConvNeXtSnakeDecoder(latent_dim=64, out_channels=2, ratio=2048)

    class DummyAutoencoder(nn.Module):
        def __init__(self, enc, dec):
            super().__init__()
            self.encoder = enc
            self.decoder = dec

    ae = DummyAutoencoder(encoder, decoder)
    wrapper = StableAudioVAEWrapper(ae, sr=44100, latent_dim=64, ratio=2048)

    with tempfile.TemporaryDirectory() as tmpdir:
        ts_path = os.path.join(tmpdir, "test_stable_audio.ts")
        export_torchscript(wrapper, ts_path)
        assert os.path.exists(ts_path), "TorchScript file was not created!"

        # Reload with torch.jit to simulate C++ LibTorch load
        loaded = torch.jit.load(ts_path)
        loaded.eval()

        # Check required nn~ metadata buffers
        encode_params = getattr(loaded, "encode_params").tolist()
        decode_params = getattr(loaded, "decode_params").tolist()
        forward_params = getattr(loaded, "forward_params").tolist()

        print(f"  encode_params:  {encode_params}")
        print(f"  decode_params:  {decode_params}")
        print(f"  forward_params: {forward_params}")

        assert encode_params == [2, 1, 64, 2048], f"Unexpected encode_params: {encode_params}"
        assert decode_params == [64, 2048, 2, 1], f"Unexpected decode_params: {decode_params}"
        assert forward_params == [2, 1, 2, 1], f"Unexpected forward_params: {forward_params}"

        # Test forward audio tensor inference (2048 audio samples)
        dummy_in = torch.randn(1, 2, 2048)
        out = loaded.forward(dummy_in)
        assert out.shape == (1, 2, 2048), f"Expected shape (1, 2, 2048), got {out.shape}"

        # Test encode -> decode
        latents = loaded.encode(dummy_in)
        assert latents.shape == (1, 64, 1), f"Expected shape (1, 64, 1), got {latents.shape}"
        recon = loaded.decode(latents)
        assert recon.shape == (1, 2, 2048), f"Expected shape (1, 2, 2048), got {recon.shape}"

        # Inspect named parameters for Snake activation layers
        param_names = [name for name, _ in loaded.named_parameters()]
        print(f"  Discovered layer names: {param_names}")
        assert any("snake" in p for p in param_names), "Snake parameter missing from exported model!"
        print("  ✓ Stable Audio export passed successfully!")


class DummyEncodecModel(nn.Module):
    """Simulates an Encodec continuous encoder/decoder with LSTM layers."""
    def __init__(self, in_channels=1, latent_dim=128, ratio=320):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.LSTM(32, latent_dim, batch_first=True)[0] if False else nn.Conv1d(32, latent_dim, kernel_size=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 32, kernel_size=1),
            nn.Conv1d(32, in_channels, kernel_size=7, stride=1, padding=3)
        )
        self.quantizer = None

    def forward(self, x):
        return x


def test_encodec_export():
    print("\n--- [Test 2] Testing Encodec Continuous Exporter ---")
    enc_model = DummyEncodecModel(in_channels=1, latent_dim=128, ratio=320)
    wrapper = EncodecContinuousWrapper(enc_model, sr=24000, channels=1, ratio=320, latent_dim=128)

    with tempfile.TemporaryDirectory() as tmpdir:
        ts_path = os.path.join(tmpdir, "test_encodec.ts")
        export_torchscript(wrapper, ts_path)
        assert os.path.exists(ts_path), "TorchScript file was not created!"

        loaded = torch.jit.load(ts_path)
        loaded.eval()

        encode_params = getattr(loaded, "encode_params").tolist()
        decode_params = getattr(loaded, "decode_params").tolist()

        print(f"  encode_params:  {encode_params}")
        print(f"  decode_params:  {decode_params}")

        assert encode_params == [1, 1, 128, 320], f"Unexpected encode_params: {encode_params}"
        assert decode_params == [128, 320, 1, 1], f"Unexpected decode_params: {decode_params}"
        print("  ✓ Encodec continuous export passed successfully!")


if __name__ == "__main__":
    test_stable_audio_export()
    test_encodec_export()
    print("\n==========================================")
    print("ALL MULTI-MODEL CODEC EXPORT TESTS PASSED!")
    print("==========================================")
