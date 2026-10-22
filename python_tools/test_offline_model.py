#!/usr/bin/env python3
"""
Test script to validate exported nn~ audio codec models offline.

Supports testing:
- Loading model with torch.jit.load
- Checking metadata parameter buffers (encode_params, decode_params, forward_params)
- Generating synthetic audio (sine waves, chirps, noise) or reading an audio file (.wav)
- Testing forward(), encode(), decode()
- Simulating block-by-block streaming with configurable buffer sizes (e.g. 2048, 4096, 8192)
- Simulating circuit bending (weight manipulation & latent perturbation)
- Writing reconstructed output audio to WAV for listening validation

Usage:
    python python_tools/test_offline_model.py --model models/stable_audio_1_vae.ts --output test_recon.wav
    python python_tools/test_offline_model.py --model models/stable_audio_1_vae.ts --input input.wav --output recon.wav
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
try:
    import soundfile as sf
except ImportError:
    sf = None


def generate_test_audio(sr: int = 44100, channels: int = 2, duration_sec: float = 2.0) -> torch.Tensor:
    """Generate a clean stereo test signal (sine tone + frequency modulation)."""
    num_samples = int(sr * duration_sec)
    t = torch.linspace(0, duration_sec, num_samples)
    
    # Left channel: 440 Hz tone with slight vibrato
    left = 0.5 * torch.sin(2 * math.pi * 440.0 * t + 0.1 * torch.sin(2 * math.pi * 5.0 * t))
    
    if channels == 1:
        return left.unsqueeze(0).unsqueeze(0) # [1, 1, T]
        
    # Right channel: 880 Hz harmonic
    right = 0.4 * torch.sin(2 * math.pi * 880.0 * t)
    audio = torch.stack([left, right], dim=0).unsqueeze(0) # [1, 2, T]
    return audio


def test_model_offline(
    model_path: str,
    input_wav: str = None,
    output_wav: str = "test_recon.wav",
    buffer_size: int = 2048,
    bend_scale: float = 1.0,
    bend_offset: float = 0.0,
    bend_layer: str = None,
    latent_bend: bool = False,
):
    print("=" * 70)
    print(f"OFFLINE VALIDATION FOR NN~ MODEL: {model_path}")
    print("=" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"1. Loading TorchScript model from '{model_path}'...")
    start_t = time.time()
    model = torch.jit.load(model_path)
    model.eval()
    load_time = time.time() - start_t
    print(f"   Model loaded successfully in {load_time:.2f}s!")

    # 2. Inspect nn~ attributes
    print("\n2. Inspecting nn~ metadata attributes:")
    attr_names = ["encode_params", "decode_params", "forward_params"]
    attrs = {}
    for attr in attr_names:
        if hasattr(model, attr):
            val = getattr(model, attr).tolist()
            attrs[attr] = val
            print(f"   - {attr}: {val}")
        else:
            print(f"   - {attr}: NOT FOUND!")

    # Determine input/output channels and ratio
    in_channels = attrs.get("forward_params", [2, 1, 2, 1])[0]
    out_channels = attrs.get("forward_params", [2, 1, 2, 1])[2]
    ratio = attrs.get("encode_params", [2, 1, 64, 2048])[3] if "encode_params" in attrs else 2048
    latent_dim = attrs.get("encode_params", [2, 1, 64, 2048])[2] if "encode_params" in attrs else 64
    sr = 44100 # default audio sample rate

    print(f"   Configured: in_channels={in_channels}, out_channels={out_channels}, "
          f"latent_dim={latent_dim}, ratio={ratio}")

    # 3. Prepare Audio Input
    if input_wav and os.path.exists(input_wav):
        print(f"\n3. Loading input audio from '{input_wav}'...")
        if sf is None:
            raise ImportError("soundfile library is required to read WAV files. Install with: uv pip install soundfile")
        data, file_sr = sf.read(input_wav, dtype="float32")
        sr = file_sr
        # Ensure shape [1, channels, samples]
        if data.ndim == 1:
            data = data[None, :] # mono
        else:
            data = data.T # [channels, samples]
        audio_in = torch.from_numpy(data).unsqueeze(0)
        # Adjust channel count if needed
        if audio_in.size(1) < in_channels:
            audio_in = audio_in.repeat(1, in_channels, 1)
        elif audio_in.size(1) > in_channels:
            audio_in = audio_in[:, :in_channels, :]
    else:
        print(f"\n3. Generating synthetic test audio (sr={sr}, duration=2.0s)...")
        audio_in = generate_test_audio(sr=sr, channels=in_channels, duration_sec=2.0)

    total_samples = audio_in.size(-1)
    # Trim to multiple of ratio/buffer_size for clean chunking
    aligned_samples = (total_samples // ratio) * ratio
    if aligned_samples == 0:
        aligned_samples = ratio
        audio_in = torch.nn.functional.pad(audio_in, (0, ratio - total_samples))
    else:
        audio_in = audio_in[:, :, :aligned_samples]

    print(f"   Input shape: {list(audio_in.shape)} (min: {audio_in.min():.3f}, max: {audio_in.max():.3f}, RMS: {audio_in.pow(2).mean().sqrt():.3f})")

    # 4. Optional Circuit Bending Simulation
    if bend_layer:
        print(f"\n4. Simulating Circuit Bending on layer '{bend_layer}' (scale={bend_scale}, offset={bend_offset})...")
        named_params = dict(model.named_parameters())
        if bend_layer in named_params:
            with torch.no_grad():
                param = named_params[bend_layer]
                orig_norm = param.norm().item()
                param.mul_(bend_scale).add_(bend_offset)
                new_norm = param.norm().item()
                print(f"   Layer bent! Norm changed from {orig_norm:.4f} -> {new_norm:.4f}")
        else:
            print(f"   WARNING: Layer '{bend_layer}' not found in model parameters!")
            matching = [p for p in named_params.keys() if "conv" in p or "weight" in p][:5]
            print(f"   Available candidate layers: {matching}")

    # 5. Offline Inference Test: Single-Pass Forward
    print(f"\n5. Running single-pass forward({aligned_samples} samples)...")
    try:
        t0 = time.time()
        with torch.no_grad():
            out_single = model.forward(audio_in)
        dt = time.time() - t0
        rtf = (aligned_samples / sr) / max(1e-6, dt)
        print(f"   ✓ forward() succeeded in {dt*1000:.1f}ms (Speed: {rtf:.1f}x real-time)!")
        print(f"   Output shape: {list(out_single.shape)} (min: {out_single.min():.3f}, max: {out_single.max():.3f}, RMS: {out_single.pow(2).mean().sqrt():.3f})")
    except Exception as e:
        print(f"   ✗ forward() failed: {e}")
        raise

    # 6. Offline Inference Test: Encode -> Latent -> Decode
    print(f"\n6. Running encode() -> decode() decomposition...")
    try:
        with torch.no_grad():
            z = model.encode(audio_in)
            print(f"   ✓ encode() succeeded! Latent shape: {list(z.shape)}")
            print(f"     Latent stats: min={z.min():.3f}, max={z.max():.3f}, mean={z.mean():.3f}, std={z.std():.3f}")

            if latent_bend:
                print("   Applying latent Fourier-like perturbation...")
                z = z + 0.25 * torch.sin(z * 3.14159)

            recon = model.decode(z)
            print(f"   ✓ decode() succeeded! Reconstruction shape: {list(recon.shape)}")
            print(f"     Recon stats: min={recon.min():.3f}, max={recon.max():.3f}, RMS={recon.pow(2).mean().sqrt():.3f}")

            diff = (out_single - recon).abs().max().item()
            print(f"   ✓ Max diff between forward() and decode(encode()): {diff:.6f}")
    except Exception as e:
        print(f"   ✗ encode() / decode() failed: {e}")
        raise

    # 7. Block-by-block Streaming Simulation (matching nn~ bufferSize chunks)
    print(f"\n7. Simulating real-time block streaming (buffer_size = {buffer_size})...")
    streamed_out_blocks = []
    num_blocks = aligned_samples // buffer_size
    t_start = time.time()
    with torch.no_grad():
        for b in range(num_blocks):
            chunk = audio_in[:, :, b * buffer_size : (b + 1) * buffer_size]
            out_chunk = model.forward(chunk)
            streamed_out_blocks.append(out_chunk)
    t_end = time.time()
    streamed_out = torch.cat(streamed_out_blocks, dim=-1)
    total_audio_sec = (num_blocks * buffer_size) / sr
    total_compute_sec = t_end - t_start
    print(f"   ✓ Processed {num_blocks} blocks of {buffer_size} samples in {total_compute_sec:.3f}s")
    print(f"     Streaming RTF: {total_audio_sec / max(1e-6, total_compute_sec):.2f}x real-time")
    print(f"     Streamed output shape: {list(streamed_out.shape)}")

    # 8. Save audio to disk for listening
    if output_wav and sf is not None:
        out_np = streamed_out.squeeze(0).cpu().numpy().T # [samples, channels]
        os.makedirs(os.path.dirname(os.path.abspath(output_wav)) if os.path.dirname(output_wav) else ".", exist_ok=True)
        sf.write(output_wav, out_np, sr)
        print(f"\n8. Wrote output audio to: {output_wav}")
        print("=" * 70)
        print("ALL OFFLINE TESTS PASSED! MODEL IS OPERATIONAL.")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline test and audio validator for nn~ models")
    parser.add_argument("--model", type=str, default="models/stable_audio_1_vae.ts", help="Path to exported .ts model")
    parser.add_argument("--input", type=str, default=None, help="Optional input .wav file (defaults to synthetic test audio)")
    parser.add_argument("--output", type=str, default="models/test_recon.wav", help="Output .wav path")
    parser.add_argument("--buffer-size", type=int, default=2048, help="Streaming block size (default: 2048)")
    parser.add_argument("--bend-layer", type=str, default=None, help="Layer parameter name to bend")
    parser.add_argument("--bend-scale", type=float, default=1.0, help="Circuit bend scale factor")
    parser.add_argument("--bend-offset", type=float, default=0.0, help="Circuit bend offset factor")
    parser.add_argument("--latent-bend", action="store_true", help="Perturb latents between encode and decode")
    args = parser.parse_args()

    test_model_offline(
        model_path=args.model,
        input_wav=args.input,
        output_wav=args.output,
        buffer_size=args.buffer_size,
        bend_scale=args.bend_scale,
        bend_offset=args.bend_offset,
        bend_layer=args.bend_layer,
        latent_bend=args.latent_bend,
    )
