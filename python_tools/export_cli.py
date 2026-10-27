"""
Unified CLI for exporting neural audio models for nn_tilde_bending.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path so direct script execution works
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(
        description="Unified Exporter for nn_tilde_bending (Stable Audio, Encodec, RAVE, etc.)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stable Audio / Open VAE / SAME-S
    sa_parser = subparsers.add_parser("stable_audio", help="Export Stable Audio Open / 1.0 autoencoder (TorchScript or ONNX)")
    sa_parser.add_argument("-i", "--checkpoint", "--input", dest="checkpoint", type=str, default=None,
                           help="Path to checkpoint (.ts, .pt) or HF repo id (if omitted, reuses existing .ts or downloads from HF)")
    sa_parser.add_argument("--repo-id", type=str, default="stabilityai/stable-audio-open-1.0", help="Hugging Face repo id (default: stabilityai/stable-audio-open-1.0)")
    sa_parser.add_argument("--filename", type=str, default=None, help="Checkpoint filename in HF repo (optional)")
    sa_parser.add_argument("--download-dir", type=str, default="models/checkpoints/stable_audio_1_vae", help="Directory to save downloaded checkpoint")
    sa_parser.add_argument("-o", "--output", type=str, default=None, help="Output file (.ts or .onnx)")
    sa_parser.add_argument("--format", type=str, choices=["ts", "onnx", "both"], default=None, help="Format to export: ts, onnx, or both")
    sa_parser.add_argument("--onnx", action="store_true", help="Export as ONNX instead of TorchScript")
    sa_parser.add_argument("--no-bending", action="store_true", help="Don't promote parameters for ONNX bending")
    sa_parser.add_argument("--methods", nargs="+", default=["all"], help="Companion methods for ONNX: all, forward, encode, decode. Default: all")
    sa_parser.add_argument("--buffer-size", type=int, default=8192, help="Buffer size (default: 8192)")
    sa_parser.add_argument("--channels", type=int, default=2, help="Channels (default: 2)")
    sa_parser.add_argument("--sr", type=int, default=44100, help="Sample rate (default: 44100)")
    sa_parser.add_argument("--ratio", type=int, default=2048, help="Downsampling ratio (2048 for SA 1.0/Open, 4096 for SAME-S)")
    sa_parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimensionality (default: 64)")
    sa_parser.add_argument("--force", action="store_true", help="Force re-export even if model already exists")

    # Encodec
    enc_parser = subparsers.add_parser("encodec", help="Export Meta Encodec continuous autoencoder")
    enc_parser.add_argument("--model", type=str, default="encodec_24khz", help="Model name or checkpoint")
    enc_parser.add_argument("--output", type=str, default="models/encodec_continuous.ts", help="Output .ts or .onnx file")
    enc_parser.add_argument("--channels", type=int, default=1, help="Channels (1 or 2)")
    enc_parser.add_argument("--ratio", type=int, default=320, help="Downsampling ratio")
    enc_parser.add_argument("--onnx", action="store_true", help="Export as ONNX instead of TorchScript")
    enc_parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to export for ONNX (e.g. all, forward, encode, decode). Default: all",
    )
    enc_parser.add_argument("--no-bending", action="store_true", help="Don't promote parameters for ONNX bending")

    # RAVE Checkpoint Exporter (TS or ONNX)
    rave_ckpt_parser = subparsers.add_parser("rave", help="Export RAVE from checkpoint (.ckpt) to TorchScript or ONNX")
    rave_ckpt_parser.add_argument("-i", "--checkpoint", required=True, help="Path to RAVE checkpoint (.ckpt)")
    rave_ckpt_parser.add_argument("-o", "--output", required=True, help="Output file path (.ts or .onnx)")
    rave_ckpt_parser.add_argument("--onnx", action="store_true", help="Export as ONNX instead of TorchScript")
    rave_ckpt_parser.add_argument("--no-streaming", action="store_true", help="Disable streaming mode for TorchScript export")
    rave_ckpt_parser.add_argument("--no-bending", action="store_true", help="Skip promoting initializers for ONNX bending")
    rave_ckpt_parser.add_argument("--buffer-size", type=int, default=2048, help="Buffer size (default: 2048)")
    rave_ckpt_parser.add_argument("--channels", type=int, default=1, help="Input audio channels (default: 1)")

    # RAVE TorchScript to ONNX
    rave_parser = subparsers.add_parser("rave_to_onnx", help="Convert RAVE TorchScript (.ts) to bendable ONNX")
    rave_parser.add_argument("-i", "--input", required=True, help="Input TorchScript .ts model path")
    rave_parser.add_argument("-o", "--output", default=None, help="Output .onnx file path")
    rave_parser.add_argument("--buffer-size", type=int, default=2048, help="Buffer size (default: 2048)")
    rave_parser.add_argument("--channels", type=int, default=1, help="Input audio channels (default: 1)")
    rave_parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to export for ONNX (e.g. all, forward, encode, decode, prior). Default: all",
    )
    rave_parser.add_argument("--no-bending", action="store_true", help="Skip promoting initializers to graph inputs")

    # Stable Audio TorchScript to ONNX
    sa_to_onnx_parser = subparsers.add_parser("stable_audio_to_onnx", help="Convert Stable Audio TorchScript (.ts) to bendable ONNX")
    sa_to_onnx_parser.add_argument("-i", "--input", default="models/stable_audio_1_vae.ts", help="Input TorchScript .ts model path")
    sa_to_onnx_parser.add_argument("-o", "--output", default=None, help="Output .onnx file path")
    sa_to_onnx_parser.add_argument("--buffer-size", type=int, default=8192, help="Buffer size (default: 8192)")
    sa_to_onnx_parser.add_argument("--channels", type=int, default=2, help="Input audio channels (default: 2)")
    sa_to_onnx_parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to export for ONNX (e.g. all, forward, encode, decode). Default: all",
    )
    sa_to_onnx_parser.add_argument("--no-bending", action="store_true", help="Skip promoting initializers to graph inputs")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "stable_audio":
        from python_tools.exporters.stable_audio_exporter import export_stable_audio
        export_stable_audio(
            model_checkpoint_or_instance=args.checkpoint,
            output_path=args.output,
            sr=args.sr,
            latent_dim=args.latent_dim,
            ratio=args.ratio,
            repo_id=args.repo_id,
            filename=args.filename,
            download_dir=args.download_dir,
            to_onnx=args.onnx,
            format=args.format,
            promote_for_bending=not args.no_bending,
            methods=args.methods,
            buffer_size=args.buffer_size,
            channels=args.channels,
            force=args.force,
        )
    elif args.command == "stable_audio_to_onnx":
        import os
        from python_tools.stable_audio_ts_to_onnx import convert_stable_audio_ts_to_onnx
        out = args.output
        if not out:
            base, _ = os.path.splitext(args.input)
            out = f"{base}_bendable.onnx"
        convert_stable_audio_ts_to_onnx(
            ts_model_path=args.input,
            output_onnx_path=out,
            buffer_size=args.buffer_size,
            channels=args.channels,
            promote_for_bending=not args.no_bending,
            methods=args.methods,
        )
    elif args.command == "encodec":
        from python_tools.exporters.encodec_exporter import export_encodec
        export_encodec(
            model_or_pretrained=args.model,
            output_path=args.output,
            channels=args.channels,
            ratio=args.ratio,
            to_onnx=args.onnx,
            promote_for_bending=not args.no_bending,
            methods=args.methods,
        )
    elif args.command == "rave":
        from python_tools.exporters.rave_exporter import export_rave
        export_rave(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            to_onnx=args.onnx,
            streaming=not args.no_streaming,
            promote_for_bending=not args.no_bending,
            buffer_size=args.buffer_size,
            channels=args.channels,
        )
    elif args.command == "rave_to_onnx":
        import os
        from python_tools.rave_ts_to_onnx import convert_rave_ts_to_onnx
        out = args.output
        if not out:
            base, _ = os.path.splitext(args.input)
            out = f"{base}_bendable.onnx"
        convert_rave_ts_to_onnx(
            ts_model_path=args.input,
            output_onnx_path=out,
            buffer_size=args.buffer_size,
            channels=args.channels,
            promote_for_bending=not args.no_bending,
            methods=args.methods,
        )


if __name__ == "__main__":
    main()
