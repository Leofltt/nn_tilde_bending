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

    # Stable Audio / SAME-S
    sa_parser = subparsers.add_parser("stable_audio", help="Export Stable Audio Open / SAME-S autoencoder")
    sa_parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (if omitted, downloads from Hugging Face)")
    sa_parser.add_argument("--repo-id", type=str, default="stabilityai/SAME-S", help="Hugging Face repo id (default: stabilityai/SAME-S)")
    sa_parser.add_argument("--filename", type=str, default="same_s.pt", help="Checkpoint filename in HF repo")
    sa_parser.add_argument("--download-dir", type=str, default="models/checkpoints", help="Directory to save downloaded checkpoint")
    sa_parser.add_argument("--output", type=str, default="models/same_s_vae.ts", help="Output .ts file")
    sa_parser.add_argument("--sr", type=int, default=44100, help="Sample rate")
    sa_parser.add_argument("--ratio", type=int, default=4096, help="Downsampling ratio (4096 for SAME-S, 2048 for SA Open)")
    sa_parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimensionality")

    # Encodec
    enc_parser = subparsers.add_parser("encodec", help="Export Meta Encodec continuous autoencoder")
    enc_parser.add_argument("--model", type=str, default="encodec_24khz", help="Model name or checkpoint")
    enc_parser.add_argument("--output", type=str, default="models/encodec_continuous.ts", help="Output .ts file")
    enc_parser.add_argument("--channels", type=int, default=1, help="Channels (1 or 2)")
    enc_parser.add_argument("--ratio", type=int, default=320, help="Downsampling ratio")

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
        )
    elif args.command == "encodec":
        from python_tools.exporters.encodec_exporter import export_encodec
        export_encodec(args.model, args.output, args.channels, args.ratio)


if __name__ == "__main__":
    main()
