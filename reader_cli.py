"""Reader CLI — Accessibility-focused text-to-speech interface.

Designed for blind users. Screen-reader friendly, no visual-only
elements. Supports stdin pipe, file reading, and keyboard controls.

Usage:
    python reader_cli.py --file book.txt
    cat article.txt | python reader_cli.py
    python reader_cli.py --text "Hello world"
    python reader_cli.py --clipboard
"""

import argparse
import os
import re
import sys
import time
from typing import List, Optional

import numpy as np

try:
    import soundfile as sf
except ImportError:
    raise ImportError("soundfile required: pip install soundfile")


def split_into_chunks(
    text: str,
    max_chars: int = 200,
) -> List[str]:
    """Split text into sentence-level chunks for streaming.

    Args:
        text: Full input text.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= max_chars:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks


def read_input(args: argparse.Namespace) -> str:
    """Read text input from various sources.

    Priority: --text flag > --file flag > --clipboard > stdin

    Args:
        args: Parsed arguments.

    Returns:
        Text string to read aloud.
    """
    if args.text:
        return args.text

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            return f.read()

    if args.clipboard:
        try:
            import subprocess
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return result.stdout
            # Fallback to xsel
            result = subprocess.run(
                ["xsel", "--clipboard", "--output"],
                capture_output=True, text=True,
            )
            return result.stdout
        except FileNotFoundError:
            print(
                "Error: xclip or xsel required for clipboard reading.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Read from stdin
    if not sys.stdin.isatty():
        return sys.stdin.read()

    print("MambaTTS Reader")
    print("Enter text to read (Ctrl+D to finish):")
    return sys.stdin.read()


def play_audio_chunk(
    audio: np.ndarray,
    sample_rate: int = 22050,
) -> None:
    """Play audio chunk using system audio.

    Falls back to saving temp file if direct playback unavailable.

    Args:
        audio: Audio waveform array.
        sample_rate: Sample rate.
    """
    tmp_path = "/tmp/mambatts_chunk.wav"
    sf.write(tmp_path, audio, sample_rate, subtype="PCM_16")

    # Try aplay (ALSA), then paplay (PulseAudio), then ffplay
    players = [
        ["aplay", "-q", tmp_path],
        ["paplay", tmp_path],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
    ]

    import subprocess
    for cmd in players:
        try:
            subprocess.run(cmd, check=True, timeout=30)
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    print(
        f"Warning: No audio player found. Audio saved to {tmp_path}",
        file=sys.stderr,
    )


def main() -> None:
    """Main CLI entry point for the MambaTTS reader."""
    parser = argparse.ArgumentParser(
        description="MambaTTS Reader — Text-to-Speech for Accessibility",
        epilog=(
            "Examples:\n"
            "  python reader_cli.py --text 'Hello world'\n"
            "  python reader_cli.py --file book.txt --emotion warm\n"
            "  cat article.txt | python reader_cli.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Text to read aloud"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Text file to read aloud"
    )
    parser.add_argument(
        "--clipboard", action="store_true",
        help="Read from clipboard"
    )
    parser.add_argument(
        "--emotion", type=str, default="warm",
        help="Voice emotion (neutral, happy, sad, warm, etc.)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Reading speed (0.5=slow, 1.0=normal, 1.5=fast)"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/stage2_final.pt",
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save full audio to file instead of playing"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=200,
        help="Max characters per synthesis chunk"
    )
    args = parser.parse_args()

    # Read input text
    text = read_input(args)
    if not text.strip():
        print("Error: No text provided.", file=sys.stderr)
        sys.exit(1)

    # Import synthesizer
    from synthesize import Synthesizer

    print("Loading MambaTTS model...", file=sys.stderr)
    synth = Synthesizer(
        checkpoint_path=args.checkpoint,
        device="cpu",
    )

    # Split into streamable chunks
    chunks = split_into_chunks(text, max_chars=args.chunk_size)
    total_chunks = len(chunks)

    print(
        f"Reading {len(text)} characters in {total_chunks} chunks "
        f"(emotion={args.emotion}, speed={args.speed}x)",
        file=sys.stderr,
    )

    all_audio = []

    for i, chunk in enumerate(chunks):
        print(
            f"  [{i+1}/{total_chunks}] {chunk[:50]}{'...' if len(chunk) > 50 else ''}",
            file=sys.stderr,
        )

        audio = synth.synthesize(
            text=chunk,
            emotion=args.emotion,
            speed=args.speed,
        )
        all_audio.append(audio)

        if not args.save:
            play_audio_chunk(audio)

    if args.save:
        # Concatenate and save
        full_audio = np.concatenate(all_audio)
        sf.write(args.save, full_audio, 22050, subtype="PCM_16")
        print(f"\nAudio saved → {args.save}", file=sys.stderr)
    else:
        print("\nDone reading.", file=sys.stderr)


if __name__ == "__main__":
    main()
