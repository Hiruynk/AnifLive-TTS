"""Generate a complete WAV file with the synchronous AnifLive-TTS SDK."""

from __future__ import annotations

import argparse
from pathlib import Path

from aniflive_tts import Client, Language, SynthesisRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9880")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", choices=[item.value for item in Language], default="en")
    parser.add_argument("--output", type=Path, required=True, help="Destination WAV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = SynthesisRequest(
        language=Language(args.language),
        text=args.text,
    )

    with Client(args.url) as client:
        client.probe()
        result = client.synthesize(request)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result.audio)
    print(f"Wrote {len(result.audio)} WAV bytes to {args.output}")


if __name__ == "__main__":
    main()
