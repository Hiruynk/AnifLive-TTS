"""Save an incremental raw PCM stream with the asynchronous SDK."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from aniflive_tts import AsyncClient, Language, RequiredFeature, SynthesisRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9880")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--language", choices=[item.value for item in Language], default="en")
    parser.add_argument("--output", type=Path, required=True, help="Destination raw PCM path")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    request = SynthesisRequest(
        language=Language(args.language),
        text=args.text,
    )
    async with AsyncClient(args.url) as client:
        await client.probe(required=(RequiredFeature.PCM_STREAMING,))
        stream = await client.stream(request)
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("wb") as output:
                async for chunk in stream:
                    output.write(chunk)
        finally:
            stream.close()

        print(
            f"Wrote {stream.sample_format} PCM, {stream.sample_rate} Hz, "
            f"{stream.channels} channel(s) to {args.output}"
        )


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
