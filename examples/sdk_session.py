"""Synthesize a finite or live speech session to raw PCM."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from aniflive_tts import (
    AsyncClient,
    ContinuityPolicy,
    Language,
    RequiredFeature,
    SegmentRequest,
    SessionCreate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9880")
    parser.add_argument("--text", required=True, help="Text for one session segment")
    parser.add_argument("--language", choices=[item.value for item in Language], default="en")
    parser.add_argument("--output", type=Path, required=True, help="Destination raw PCM path")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Read session audio concurrently with appending the segment",
    )
    return parser.parse_args()


async def save_pcm(audio: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("wb") as output:
            async for chunk in audio:
                output.write(chunk)
    finally:
        # This closes the local audio response; session.close() is remote.
        audio.close()


async def run(args: argparse.Namespace) -> None:
    async with AsyncClient(args.url) as client:
        await client.probe(
            required=(RequiredFeature.PCM_STREAMING, RequiredFeature.SPEECH_SESSIONS)
        )
        session = await client.create_session(
            SessionCreate(continuity_policy=ContinuityPolicy.A)
        )
        segment = SegmentRequest(
            segment_id="example-1",
            text=args.text,
            language=Language(args.language),
        )
        reader: asyncio.Task[None] | None = None

        try:
            if args.live:
                # Start the only audio consumer before appending more work.
                audio = await session.audio()
                reader = asyncio.create_task(save_pcm(audio, args.output))
                await session.append(segment)
                await session.flush()
                await reader
            else:
                # Finite flow: append, flush, then open and drain the response.
                await session.append(segment)
                await session.flush()
                audio = await session.audio()
                await save_pcm(audio, args.output)
        finally:
            if reader is not None:
                if not reader.done():
                    reader.cancel()
                try:
                    await reader
                except asyncio.CancelledError:
                    pass
            # DELETE currently acts as remote cancel and retains the terminal record.
            await session.close()

    print(f"Wrote session PCM to {args.output}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
