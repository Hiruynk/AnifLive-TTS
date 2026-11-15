"""Register a zero-shot Voice and synthesize with the shared V2ProPlus engine."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from aniflive_tts import Client, ReferenceCreate, RequiredFeature, VoiceCreate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9880")
    parser.add_argument("--name", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--reference-language", choices=("zh", "yue", "en", "ja", "ko"), required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", choices=("zh", "yue", "en", "ja", "ko"), required=True)
    parser.add_argument("--output", type=Path, default=Path("speech.wav"))
    args = parser.parse_args()

    with Client(args.url) as client:
        client.probe(required=(RequiredFeature.VOICE_MANAGEMENT,))
        voice = client.create_voice(VoiceCreate(name=args.name))
        print(f"Voice ID: {voice.id}", flush=True)
        reference = client.upload_reference(
            voice.id,
            args.reference.read_bytes(),
            ReferenceCreate(
                reference_language=args.reference_language,
                transcript=args.transcript,
                is_default=True,
            ),
            filename=args.reference.name,
        )
        print(f"Reference ID: {reference.id}", flush=True)
        client.prepare_reference(voice.id, reference.id)
        deadline = time.monotonic() + 120
        while True:
            reference = client.reference(voice.id, reference.id)
            if reference.prepare_state == "ready":
                break
            if reference.prepare_state == "failed":
                raise RuntimeError(f"Reference preparation failed: {reference.error_code}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Reference preparation is still pending; inspect its state")
            time.sleep(0.5)
        result = client.synthesize(
            text=args.text, language=args.language, voice_profile=voice.id,
        )
        args.output.write_bytes(result.audio)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
