"""Voice-specific expression catalog calls preserve the no-argument API."""
import asyncio

from aniflive_tts.sdk.client import AsyncClient, Client


class FakeNative:
    def __init__(self):
        self.calls = []

    def expressions(self, *args, **kwargs):
        self.calls.append(("expressions", args, kwargs))
        return {"catalog": "sync"}

    async def expressions_async(self, *args, **kwargs):
        self.calls.append(("expressions_async", args, kwargs))
        return {"catalog": "async"}


def test_sync_expressions_keeps_no_argument_call_and_accepts_voice_profile():
    client = Client.__new__(Client)
    native = FakeNative()
    client._native = native

    assert client.expressions() == {"catalog": "sync"}
    assert client.expressions(voice_profile="voice +/one") == {"catalog": "sync"}
    assert native.calls == [
        ("expressions", (), {}),
        ("expressions", (), {"voice_profile": "voice +/one"}),
    ]


def test_async_expressions_keeps_no_argument_call_and_accepts_voice_profile():
    client = AsyncClient.__new__(AsyncClient)
    native = FakeNative()
    client._native = native
    client._loop = None

    async def scenario():
        assert await client.expressions() == {"catalog": "async"}
        assert await client.expressions(voice_profile="voice +/one") == {"catalog": "async"}

    asyncio.run(scenario())
    assert native.calls == [
        ("expressions_async", (), {}),
        ("expressions_async", (), {"voice_profile": "voice +/one"}),
    ]
