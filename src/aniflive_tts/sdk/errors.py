"""Public exception hierarchy for the AnifLive-TTS SDK.

Service messages and diagnostics remain available as attributes, but are deliberately
omitted from string and repr output because servers may echo request content.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class AnifLiveTTSError(Exception):
    """Base class for errors raised by the public SDK."""

    def __str__(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"


class ClientError(AnifLiveTTSError):
    """An error before or while interpreting an HTTP API response."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: str | None = None,
    ) -> None:
        self.message = message
        self.diagnostic = diagnostic
        super().__init__()

    def __str__(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<redacted>)"


class ClosedError(ClientError):
    """The local client or stream has been closed."""


class ConcurrentReadError(ClientError):
    """A stream already has a read in progress."""


class TransportError(ClientError):
    """A connection or transport failure."""


class TimeoutError(ClientError):
    """A client-side request timeout."""


class ProtocolError(ClientError):
    """The server response did not satisfy the expected protocol."""


class CompatibilityError(ClientError):
    """The service lacks a feature required by the caller."""


class APIError(AnifLiveTTSError):
    """An HTTP API failure with structured, explicitly inspectable details."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        server_code: Any = None,
        retry_after: str | None = None,
        diagnostic: str | None = None,
        response_body: bytes | str | Mapping[str, Any] | None = None,
    ) -> None:
        if not 100 <= status_code <= 599:
            raise ValueError("status_code must be a valid HTTP status")
        if retry_after is not None and not isinstance(retry_after, str):
            raise TypeError("retry_after must preserve the raw header string")

        self.status_code = status_code
        self.server_code = server_code
        self.message = message
        self.retry_after = retry_after
        self.diagnostic = diagnostic
        self.response_body = response_body
        super().__init__()

    def __str__(self) -> str:
        return f"{type(self).__name__} (HTTP {self.status_code})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(status_code={self.status_code}, details=<redacted>)"


class BadRequestError(APIError):
    """The request was invalid (HTTP 400)."""


class NotFoundError(APIError):
    """The requested resource was not found (HTTP 404)."""


class ConflictError(APIError):
    """The request conflicts with current service state (HTTP 409)."""


class PayloadTooLargeError(APIError):
    """The request exceeded a server limit (HTTP 413)."""


class BusyError(APIError):
    """The service is busy or rate limited (HTTP 429)."""


class ServerError(APIError):
    """The service failed while handling the request (HTTP 5xx)."""
