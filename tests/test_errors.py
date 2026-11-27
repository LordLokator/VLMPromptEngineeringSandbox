import json

import pytest

from utils.errors import ERROR_KINDS, DecodeError, classify_error


class StatusError(Exception):
    def __init__(self, status):
        super().__init__("secret body")
        self.status_code = status


class ResponseError(Exception):
    def __init__(self, status):
        super().__init__("secret body")
        self.response = type("R", (), {"status_code": status})()


class APIConnectionError(Exception):
    pass


class APITimeoutError(APIConnectionError):
    pass


class AuthenticationError(Exception):
    pass


class RateLimitError(Exception):
    pass


class OurDecode(DecodeError, ValueError):
    pass


@pytest.mark.parametrize(
    "error, kind",
    [
        (StatusError(401), "auth"),
        (StatusError(403), "auth"),
        (StatusError(429), "rate_limit"),
        (StatusError(408), "timeout"),
        (StatusError(400), "bad_request"),
        (ResponseError(404), "bad_request"),
        (StatusError(500), "internal"),
        (AuthenticationError("secret body"), "auth"),
        (RateLimitError("secret body"), "rate_limit"),
        (APITimeoutError("secret body"), "timeout"),
        (TimeoutError("secret body"), "timeout"),
        (APIConnectionError("secret body"), "connection"),
        (ConnectionRefusedError("secret body"), "connection"),
        (OurDecode("secret body"), "decode"),
        (json.JSONDecodeError("secret body", "x", 0), "decode"),
        (RuntimeError("secret body"), "internal"),
    ],
)
def test_classify(error, kind):
    got, message = classify_error(error)
    assert got == kind
    assert message.startswith(ERROR_KINDS[kind]) and type(error).__name__ in message
    assert "secret body" not in message


def test_status_stays_in_the_message():
    assert classify_error(StatusError(429))[1] == "Rate limit reached (StatusError, HTTP 429)"
