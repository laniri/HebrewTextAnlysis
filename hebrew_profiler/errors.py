"""Custom error types for the Hebrew Linguistic Profiling Engine."""


class StanzaError(Exception):
    """Raised when the Stanza pipeline fails during processing."""


class StanzaSetupError(Exception):
    """Raised when Stanza is not installed or the Hebrew model is not downloaded."""


class YAPConnectionError(Exception):
    """Raised when the YAP API is unreachable."""


class YAPHTTPError(Exception):
    """Raised when YAP returns a non-2xx HTTP status."""

    def __init__(self, http_status: int, message: str) -> None:
        self.http_status = http_status
        self.message = message
        super().__init__(f"YAP HTTP {http_status}: {message}")


class EncodingError(Exception):
    """Raised when an input file contains invalid UTF-8 encoding."""


class MalformedParserOutput(Exception):
    """Raised when Stanza or YAP returns unparseable data."""
