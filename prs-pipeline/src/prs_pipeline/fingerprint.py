"""Remote fingerprint utilities for external data dependencies."""

from __future__ import annotations

import hashlib

import httpx


def _stable_header_value(headers: httpx.Headers, key: str) -> str:
    """Return a normalized header value used in fingerprint payloads."""
    value = headers.get(key, "")
    return value.strip()


def fingerprint_http_resource(url: str, include_body_hash: bool = False) -> tuple[str, dict[str, str]]:
    """Build a deterministic fingerprint for a remote HTTP resource.

    The fingerprint combines URL + selected response metadata and optionally
    the response body hash (for small manifest-like files).
    """
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        response = client.head(url)
        if response.status_code == 405:
            response = client.get(url, headers={"Range": "bytes=0-0"})
        response.raise_for_status()

        etag = _stable_header_value(response.headers, "etag")
        last_modified = _stable_header_value(response.headers, "last-modified")
        content_length = _stable_header_value(response.headers, "content-length")
        content_type = _stable_header_value(response.headers, "content-type")

        body_hash = ""
        if include_body_hash:
            body_response = client.get(url)
            body_response.raise_for_status()
            body_hash = hashlib.sha256(body_response.content).hexdigest()

    payload = "|".join([url, etag, last_modified, content_length, content_type, body_hash])
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    metadata: dict[str, str] = {
        "url": url,
        "etag": etag,
        "last_modified": last_modified,
        "content_length": content_length,
        "content_type": content_type,
    }
    if body_hash:
        metadata["body_sha256"] = body_hash
    metadata["fingerprint_sha256"] = fingerprint
    return fingerprint, metadata
