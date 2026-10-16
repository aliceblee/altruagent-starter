"""Check that this machine can reach Claude with the key in .env.

    python scripts/check_claude.py

Sends one tiny request and prints either Claude's reply or the full chain of
errors behind a failure (certificate, proxy, DNS, firewall...). Never prints
the key.
"""

from __future__ import annotations

import os
import ssl
import sys

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("ANTHROPIC_API_KEY is not set (add it to .env).")
        return 1
    print(f"Key: set ({len(key)} characters, starts with {key[:7]}...)")
    for name in ("ANTHROPIC_BASE_URL", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        if os.environ.get(name) or os.environ.get(name.lower()):
            print(f"{name} is set: {os.environ.get(name) or os.environ.get(name.lower())}")
    print(f"Python {sys.version.split()[0]}, OpenSSL {ssl.OPENSSL_VERSION}")
    try:
        import anthropic
    except ImportError:
        print("The anthropic package isn't installed here: run `pip install -e .` in this venv.")
        return 1
    print(f"anthropic {anthropic.__version__}")
    model = os.environ.get("CLAUDE_MODEL") or "claude-haiku-5-5"
    try:
        reply = anthropic.Anthropic(max_retries=0, timeout=20).messages.create(
            model=model, max_tokens=50, messages=[{"role": "user", "content": "Reply with the word OK."}],
        )
    except Exception as exc:  # noqa: BLE001 - this script exists to show the error
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        inner = exc
        while (inner.__cause__ or inner.__context__) is not None:
            inner = inner.__cause__ or inner.__context__
            print(f"  caused by {type(inner).__name__}: {inner}")
        text = repr(inner)
        if "CERTIFICATE_VERIFY_FAILED" in text or "certificate" in text.lower():
            print("\nLooks like a certificate problem. Try: pip install --upgrade truststore certifi,\n"
                  "and on macOS run 'Install Certificates.command' from your Python folder in Applications.\n"
                  "On a school/work network that inspects HTTPS, try another network (e.g. a phone hotspot).")
        elif "getaddrinfo" in text or "Name or service" in text or "nodename" in text:
            print("\nDNS can't resolve api.anthropic.com: check the network / VPN.")
        elif "Connect" in type(inner).__name__ or "timed out" in text.lower():
            print("\nCan't open a connection to api.anthropic.com: a firewall, VPN or network may be blocking it.\n"
                  "Try another network (e.g. a phone hotspot) to confirm.")
        return 1
    text = next((b.text for b in reply.content if getattr(b, "type", None) == "text"), "")
    print(f"\nOK: {model} answered {text.strip()!r}. Claude is reachable from this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
