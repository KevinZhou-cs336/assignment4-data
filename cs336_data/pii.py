
import re


EMAIL_MASK = "|||EMAIL_ADDRESS|||"
EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}"
)

def mask_emails(text: str) -> tuple[str, int]:
    masked, n = EMAIL_RE.subn(EMAIL_MASK, text)

    return masked, n


PHONE_NUMBER_MASK = "|||PHONE_NUMBER|||"
PHONE_RE = re.compile(
    r"""
    (?<!\d)                 # must not be preceded by a digit (avoid matching inside a longer number)
    (?:\+?1[\s.-]?)?        # optional country code: 1, +1, "+1 ", 1-, etc.
    (?:\(\d{3}\)|\d{3})     # area code: (283) or 283
    [\s.-]?                 # optional separator: space, dot, or hyphen
    \d{3}                   # exchange code (3 digits)
    [\s.-]?                 # optional separator
    \d{4}                   # line number (4 digits)
    (?!\d)                  # must not be followed by a digit
    """,
    re.VERBOSE,
)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    """Replace US-style phone numbers with |||PHONE_NUMBER|||.

    Returns:
        A tuple of (masked_text, number_of_replacements).
    """
    return PHONE_RE.subn(PHONE_NUMBER_MASK, text)



IP_ADDRESS_MASK = "|||IP_ADDRESS|||"
IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"  # 0-255, no leading zeros
IPV4_RE = re.compile(
    rf"""
    (?<![\d.])              # not part of a longer dotted number (e.g. 1.192.0.2.146)
    {IPV4_OCTET}(?:\.{IPV4_OCTET}){{3}}
    (?!\.?\d)               # not followed by more digits; a trailing period is fine
    """,
    re.VERBOSE,
)


def mask_ips(text: str) -> tuple[str, int]:
    """Replace IPv4 addresses with |||IP_ADDRESS|||.

    Returns:
        A tuple of (masked_text, number_of_replacements).
    """
    return IPV4_RE.subn(IP_ADDRESS_MASK, text)
