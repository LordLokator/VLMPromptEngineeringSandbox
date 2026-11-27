"""Which Host header values the web server answers to."""
from ipaddress import AddressValueError, IPv6Address

ALLOWED_HOSTS_ENV = "VLM_ALLOWED_HOSTS"
# Starlette compares the host without its port and keeps the brackets of IPv6 literals.
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")


def _is_ipv6(text: str) -> bool:
    try:
        IPv6Address(text)
    except AddressValueError:
        return False
    return True


def parse_allowed_hosts(value: str) -> list[str]:
    """Parse the comma-separated variable. Exits with a message on an entry that could never match or allows all."""
    hosts: list[str] = []
    for raw in value.split(","):
        entry = raw.strip().lower()
        if not entry:
            continue
        bare = entry[1:-1] if entry.startswith("[") and entry.endswith("]") else entry
        if _is_ipv6(bare):
            # Browsers send the compressed form, and the middleware compares text.
            entry, valid = f"[{IPv6Address(bare).compressed}]", True
        else:
            name = entry.removeprefix("*.")
            valid = bool(name) and not set(name) & set("*:/[] \t@?#")
        if not valid:
            raise SystemExit(
                f"{ALLOWED_HOSTS_ENV}: invalid entry {raw.strip()!r}. Use host names or IP addresses without "
                "scheme, port or path, separated by commas; '*.example.com' is allowed, a bare '*' is not."
            )
        hosts.append(entry)
    return hosts


def require_extra_hosts(bind_host: str, extra_hosts: list[str]) -> None:
    """For a non-loopback bind: exit unless the names that clients will use are listed."""
    if not extra_hosts:
        raise SystemExit(
            f"VLM_HOST={bind_host} is not a loopback address: set {ALLOWED_HOSTS_ENV} to the host names or IP "
            "addresses that browsers will use to reach this machine, for example "
            f"{ALLOWED_HOSTS_ENV}=192.168.1.20,mybox.local"
        )
