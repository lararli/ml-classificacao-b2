"""Terminal color helpers for pipeline output."""

GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
MAGENTA = "\033[1;35m"
DIM = "\033[0;37m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(text: str) -> str:
    return f"{GREEN}{text}{RESET}"


def fail(text: str) -> str:
    return f"{RED}{text}{RESET}"


def warn(text: str) -> str:
    return f"{YELLOW}{text}{RESET}"


def info(text: str) -> str:
    return f"{CYAN}{text}{RESET}"


def bold(text: str) -> str:
    return f"{BOLD}{text}{RESET}"


def dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


def header(text: str) -> str:
    return f"\n{CYAN}{text}{RESET}"
