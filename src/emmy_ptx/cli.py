"""Command-line entry point for emmy-ptx."""
import sys

from .compiler import Compiler


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: emmy-ptx <kernel.ir.json>", file=sys.stderr)
        return 2
    try:
        ptx = Compiler(sys.argv[1]).compile()
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    if not ptx.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
