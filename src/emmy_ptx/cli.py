"""Command-line entry point for emmy-ptx."""
import argparse
import sys

from .compiler import Compiler, SUPPORTED_TARGETS, SUPPORTED_VERSIONS


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="emmy-ptx",
        description="Compile Emmy Kernel IR to NVIDIA PTX.",
    )
    parser.add_argument("ir_path", help="path to a Kernel IR JSON file")
    parser.add_argument(
        "--target", default="sm_89", choices=sorted(SUPPORTED_TARGETS),
        help="GPU target architecture (default: sm_89)",
    )
    parser.add_argument(
        "--version", default="8.0", dest="ptx_version",
        choices=sorted(SUPPORTED_VERSIONS),
        help="PTX ISA version (default: 8.0)",
    )
    parser.add_argument("-o", "--output", help="write PTX to this file instead of stdout")
    args = parser.parse_args()

    try:
        ptx = Compiler(args.ir_path, target=args.target, ptx_version=args.ptx_version).compile()
    except (SystemExit, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.output:
        with open(args.output, "w") as f:
            f.write(ptx)
    else:
        sys.stdout.write(ptx)
        if not ptx.endswith("\n"):
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
