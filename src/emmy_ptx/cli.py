"""Command-line entry point for emmy-ptx."""
import argparse
import os
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
    parser.add_argument(
        "--check", action="store_true",
        help="validate the IR without compiling; print issues and exit",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="cache compiled PTX by IR hash (skip recompilation on cache hit)",
    )
    args = parser.parse_args()

    if args.check:
        try:
            compiler = Compiler(args.ir_path, target=args.target, ptx_version=args.ptx_version)
            issues = compiler.check()
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if issues:
            print(f"{len(issues)} issue(s) found:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print("OK — no issues found")
        return 0

    try:
        from .compiler import compile_file
        ptx = compile_file(
            args.ir_path, target=args.target, ptx_version=args.ptx_version,
            cache_dir=args.cache_dir,
        )
    except (SystemExit, ValueError, Exception) as exc:
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
