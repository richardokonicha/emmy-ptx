"""Compile a single Emmy Kernel IR file to PTX."""
import sys
from emmy_ptx import Compiler

if len(sys.argv) != 2:
    print("usage: python compile_one.py <kernel.ir.json>", file=sys.stderr)
    raise SystemExit(2)

ptx = Compiler(sys.argv[1]).compile()
print(ptx, end="")
