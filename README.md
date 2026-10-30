# emmy-ptx

Compile **Emmy Kernel IR** straight into **NVIDIA PTX** — no CUDA C++ step, no `nvcc`.

Emmy compiles PyTorch to CUDA C++ and hands it to `nvcc` to make PTX. That middle step is slow
(roughly 19x the cost of `ptxas` alone) and Emmy's autotuner wants to compile thousands of kernel
variants. `emmy-ptx` produces the same PTX directly, in about one second per kernel, by lowering
the typed Kernel IR to PTX text with one method per IR node.

## Install

```bash
pip install emmy-ptx
```

Requires Python 3.10+ and `emmy-ml` (used only to load the IR via `Graph.from_dict`).

## Usage

```bash
# Default: PTX 8.0, target sm_89
emmy-ptx path/to/kernel.ir.json > out.ptx

# Target a different GPU architecture
emmy-ptx path/to/kernel.ir.json --target sm_90 > out.ptx

# Select PTX ISA version
emmy-ptx path/to/kernel.ir.json --version 8.5 --target sm_90 > out.ptx

# Write to a file
emmy-ptx path/to/kernel.ir.json -o out.ptx
```

Or from Python:

```python
from emmy_ptx import Compiler, compile_file

# Default target
ptx = Compiler("path/to/kernel.ir.json").compile()

# Custom target
ptx = Compiler("path/to/kernel.ir.json", target="sm_90", ptx_version="8.5").compile()

# Convenience function
ptx = compile_file("path/to/kernel.ir.json", target="sm_90")
```

The output is PTX text that `ptxas` assembles directly. The entry-point name, parameter order, and
launch geometry come from the problem's `meta.json`, not from the IR body.

## What it handles

- **All five kernel categories:** pointwise, reductions, matmuls, fused (rmsnorm/softmax), attention
- **Dtypes:** f32, f64 (end to end, including reductions), integer, pred
- **Ops:** every elementwise op Emmy 0.3.11 can emit, including float `floor_divide`/`remainder`/`mod`,
  integer shifts, `bitcast`, `pad`
- **Memory:** vector v2/v4 global and shared-memory accesses (alignment-guarded), dynamic shared-memory
  pool for large matmuls, `cp.async` staging
- **Control flow:** strided loops, conditionals, `Select` (ternary)
- **Reductions:** warp shuffles, tree reductions, barriers, segmented (transposed) trees
- **Mixed precision:** operands are converted to the promoted dtype before the operation

## How it works

The compiler is a single Python module (~2,200 lines, stdlib only) with four layers:

1. **Reader** — loads the IR with `Graph.from_dict`, finds the `KernelOp`, recovers shapes
   and dtypes from the raw JSON and `meta.json` (the IR's own annotations are unreliable).
2. **Lowering** — walks the body statements in order, maintaining a map from IR value names
   to registers. SSA phi nodes become `alloca` + load/store; loops become structured PTX
   control flow; reductions become shared-memory staging + warp shuffles + `bar.sync`.
3. **Emitter** — renders PTX text with per-kernel register budgets and shared-memory layouts.
4. **Driver** — dispatch table keyed on IR op name; one method per node type.

The entry-point name, parameter order, and launch geometry all come from `meta.json`, not
from the IR body. This is the contract the challenge (and any real deployment) relies on.

## Architecture

```
compile.py (single module)
├── Compiler class
│   ├── __init__       — load IR, find kernel, find meta
│   ├── prologue()     — .version/.target, signature, param loads, cvta, thread id
│   ├── stmt()         — dispatch table (20+ entry types)
│   ├── expr()         — recursive index expression compiler
│   ├── coerce()       — dtype conversion (f32/f64/i32/i64)
│   └── op_on_regs()   — elementwise op lowering
├── PTXEmitter        — register pools, instruction emission
└── main()             — CLI entry point
```

## What it does NOT do

- f16: the runner's `ptxas` has no `.f16` type, so half-precision kernels cannot exist on this target
- `SetMaxNReg`: dropped with a warning — this `ptxas` does not know the instruction
- `KernelOp.serial`: rejected loudly (not implemented)

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## License

MIT
