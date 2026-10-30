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
emmy-ptx path/to/kernel.ir.json > out.ptx
```

Or from Python:

```python
from emmy_ptx import Compiler
ptx = Compiler("path/to/kernel.ir.json").compile()
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
