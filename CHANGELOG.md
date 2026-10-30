# Changelog

All notable changes to emmy-ptx are documented here.

## [0.1.0] — 2026-10-30

### Added
- Initial release: from-scratch Emmy Kernel IR to PTX compiler
- All five kernel categories: pointwise, reductions, matmuls, fused, attention
- f32, f64, integer, and pred dtype support
- Vectorized loads/stores (v2/v4) with alignment guards
- Dynamic shared-memory pool for large matmuls
- `cp.async` staging for matmuls
- Mixed-precision coercion (f32/f64)
- Float `floor_divide`/`remainder`/`mod`, integer shifts, `bitcast`, `pad`
- SSA-to-imperative accumulation (phi nodes → alloca + load/store)
- Warp shuffles, tree reductions, barriers, segmented trees
- Online/streaming softmax for attention (no S×S materialization)
- CLI (`emmy-ptx`), Python API (`Compiler`, `compile_file`)
- 41 unit tests
- 41 generated variant packs with independent NumPy references
- Mnemonic probe: 79/80 instructions verified against ptxas
- MIT license

### Known limitations
- f16: not supported (runner's ptxas has no `.f16` type)
- `SetMaxNReg`: dropped with a warning (ptxas does not know the instruction)
- `KernelOp.serial`: rejected loudly (not implemented)
- Unsigned integer division uses signed `div.s32` (correct for values < 2^31)
