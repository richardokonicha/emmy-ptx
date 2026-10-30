# Changelog

All notable changes to emmy-ptx are documented here.

## [Unreleased]

### Added
- Better error messages: compilation failures now include the offending IR statement
- Python 3.10 compatibility verified (no 3.11+ features used)
- `--check` mode: validate IR without compiling
- `--cache-dir` flag: cache compiled PTX by IR hash (skip recompilation on cache hit)
- `--target` CLI flag to switch GPU architectures (sm_70–sm_100)
- `--version` CLI flag to select PTX ISA version (7.0–9.0)
- `-o` / `--output` flag for file output
- `Compiler(target=..., ptx_version=...)` constructor parameters
- `ptx_header()`, `SUPPORTED_TARGETS`, `SUPPORTED_VERSIONS` exported from package

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
