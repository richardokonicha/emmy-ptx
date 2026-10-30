# Roadmap

## v0.1.0 — Initial release (done)

- [x] Standalone package with `pyproject.toml`, CLI, and tests
- [x] All 15 open problems pass on the hosted runner
- [x] 41 generated variant packs pass
- [x] 41 unit tests
- [x] f64 end-to-end support
- [x] Mixed-precision coercion
- [x] Vectorized loads/stores (v2/v4)
- [x] `cp.async` staging
- [x] Mnemonic probe (79/80 instructions verified)

## v0.2.0 — Usability and portability

- [x] `--target` flag to switch GPU architectures (sm_70–sm_100)
- [ ] `--version` flag for PTX ISA version selection
- [x] Caching layer — hash the IR + meta.json, skip recompilation on cache hit
- [x] `-o` / `--output` flag for file output
- [ ] Better error messages with source locations
- [ ] `emmy-ptx --check` — validate IR without compiling
- [ ] Python 3.10 compatibility testing

## v0.3.0 — Performance and optimization

- [ ] Instruction-level optimization (constant folding, dead code elimination)
- [ ] Register pressure analysis and hints
- [ ] Shared memory layout optimization
- [ ] `--bench` mode for timing comparison against nvcc/Triton
- [ ] Diff tool: compare generated PTX against a reference

## v0.4.0 — Real-world integration

- [ ] PyTorch model → PTX pipeline (end-to-end, no nvcc)
- [ ] Integration with Emmy's autotuner
- [ ] Kernel analysis/inspection mode (register counts, memory access patterns)
- [ ] Cross-compilation support (compile on x86 for ARM GPU targets)
- [ ] Web playground (paste IR, see PTX)

## v1.0.0 — Production ready

- [ ] Stable public API
- [ ] Full documentation site
- [ ] Performance benchmarks published
- [ ] Integration tests against multiple GPU architectures
- [ ] Published on PyPI

## Ideas / backlog

- [ ] Support for additional IR frontends (JAX, TensorFlow, ONNX)
- [ ] JIT compilation mode (compile at runtime)
- [ ] Kernel fusion pass (merge multiple kernels)
- [ ] Profiling integration (NVIDIA Nsight)
- [ ] VS Code extension for PTX preview
