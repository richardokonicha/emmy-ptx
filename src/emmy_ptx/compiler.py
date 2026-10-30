#!/usr/bin/env python3
"""Emmy Kernel IR -> PTX compiler.

Usage: python compile.py path/to/<kernel>.ir.json > out.ptx

The IR is loaded through the public ``Graph.from_dict`` API.  Shapes and
dtypes are read from the raw JSON because the typed graph does not expose
them.  Every IR node is lowered directly to PTX; no existing CUDA/PTX
renderer is used.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    FlatIndex,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from emmy.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    KernelOp,
    Smem,
    SetMaxNReg,
    Sync,
    TreeHalve,
    WarpShuffle,
)
from emmy.compiler.ir.stmt import (
    Accum,
    Assign,
    Body,
    Cond,
    Init,
    Let,
    Load,
    Loop,
    Select,
    StridedLoop,
    Write,
)

SUPPORTED_TARGETS = {
    "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100",
}
SUPPORTED_VERSIONS = {"7.0", "7.5", "8.0", "8.5", "9.0"}


def ptx_header(version: str = "8.0", target: str = "sm_89") -> str:
    if target not in SUPPORTED_TARGETS:
        raise ValueError(f"unsupported target {target!r}; choose from {sorted(SUPPORTED_TARGETS)}")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported PTX version {version!r}; choose from {sorted(SUPPORTED_VERSIONS)}")
    return f".version {version}\n.target {target}\n.address_size 64\n"
SMEM_STATIC_LIMIT = 48 * 1024
# Upper bound on fully-unrolled trip count.  Unrolling a large loop would
# explode the PTX (extent 100k -> 30+ MB) and stall ptxas, so wide loops
# stay rolled.
MAX_UNROLL = 64

_DTYPE_ALIASES = {
    "float": "f32",
    "double": "f64",
    "half": "f16",
    "int": "s32",
    "bool": "s32",
}


def dtype_name(dt) -> str:
    name = getattr(dt, "name", str(dt))
    return _DTYPE_ALIASES.get(name, name)


def dtype_bytes(dt) -> int:
    name = dtype_name(dt)
    if name in ("f32", "s32", "u32"):
        return 4
    if name in ("f16", "u16"):
        return 2
    if name in ("f64", "s64", "u64"):
        return 8
    raise SystemExit(f"unsupported dtype size: {name}")


def is_float(dt) -> bool:
    return dtype_name(dt).startswith("f")


# Ops that always compute in f32: their result is f32 whatever the operand
# was, so the operand is converted before the instruction is emitted.
F32_ONLY_OPS = frozenset({
    "exp", "exp_fast", "sign", "log", "log2", "log10", "softplus",
    "sin", "cos", "tan", "pow", "relu", "sigmoid", "silu", "tanh",
    "erf", "gelu", "gelu_tanh", "reciprocal", "sqrt", "rsqrt", "fma",
})


def ptx_type(dt) -> str:
    name = dtype_name(dt)
    if name == "f32":
        return "f32"
    if name == "f16":
        # ptxas rejects ld/global.f16; f16 values live in .b16 registers and
        # are loaded/stored as 16-bit quantities.
        return "b16"
    if name == "f64":
        return "f64"
    if name in ("s32", "i32", "int"):
        return "s32"
    if name in ("s64", "i64"):
        return "s64"
    if name in ("b32", "u32"):
        return "b32"
    raise SystemExit(f"unsupported PTX type: {name}")


def reg_kind(dt) -> str:
    """Register-kind letter for a dtype (used by Regs.new/declarations)."""
    name = dtype_name(dt)
    if name == "f32":
        return "f"
    if name == "f16":
        return "h"
    if name == "f64":
        return "d"
    if name in ("s64", "i64", "u64"):
        return "rd"
    if name in ("s32", "i32", "int", "b32", "u32"):
        return "r"
    raise SystemExit(f"unsupported register kind: {name}")


def ldst_type(dt) -> str:
    """PTX load/store type for a dtype (f16 uses b16; f64 uses f64)."""
    return ptx_type(dt)


def float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def float16_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", float(value)))[0]


def float64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


class Regs:
    """Virtual register pool.  ptxas does the physical allocation.

    Kind letters map to PTX register types: ``f`` f32, ``r`` b32/s32,
    ``rd`` b64/u64, ``h`` b16 (f16 storage), ``d`` f64, ``p`` pred.
    """

    def __init__(self):
        self.n = {"f": 0, "r": 0, "rd": 0, "h": 0, "d": 0, "p": 0}

    def new(self, kind: str) -> str:
        self.n[kind] += 1
        return f"%{kind}{self.n[kind]}"

    def declarations(self):
        return [
            f"        .reg .pred %p<{self.n['p'] + 1}>;",
            f"        .reg .f32  %f<{self.n['f'] + 1}>;",
            f"        .reg .b16  %h<{self.n['h'] + 1}>;",
            f"        .reg .f64  %d<{self.n['d'] + 1}>;",
            f"        .reg .b32  %r<{self.n['r'] + 1}>;",
            f"        .reg .b64  %rd<{self.n['rd'] + 1}>;",
        ]


class Emitter:
    def __init__(self):
        self.lines = []
        self.regs = Regs()
        self.label_n = 0

    def ins(self, text: str):
        self.lines.append("        " + text)

    def raw(self, text: str):
        self.lines.append(text)

    def label(self, prefix: str) -> str:
        self.label_n += 1
        return f"{prefix}_{self.label_n}"


class Compiler:
    def __init__(self, ir_path: Path, *, target: str = "sm_89", ptx_version: str = "8.0"):
        self.ir_path = Path(ir_path)
        self.target = target
        self.ptx_version = ptx_version
        self.raw = json.loads(self.ir_path.read_text())
        self.graph = Graph.from_dict(self.raw)
        self.kernel = self._find_kernel()
        if getattr(self.kernel, "serial", None):
            raise SystemExit(
                "KernelOp.serial is not empty: serial axes are launched one "
                "coordinate at a time with runtime integer parameters, which "
                "this compiler does not implement"
            )
        self.meta = self._find_meta()
        self.kmeta = self._kernel_meta()
        self.entry = self.kernel.name or self.kmeta.get("entry", self.kernel.name)

        self.tensors: dict[str, dict] = {}
        for node in self.raw.get("nodes", {}).values():
            for out in node.get("outputs", []):
                self.tensors[out["name"]] = out
        for arg in self.kmeta.get("args", []):
            self.tensors.setdefault(
                arg["name"],
                {"name": arg["name"], "shape": arg.get("shape", []), "dtype": arg.get("dtype", "f32")},
            )

        self.arg_order = list(self.kmeta.get("arg_order") or (list(self.kernel.inputs) + list(self.kernel.outputs)))
        self.args = {a["name"]: a for a in self.kmeta.get("args", [])}

        self.emitter = Emitter()
        self.values: dict[str, str] = {}
        self.value_dtypes: dict[str, str] = {}
        self.base: dict[str, str] = {}
        self.special: dict[str, str] = {}
        self.smem: dict[str, dict] = {}
        self.smem_bases: dict[str, str] = {}
        self.smem_declared: set[str] = set()
        self.shared_decls: list[str] = []
        self.dynamic_smem = False

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def _find_kernel(self) -> KernelOp:
        for node in self.graph.nodes.values():
            if isinstance(node.op, KernelOp):
                return node.op
        raise SystemExit("no KernelOp found in IR")

    def _meta_mentions_kernel(self, meta: dict) -> bool:
        """True if ``meta`` plausibly describes this kernel.

        An ancestor's meta.json is only adopted when it names this kernel (or
        holds exactly one kernel), so a neighbouring problem's metadata can
        never supply the wrong argument order.
        """
        kernels = meta.get("kernels")
        if not isinstance(kernels, list) or not kernels:
            return False
        if len(kernels) == 1:
            return True
        return any(k.get("entry") == self.kernel.name for k in kernels)

    def _find_meta(self) -> dict:
        meta_path = self.ir_path.with_name("meta.json")
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        # Walk up the directory tree so kernels shipped without a sibling
        # meta.json are still discoverable.
        candidate = self.ir_path.parent
        for ancestor in [candidate, *candidate.parents]:
            meta_path = ancestor / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if self._meta_mentions_kernel(meta):
                    return meta
        # No contract file at all.  meta.json is the contract whenever it
        # exists, but a missing one must not cost the kernel: fall back to the
        # IR's own argument order and say so on stderr.
        print(f"warning: no meta.json found for {self.ir_path}; using the IR's own "
              "argument order (inputs followed by outputs)", file=sys.stderr)
        return {}

    def _kernel_meta(self) -> dict:
        if not self.meta:
            return {}
        kernels = self.meta.get("kernels")
        if not isinstance(kernels, list) or not kernels:
            raise SystemExit("meta.json has no kernels")
        for k in kernels:
            if k.get("entry") == self.kernel.name:
                return k
        if len(kernels) == 1:
            return kernels[0]
        raise SystemExit(
            f"kernel {self.kernel.name!r} is not listed in meta.json "
            f"({', '.join(str(k.get('entry')) for k in kernels)}); its argument "
            "order would be wrong"
        )

    def shape_of(self, name: str):
        if name in self.smem:
            return self.smem[name]["extents"]
        if name not in self.tensors:
            raise SystemExit(f"unknown buffer {name!r}")
        return self.tensors[name].get("shape", [])

    def dtype_of(self, name: str):
        if name in self.smem:
            return self.smem[name]["dtype"]
        if name not in self.tensors:
            raise SystemExit(f"unknown buffer {name!r}")
        return self.tensors[name].get("dtype", "f32")

    # ------------------------------------------------------------------
    # shared memory planning
    # ------------------------------------------------------------------
    @staticmethod
    def _children(stmt):
        if isinstance(stmt, (Body, list, tuple)):
            for child in stmt:
                yield child
            return
        for attr in ("body", "else_body"):
            child = getattr(stmt, attr, None)
            if child is not None:
                yield from child

    def _walk(self, stmt):
        for child in self._children(stmt):
            yield child
            yield from self._walk(child)

    def plan_smem(self):
        smem_stmts = [s for s in self._walk(self.kernel.body) if isinstance(s, Smem)]
        if not smem_stmts:
            return
        total = 0
        for s in smem_stmts:
            n = 1
            for e in s.extents:
                n *= int(e)
            size = n * dtype_bytes(s.dtype)
            align = int(s.align or 4)
            self.smem[s.name] = {
                "extents": [int(e) for e in s.extents],
                "dtype": dtype_name(s.dtype),
                "size": size,
                "align": align,
                "offset": 0,
                "static": None,
            }
            total += size
        self.dynamic_smem = total > SMEM_STATIC_LIMIT
        if not self.dynamic_smem:
            for name, info in self.smem.items():
                info["static"] = name
                self.shared_decls.append(
                    f".shared .align {info['align']} .b8 {name}[{info['size']}];"
                )
        else:
            off = 0
            for name, info in self.smem.items():
                align = info["align"]
                off = (off + align - 1) // align * align
                info["offset"] = off
                off += info["size"]
            self.shared_decls.append(".extern .shared .align 16 .b8 __emmy_smem[];")

    def _materialize_smem_base(self, name: str) -> str:
        if name in self.smem_bases:
            return self.smem_bases[name]
        info = self.smem[name]
        base = self.emitter.regs.new("r")
        if info["static"] is not None:
            self.emitter.ins(f"mov.u32 {base}, {info['static']};")
        else:
            pool = self.smem_bases.get("__pool__")
            if pool is None:
                pool = self.emitter.regs.new("r")
                self.emitter.ins(f"mov.u32 {pool}, __emmy_smem;")
                self.smem_bases["__pool__"] = pool
            if info["offset"]:
                self.emitter.ins(f"add.s32 {base}, {pool}, {info['offset']};")
            else:
                base = pool
        self.smem_bases[name] = base
        return base

    def smem_base(self, name: str) -> str:
        return self._materialize_smem_base(name)

    def smem_elem_addr(self, name: str, flat: str) -> str:
        info = self.smem[name]
        off = self.emitter.regs.new("r")
        self.emitter.ins(f"mul.lo.s32 {off}, {flat}, {dtype_bytes(info['dtype'])};")
        addr = self.emitter.regs.new("r")
        self.emitter.ins(f"add.s32 {addr}, {self.smem_base(name)}, {off};")
        return addr

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def int_reg(self, value: int) -> str:
        reg = self.emitter.regs.new("r")
        self.emitter.ins(f"mov.s32 {reg}, {int(value)};")
        return reg

    def f64_const(self, value: float) -> str:
        """Materialize an f64 constant.

        This ptxas rejects a 64-bit immediate in `mov.f64`, so the bit
        pattern is built from two 32-bit halves and moved in as a .b64 pair.
        """
        bits = float64_bits(value)
        lo = self.emitter.regs.new("r")
        hi = self.emitter.regs.new("r")
        self.emitter.ins(f"mov.u32 {lo}, 0x{bits & 0xFFFFFFFF:08X};")
        self.emitter.ins(f"mov.u32 {hi}, 0x{(bits >> 32) & 0xFFFFFFFF:08X};")
        pair = self.emitter.regs.new("rd")
        self.emitter.ins(f"mov.b64 {pair}, {{{lo}, {hi}}};")
        reg = self.emitter.regs.new("d")
        self.emitter.ins(f"mov.f64 {reg}, {pair};")
        return reg

    def fconst(self, value: float, dtype="f32") -> str:
        name = dtype_name(dtype)
        if name == "f32":
            reg = self.emitter.regs.new("f")
            self.emitter.ins(f"mov.f32 {reg}, 0f{float_bits(value):08X};")
        elif name == "f64":
            reg = self.f64_const(value)
        elif name == "f16":
            reg = self.emitter.regs.new("h")
            self.emitter.ins(f"mov.b16 {reg}, 0x{float16_bits(value):04X};")
        else:
            reg = self.int_reg(int(value))
        return reg

    def mul_int(self, a: str, b) -> str:
        reg = self.emitter.regs.new("r")
        self.emitter.ins(f"mul.lo.s32 {reg}, {a}, {b};")
        return reg

    def add_int(self, a: str, b) -> str:
        reg = self.emitter.regs.new("r")
        self.emitter.ins(f"add.s32 {reg}, {a}, {b};")
        return reg

    def _sdiv_const(self, a: str, c: int) -> str:
        """a / c with C semantics, i.e. truncation toward zero."""
        if c == 1:
            return a
        if c == 0:
            reg = self.emitter.regs.new("r")
            self.emitter.ins(f"div.s32 {reg}, {a}, {c};")
            return reg
        negate = c < 0
        mag = -c if negate else c
        if mag & (mag - 1) == 0:
            # An arithmetic shift rounds toward -inf, so a negative dividend
            # is biased up by (2**k - 1) first to get truncation.
            sign = self.emitter.regs.new("r")
            self.emitter.ins(f"shr.s32 {sign}, {a}, 31;")
            mask = self.emitter.regs.new("r")
            self.emitter.ins(f"and.b32 {mask}, {sign}, {mag - 1};")
            adj = self.emitter.regs.new("r")
            self.emitter.ins(f"add.s32 {adj}, {a}, {mask};")
            q = self.emitter.regs.new("r")
            self.emitter.ins(f"shr.s32 {q}, {adj}, {mag.bit_length() - 1};")
            if negate:
                nq = self.emitter.regs.new("r")
                self.emitter.ins(f"neg.s32 {nq}, {q};")
                return nq
            return q
        reg = self.emitter.regs.new("r")
        self.emitter.ins(f"div.s32 {reg}, {a}, {c};")
        return reg

    def _srem_const(self, a: str, c: int) -> str:
        """a % c with C semantics: the sign follows the dividend."""
        if c == 0:
            reg = self.emitter.regs.new("r")
            self.emitter.ins(f"rem.s32 {reg}, {a}, {c};")
            return reg
        q = self._sdiv_const(a, c)
        prod = self.emitter.regs.new("r")
        self.emitter.ins(f"mul.lo.s32 {prod}, {q}, {c};")
        reg = self.emitter.regs.new("r")
        self.emitter.ins(f"sub.s32 {reg}, {a}, {prod};")
        return reg

    @staticmethod
    def const_int(e):
        if isinstance(e, Literal):
            name = dtype_name(e.dtype)
            if name.startswith("s") or name.startswith("i") or name == "int":
                return int(e.value)
            if isinstance(e.value, int):
                return int(e.value)
            return None
        if isinstance(e, BinaryExpr):
            a = Compiler.const_int(e.left)
            b = Compiler.const_int(e.right)
            if a is None or b is None:
                return None
            if e.op == "+":
                return a + b
            if e.op == "-":
                return a - b
            if e.op == "*":
                return a * b
            if e.op == "/" and b:
                return a // b
            if e.op == "%" and b:
                return a % b
        return None

    def add_const_expr(self, e, k: int):
        c = self.const_int(e)
        if c is not None:
            return Literal(c + k, "int")
        return BinaryExpr("+", e, Literal(k, "int"))

    def flatten(self, name: str, indices) -> str:
        shape = self.shape_of(name)
        indices = tuple(indices or ())
        if not indices:
            return self.int_reg(0)
        if len(indices) != len(shape):
            raise SystemExit(
                f"index rank {len(indices)} != shape rank {len(shape)} for {name}"
            )
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        acc_reg = None
        acc_const = 0
        for idx, stride in zip(indices, strides):
            c = self.const_int(idx)
            if c is not None:
                acc_const += c * stride
                continue
            reg, _ = self.expr(idx)
            if stride != 1:
                reg = self.mul_int(reg, stride)
            acc_reg = reg if acc_reg is None else self.add_int(acc_reg, reg)
        if acc_reg is None:
            return self.int_reg(acc_const)
        if acc_const:
            acc_reg = self.add_int(acc_reg, acc_const)
        return acc_reg

    def _is_multiple_of(self, e, w: int) -> bool:
        """True if expression ``e`` is provably divisible by ``w``."""
        if isinstance(e, Literal):
            try:
                return int(e.value) % w == 0
            except (TypeError, ValueError):
                return False
        if isinstance(e, BinaryExpr):
            if e.op in ("+", "-"):
                return self._is_multiple_of(e.left, w) and self._is_multiple_of(e.right, w)
            if e.op == "*":
                return self._is_multiple_of(e.left, w) or self._is_multiple_of(e.right, w)
        return False

    def _flat_index_aligned(self, name: str, index, w: int) -> bool:
        """True if the flat element index of ``index`` is provably 0 (mod ``w``)."""
        shape = self.shape_of(name)
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * int(shape[i + 1])
        for idx, stride in zip(index, strides):
            if int(stride) % w != 0 and not self._is_multiple_of(idx, w):
                return False
        return True

    def _size_divisible(self, name: str, w: int) -> bool:
        n = 1
        for e in self.shape_of(name):
            n *= int(e)
        return n % w == 0

    def global_addr(self, name: str, flat: str) -> str:
        if name not in self.base:
            raise SystemExit(f"no pointer parameter for buffer {name!r}")
        off = self.emitter.regs.new("rd")
        self.emitter.ins(f"mul.wide.s32 {off}, {flat}, {dtype_bytes(self.dtype_of(name))};")
        addr = self.emitter.regs.new("rd")
        self.emitter.ins(f"add.s64 {addr}, {self.base[name]}, {off};")
        return addr

    def to_pred(self, e) -> str:
        reg, dt = self.expr(e)
        if dt == "pred":
            return reg
        p = self.emitter.regs.new("p")
        if dt == "f64":
            zero = self.emitter.regs.new("d")
            self.emitter.ins(f"mov.f64 {zero}, 0d0000000000000000;")
            self.emitter.ins(f"setp.ne.f64 {p}, {reg}, {zero};")
        elif dt == "f32":
            self.emitter.ins(f"setp.ne.f32 {p}, {reg}, 0f00000000;")
        elif dt == "f16":
            self.emitter.ins(f"setp.ne.b16 {p}, {reg}, 0;")
        else:
            self.emitter.ins(f"setp.ne.s32 {p}, {reg}, 0;")
        return p

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------
    def lit(self, value, dtype="f32"):
        name = dtype_name(dtype)
        if name.startswith("s") or name.startswith("i") or name == "int":
            reg = self.emitter.regs.new("r")
            self.emitter.ins(f"mov.s32 {reg}, {int(value)};")
            return reg, "s32"
        if name == "f64":
            return self.f64_const(float(value)), "f64"
        if name == "f16":
            reg = self.emitter.regs.new("h")
            self.emitter.ins(f"mov.b16 {reg}, 0x{float16_bits(float(value)):04X};")
            return reg, "f16"
        reg = self.emitter.regs.new("f")
        self.emitter.ins(f"mov.f32 {reg}, 0f{float_bits(value):08X};")
        return reg, "f32"

    def expr(self, e):
        if isinstance(e, str):
            e = Var(e)
        if isinstance(e, Literal):
            name = dtype_name(e.dtype)
            if name.startswith("s") or name.startswith("i") or name == "int" or isinstance(e.value, int):
                return self.lit(e.value, "s32")
            return self.lit(e.value, name)
        if isinstance(e, Var):
            if e.name not in self.values:
                raise SystemExit(f"unknown variable {e.name!r}")
            return self.values[e.name], self.value_dtypes[e.name]
        if isinstance(e, BinaryExpr):
            return self.binary(e)
        if isinstance(e, TernaryExpr):
            return self.ternary(e)
        if isinstance(e, CastExpr):
            return self.cast(e)
        if isinstance(e, Builtin):
            return self.builtin(e)
        if isinstance(e, FuncCallExpr):
            regs = [self.expr(a)[0] for a in e.args]
            return self.op_on_regs(e.name, regs, "f32")
        if isinstance(e, FlatIndex):
            return self.flatten(e.buffer, e.index), "s32"
        raise SystemExit(f"unsupported expression {type(e).__name__}: {e!r}")

    def coerce(self, reg, dt, target):
        """Convert ``reg`` from ``dt`` to ``target``; returns the new register.

        The register is returned unchanged when no conversion is needed, so
        callers can always use the result as a register of type ``target``.
        Every conversion below was checked against the runner's ``ptxas``:
        integer-to-integer uses ``cvt`` *without* a rounding modifier, which
        this assembler rejects with "Illegal rounding modifier".
        """
        dt = dtype_name(dt)
        target = dtype_name(target)
        if dt == "pred":
            t = self.emitter.regs.new("r")
            self.emitter.ins(f"selp.b32 {t}, 1, 0, {reg};")
            reg, dt = t, "s32"
        if dt == target:
            return reg
        if dt == "f16" or target == "f16":
            # This ptxas rejects ld/global.f16 and has no .b16<->.f32/.f64
            # conversion, so f16 cannot participate in arithmetic coercion.
            raise SystemExit(f"f16 coercion {dt} -> {target} is unsupported")
        wide = dt in ("s64", "u64", "i64")
        if wide and target in ("s32", "u32", "i32", "int"):
            out = self.emitter.regs.new("r")
            self.emitter.ins(f"cvt.s32.{dt} {out}, {reg};")
            return out
        if target in ("s64", "u64", "i64") and dt in ("s32", "u32", "i32", "int"):
            out = self.emitter.regs.new("rd")
            self.emitter.ins(f"cvt.{target}.s32 {out}, {reg};")
            return out
        if is_float(target) and not is_float(dt):
            out = self.emitter.regs.new(reg_kind(target))
            op = {
                "f32": "cvt.rn.f32.s32" if not wide else "cvt.rn.f32.s64",
                "f64": "cvt.rn.f64.s32" if not wide else "cvt.rn.f64.s64",
            }[target]
            self.emitter.ins(f"{op} {out}, {reg};")
            return out
        if not is_float(target) and is_float(dt):
            out = self.emitter.regs.new(reg_kind(target))
            op = {"f32": "cvt.rzi.s32.f32", "f64": "cvt.rzi.s32.f64"}[dt]
            self.emitter.ins(f"{op} {out}, {reg};")
            return out
        # float <-> float (f32 <-> f64).  The table is keyed by the *source*
        # type.  Widening f32 -> f64 is exact, so ptxas wants no rounding
        # modifier there ("cvt.rn.f64.f32" is an "Illegal rounding modifier");
        # narrowing f64 -> f32 needs one ("cvt.f32.f64" is a "Rounding
        # modifier required").
        if is_float(dt) and is_float(target):
            out = self.emitter.regs.new(reg_kind(target))
            op = {"f32": "cvt.f64.f32", "f64": "cvt.rn.f32.f64"}[dt]
            self.emitter.ins(f"{op} {out}, {reg};")
            return out
        return reg

    def binary(self, e: BinaryExpr):
        op = e.op
        if op in ("/", "%") and isinstance(e.right, Literal) and isinstance(e.right.value, int):
            c = int(e.right.value)
            if c != 0:
                a, adt = self.expr(e.left)
                if not is_float(adt):
                    reg = self._sdiv_const(a, c) if op == "/" else self._srem_const(a, c)
                    return reg, "s32"
        a, adt = self.expr(e.left)
        b, bdt = self.expr(e.right)

        if op in ("==", "!=", "<", "<=", ">", ">="):
            target = "f64" if (adt == "f64" or bdt == "f64") else ("f32" if (is_float(adt) or is_float(bdt)) else "s32")
            a = self.coerce(a, adt, target)
            b = self.coerce(b, bdt, target)
            p = self.emitter.regs.new("p")
            kind = {"f32": "f32", "f64": "f64"}.get(target, "s32")
            cmp = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[op]
            self.emitter.ins(f"setp.{cmp}.{kind} {p}, {a}, {b};")
            return p, "pred"

        if op in ("and", "or"):
            if adt == "pred" and bdt == "pred":
                p = self.emitter.regs.new("p")
                self.emitter.ins(f"{op}.pred {p}, {a}, {b};")
                return p, "pred"
            out = self.emitter.regs.new("r")
            self.emitter.ins(f"{op}.b32 {out}, {a}, {b};")
            return out, "s32"

        target = "f64" if (adt == "f64" or bdt == "f64") else ("f32" if (is_float(adt) or is_float(bdt)) else "s32")
        a = self.coerce(a, adt, target)
        b = self.coerce(b, bdt, target)

        if target == "f32":
            ptxop = {
                "+": "add.f32",
                "-": "sub.f32",
                "*": "mul.f32",
                "/": "div.rn.f32",
                "maximum": "max.f32",
                "minimum": "min.f32",
                "max": "max.f32",
                "min": "min.f32",
            }.get(op)
            if ptxop is None:
                raise SystemExit(f"unsupported float binary op {op!r}")
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"{ptxop} {out}, {a}, {b};")
            return out, "f32"

        if target == "f64":
            ptxop = {
                "+": "add.f64",
                "-": "sub.f64",
                "*": "mul.f64",
                "/": "div.rn.f64",
                "maximum": "max.f64",
                "minimum": "min.f64",
                "max": "max.f64",
                "min": "min.f64",
            }.get(op)
            if ptxop is None:
                raise SystemExit(f"unsupported float binary op {op!r}")
            out = self.emitter.regs.new("d")
            self.emitter.ins(f"{ptxop} {out}, {a}, {b};")
            return out, "f64"

        ptxop = {
            "+": "add.s32",
            "-": "sub.s32",
            "*": "mul.lo.s32",
            "/": "div.s32",
            "%": "rem.s32",
            "and": "and.b32",
            "or": "or.b32",
            "xor": "xor.b32",
            "maximum": "max.s32",
            "minimum": "min.s32",
            "max": "max.s32",
            "min": "min.s32",
            "<<": "shl.b32",
            ">>": "shr.s32",
        }.get(op)
        if ptxop is None:
            raise SystemExit(f"unsupported integer binary op {op!r}")
        out = self.emitter.regs.new("r")
        self.emitter.ins(f"{ptxop} {out}, {a}, {b};")
        return out, "s32"

    def ternary(self, e: TernaryExpr):
        p = self.to_pred(e.cond)
        a, adt = self.expr(e.if_true)
        b, bdt = self.expr(e.if_false)
        target = "f64" if (adt == "f64" or bdt == "f64") else ("f32" if (is_float(adt) or is_float(bdt)) else "s32")
        a = self.coerce(a, adt, target)
        b = self.coerce(b, bdt, target)
        out = self.emitter.regs.new(reg_kind(target))
        self.emitter.ins(f"selp.{ptx_type(target)} {out}, {a}, {b}, {p};")
        return out, target

    def cast(self, e: CastExpr):
        reg, src = self.expr(e.expr)
        dst = dtype_name(e.dtype)
        if dst == src:
            return reg, src
        if dst == "f32" and not is_float(src):
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"cvt.rn.f32.s32 {out}, {reg};")
            return out, "f32"
        if dst == "s32" and is_float(src):
            out = self.emitter.regs.new("r")
            self.emitter.ins(f"cvt.rzi.s32.f32 {out}, {reg};")
            return out, "s32"
        raise SystemExit(f"unsupported cast {src} -> {dst}")

    def builtin(self, e: Builtin):
        if e.name in self.special:
            return self.special[e.name], "s32"
        special = {
            "threadIdx.x": "%tid.x",
            "threadIdx.y": "%tid.y",
            "threadIdx.z": "%tid.z",
            "blockIdx.x": "%ctaid.x",
            "blockIdx.y": "%ctaid.y",
            "blockIdx.z": "%ctaid.z",
            "blockDim.x": "%ntid.x",
            "blockDim.y": "%ntid.y",
            "blockDim.z": "%ntid.z",
            "gridDim.x": "%nctaid.x",
        }.get(e.name)
        if special is None:
            raise SystemExit(f"unsupported builtin {e.name!r}")
        reg = self.emitter.regs.new("r")
        self.emitter.ins(f"mov.u32 {reg}, {special};")
        self.special[e.name] = reg
        return reg, "s32"

    # ------------------------------------------------------------------
    # elementwise ops
    # ------------------------------------------------------------------
    def _emit_exp_f32(self, y):
        # exp(y) = 2^(y * log2 e)
        t = self.emitter.regs.new("f")
        self.emitter.ins(f"mul.f32 {t}, {y}, 0f{float_bits(1.4426950408889634):08X};")
        out = self.emitter.regs.new("f")
        self.emitter.ins(f"ex2.approx.f32 {out}, {t};")
        return out

    def _emit_lg2_f32(self, x):
        out = self.emitter.regs.new("f")
        self.emitter.ins(f"lg2.approx.f32 {out}, {x};")
        return out

    def emit_tanh_f32(self, x):
        # tanh(x) = sign(x) * (1 - e) / (1 + e),  e = 2^(-2*log2(e)*|x|) = exp(-2|x|)
        ax = self.emitter.regs.new("f")
        self.emitter.ins(f"abs.f32 {ax}, {x};")
        y = self.emitter.regs.new("f")
        self.emitter.ins(f"mul.f32 {y}, {ax}, 0f{float_bits(-2.8853900817779268):08X};")
        ex = self.emitter.regs.new("f")
        self.emitter.ins(f"ex2.approx.f32 {ex}, {y};")
        negex = self.emitter.regs.new("f")
        self.emitter.ins(f"neg.f32 {negex}, {ex};")
        num = self.emitter.regs.new("f")
        self.emitter.ins(f"add.f32 {num}, {negex}, 0f3F800000;")
        den = self.emitter.regs.new("f")
        self.emitter.ins(f"add.f32 {den}, {ex}, 0f3F800000;")
        ratio = self.emitter.regs.new("f")
        self.emitter.ins(f"div.rn.f32 {ratio}, {num}, {den};")
        p = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.lt.f32 {p}, {x}, 0f00000000;")
        sign = self.emitter.regs.new("f")
        self.emitter.ins(f"mov.f32 {sign}, 0f3F800000;")
        self.emitter.ins(f"@{p} mov.f32 {sign}, 0fBF800000;")
        out = self.emitter.regs.new("f")
        self.emitter.ins(f"mul.f32 {out}, {ratio}, {sign};")
        return out

    def emit_erf_f32(self, t):
        # Abramowitz & Stegun 7.1.26 (max abs error 1.5e-7):
        #   A = 1 / (1 + 0.3275911 * |t|);  erf(t) = sgn(t) * (1 - (u * e))
        #   with u the degree-5 Horner polynomial in A.
        a = self.emitter.regs.new("f")
        self.emitter.ins(f"abs.f32 {a}, {t};")
        y = self.emitter.regs.new("f")
        self.emitter.ins(f"mul.f32 {y}, {a}, {a};")
        ey = self.emitter.regs.new("f")
        self.emitter.ins(f"neg.f32 {ey}, {y};")
        ex = self._emit_exp_f32(ey)
        tau = self.emitter.regs.new("f")
        self.emitter.ins(f"fma.rn.f32 {tau}, {a}, 0f{float_bits(0.3275911):08X}, 0f3F800000;")
        iv = self.emitter.regs.new("f")
        self.emitter.ins(f"rcp.rn.f32 {iv}, {tau};")
        C5, C4 = 1.061405429, -1.453152027
        C3, C2, C1 = 1.421413741, -0.284496736, 0.254829592
        h = self.emitter.regs.new("f")
        self.emitter.ins(f"fma.rn.f32 {h}, 0f{float_bits(C5):08X}, {iv}, 0f{float_bits(C4):08X};")
        h2 = self.emitter.regs.new("f")
        self.emitter.ins(f"fma.rn.f32 {h2}, {h}, {iv}, 0f{float_bits(C3):08X};")
        h3 = self.emitter.regs.new("f")
        self.emitter.ins(f"fma.rn.f32 {h3}, {h2}, {iv}, 0f{float_bits(C2):08X};")
        h4 = self.emitter.regs.new("f")
        self.emitter.ins(f"fma.rn.f32 {h4}, {h3}, {iv}, 0f{float_bits(C1):08X};")
        h5 = self.emitter.regs.new("f")
        self.emitter.ins(f"mul.f32 {h5}, {h4}, {iv};")
        tail = self.emitter.regs.new("f")
        self.emitter.ins(f"mul.f32 {tail}, {h5}, {ex};")
        negt = self.emitter.regs.new("f")
        self.emitter.ins(f"neg.f32 {negt}, {tail};")
        pos = self.emitter.regs.new("f")
        self.emitter.ins(f"add.f32 {pos}, {negt}, 0f3F800000;")
        p = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.lt.f32 {p}, {t}, 0f00000000;")
        negp = self.emitter.regs.new("f")
        self.emitter.ins(f"neg.f32 {negp}, {pos};")
        out = self.emitter.regs.new("f")
        self.emitter.ins(f"mov.f32 {out}, {pos};")
        self.emitter.ins(f"@{p} mov.f32 {out}, {negp};")
        return out

    def emit_pow_f32(self, x: str, y: str) -> str:
        """x ** y, matching torch.pow for negative bases.

        ``2 ** (y * log2(x))`` only covers x > 0.  A negative base with an
        integral exponent takes the sign of x raised to y, and a negative base
        with a fractional exponent is NaN, which is what torch.pow returns.
        """
        ax = self.emitter.regs.new("f")
        self.emitter.ins(f"abs.f32 {ax}, {x};")
        l2 = self._emit_lg2_f32(ax)
        p = self.emitter.regs.new("f")
        self.emitter.ins(f"mul.f32 {p}, {l2}, {y};")
        mag = self.emitter.regs.new("f")
        self.emitter.ins(f"ex2.approx.f32 {mag}, {p};")

        # y is integral when y == floor(y)
        yi = self.emitter.regs.new("f")
        self.emitter.ins(f"cvt.rmi.f32.f32 {yi}, {y};")
        p_int = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.eq.f32 {p_int}, {y}, {yi};")

        # y is odd when y - 2 * floor(y / 2) == 1
        half = self.emitter.regs.new("f")
        self.emitter.ins(f"mul.f32 {half}, {y}, 0f3F000000;")
        qh = self.emitter.regs.new("f")
        self.emitter.ins(f"cvt.rmi.f32.f32 {qh}, {half};")
        t = self.emitter.regs.new("f")
        self.emitter.ins(f"fma.rn.f32 {t}, {qh}, 0fC0000000, {y};")
        p_odd = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.eq.f32 {p_odd}, {t}, 0f3F800000;")

        p_neg = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.lt.f32 {p_neg}, {x}, 0f00000000;")
        p_signed = self.emitter.regs.new("p")
        self.emitter.ins(f"and.pred {p_signed}, {p_neg}, {p_int};")
        p_oddneg = self.emitter.regs.new("p")
        self.emitter.ins(f"and.pred {p_oddneg}, {p_signed}, {p_odd};")

        neg = self.emitter.regs.new("f")
        self.emitter.ins(f"neg.f32 {neg}, {mag};")
        signed = self.emitter.regs.new("f")
        self.emitter.ins(f"selp.f32 {signed}, {neg}, {mag}, {p_oddneg};")

        # a negative base with a fractional exponent is NaN
        p_notint = self.emitter.regs.new("p")
        self.emitter.ins(f"not.pred {p_notint}, {p_int};")
        p_nan = self.emitter.regs.new("p")
        self.emitter.ins(f"and.pred {p_nan}, {p_neg}, {p_notint};")
        nan = self.emitter.regs.new("f")
        self.emitter.ins(f"mov.f32 {nan}, 0f7FC00000;")
        out = self.emitter.regs.new("f")
        self.emitter.ins(f"selp.f32 {out}, {nan}, {signed}, {p_nan};")

        # 0 ** 0 is 1
        p_zero = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.eq.f32 {p_zero}, {ax}, 0f00000000;")
        p_yzero = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.eq.f32 {p_yzero}, {y}, 0f00000000;")
        p_00 = self.emitter.regs.new("p")
        self.emitter.ins(f"and.pred {p_00}, {p_zero}, {p_yzero};")
        one = self.emitter.regs.new("f")
        self.emitter.ins(f"mov.f32 {one}, 0f3F800000;")
        res = self.emitter.regs.new("f")
        self.emitter.ins(f"selp.f32 {res}, {one}, {out}, {p_00};")
        return res

    @staticmethod
    def arg_dtype(reg, dt=None):
        """The dtype of an operand.

        The IR knows it when the statement carried it; otherwise the register
        name gives it away, since the compiler allocates one register class
        per dtype.
        """
        if dt is not None:
            return dtype_name(dt)
        if reg.startswith("%f"):
            return "f32"
        if reg.startswith("%d"):
            return "f64"
        if reg.startswith("%h"):
            return "f16"
        if reg.startswith("%rd"):
            return "s64"
        if reg.startswith("%p"):
            return "pred"
        return "s32"

    @staticmethod
    def promoted_dtype(dtypes):
        """The dtype a mixed-precision operation promotes to.

        f64 wins over f32, f32 over the integer types -- the same rule
        ``emmy``'s ``dtype_promote`` applies when it stamps an Assign.
        """
        if "f64" in dtypes:
            return "f64"
        if "f32" in dtypes:
            return "f32"
        return "s32"

    def op_on_regs(self, op, args, dtype="f32", dest=None, arg_dtypes=None):
        name = getattr(op, "name", str(op))
        dt = dtype_name(dtype)
        adts = list(arg_dtypes or [])
        arg_dts = [self.arg_dtype(a, adts[i] if i < len(adts) else None)
                   for i, a in enumerate(args)]

        if name in ("greater", "greater_equal", "less", "less_equal", "equal", "not_equal"):
            a, b = args
            target = self.promoted_dtype(arg_dts)
            a = self.coerce(a, arg_dts[0], target)
            b = self.coerce(b, arg_dts[1], target)
            kind = target
            cmp = {
                "greater": "gt",
                "greater_equal": "ge",
                "less": "lt",
                "less_equal": "le",
                "equal": "eq",
                "not_equal": "ne",
            }[name]
            out = self.emitter.regs.new("p")
            self.emitter.ins(f"setp.{cmp}.{kind} {out}, {a}, {b};")
            return out, "pred"

        if name == "where":
            cond, a, b = args
            cdt = arg_dts[0]
            if cdt != "pred":
                p = self.emitter.regs.new("p")
                kind = "f32" if is_float(cdt) else "s32"
                self.emitter.ins(f"setp.ne.{kind} {p}, {cond}, 0;")
                cond = p
            a = self.coerce(a, arg_dts[1], dt)
            b = self.coerce(b, arg_dts[2], dt)
            out = self.emitter.regs.new(reg_kind(dt))
            self.emitter.ins(f"selp.{ptx_type(dt)} {out}, {a}, {b}, {cond};")
            return out, dt

        if name in ("negative", "neg"):
            a = args[0]
            adt = arg_dts[0]
            out = self.emitter.regs.new(reg_kind(adt))
            self.emitter.ins(f"neg.{ptx_type(adt)} {out}, {a};")
            return out, adt

        if name in ("logical_and", "logical_or", "logical_not"):
            if name == "logical_not":
                a = args[0]
                adt = arg_dts[0]
                out = self.emitter.regs.new("p")
                if adt == "pred":
                    self.emitter.ins(f"not.pred {out}, {a};")
                else:
                    self.emitter.ins(f"setp.eq.{ptx_type(adt)} {out}, {a}, 0;")
                return out, "pred"
            a, b = args
            if arg_dts[0] == "pred" and arg_dts[1] == "pred":
                out = self.emitter.regs.new("p")
                self.emitter.ins(f"{name.split('_')[1]}.pred {out}, {a}, {b};")
                return out, "pred"
            out = self.emitter.regs.new("r")
            opc = name.split("_")[1]
            a = self.coerce(a, arg_dts[0], "s32")
            b = self.coerce(b, arg_dts[1], "s32")
            self.emitter.ins(f"{opc}.b32 {out}, {a}, {b};")
            return out, "s32"
        # These ops are f32-only.  emmy promotes their *result* to f32 (the
        # overflow guard in dtype_promote) but leaves the operands at their
        # own precision, so an f64 operand has to be converted first or
        # ptxas sees a type-mismatched instruction.
        if name in F32_ONLY_OPS:
            args = [self.coerce(a, adt, "f32") for a, adt in zip(args, arg_dts)]

        if name in ("exp", "exp_fast"):
            x = args[0]
            t = self.emitter.regs.new("f")
            self.emitter.ins(f"mul.f32 {t}, {x}, 0f3FB8AA3B;")
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"ex2.approx.f32 {out}, {t};")
            return out, "f32"

        if name == "sign":
            x = args[0]
            pg = self.emitter.regs.new("p")
            self.emitter.ins(f"setp.gt.f32 {pg}, {x}, 0f00000000;")
            pl = self.emitter.regs.new("p")
            self.emitter.ins(f"setp.lt.f32 {pl}, {x}, 0f00000000;")
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"mov.f32 {out}, 0f00000000;")
            self.emitter.ins(f"@{pg} mov.f32 {out}, 0f3F800000;")
            self.emitter.ins(f"@{pl} mov.f32 {out}, 0fBF800000;")
            return out, "f32"

        if name in ("log", "log2", "log10"):
            l2 = self._emit_lg2_f32(args[0])
            if name == "log2":
                return l2, "f32"
            out = self.emitter.regs.new("f")
            scale = 0.6931471805599453 if name == "log" else 0.3010299956639812
            self.emitter.ins(f"mul.f32 {out}, {l2}, 0f{float_bits(scale):08X};")
            return out, "f32"

        if name == "softplus":
            x = args[0]
            e = self._emit_exp_f32(x)
            s = self.emitter.regs.new("f")
            self.emitter.ins(f"add.f32 {s}, {e}, 0f3F800000;")
            l2 = self._emit_lg2_f32(s)
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"mul.f32 {out}, {l2}, 0f{float_bits(0.6931471805599453):08X};")
            return out, "f32"

        if name in ("sin", "cos", "tan"):
            if name == "tan":
                s = self.emitter.regs.new("f")
                self.emitter.ins(f"sin.approx.f32 {s}, {args[0]};")
                c = self.emitter.regs.new("f")
                self.emitter.ins(f"cos.approx.f32 {c}, {args[0]};")
                out = self.emitter.regs.new("f")
                self.emitter.ins(f"div.rn.f32 {out}, {s}, {c};")
                return out, "f32"
            out = self.emitter.regs.new("f")
            op = "sin" if name == "sin" else "cos"
            self.emitter.ins(f"{op}.approx.f32 {out}, {args[0]};")
            return out, "f32"

        if name == "pow":
            x, y = args
            return self.emit_pow_f32(x, y), "f32"

        if name == "relu":
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"max.f32 {out}, {args[0]}, 0f00000000;")
            return out, "f32"

        if name in ("sigmoid", "silu"):
            x = args[0]
            neg = self.emitter.regs.new("f")
            self.emitter.ins(f"neg.f32 {neg}, {x};")
            t = self.emitter.regs.new("f")
            self.emitter.ins(f"mul.f32 {t}, {neg}, 0f3FB8AA3B;")
            ex = self.emitter.regs.new("f")
            self.emitter.ins(f"ex2.approx.f32 {ex}, {t};")
            den = self.emitter.regs.new("f")
            self.emitter.ins(f"add.f32 {den}, {ex}, 0f3F800000;")
            sig = self.emitter.regs.new("f")
            self.emitter.ins(f"rcp.rn.f32 {sig}, {den};")
            if name == "sigmoid":
                return sig, "f32"
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"mul.f32 {out}, {x}, {sig};")
            return out, "f32"

        if name == "tanh":
            return self.emit_tanh_f32(args[0]), "f32"

        if name in ("erf", "gelu", "gelu_tanh"):
            x = args[0]
            if name == "gelu_tanh":
                x3 = self.emitter.regs.new("f")
                self.emitter.ins(f"mul.f32 {x3}, {x}, {x};")
                self.emitter.ins(f"mul.f32 {x3}, {x3}, {x};")
                inner = self.emitter.regs.new("f")
                self.emitter.ins(f"fma.rn.f32 {inner}, {x3}, 0f{float_bits(0.044715):08X}, {x};")
                s = self.emitter.regs.new("f")
                self.emitter.ins(f"mul.f32 {s}, {inner}, 0f{float_bits(0.7978845608028654):08X};")
                th = self.emit_tanh_f32(s)
                u = self.emitter.regs.new("f")
                self.emitter.ins(f"add.f32 {u}, {th}, 0f3F800000;")
                halfx = self.emitter.regs.new("f")
                self.emitter.ins(f"mul.f32 {halfx}, {x}, 0f3F000000;")
                out = self.emitter.regs.new("f")
                self.emitter.ins(f"mul.f32 {out}, {halfx}, {u};")
                return out, "f32"
            if name == "gelu":
                inner = self.emitter.regs.new("f")
                self.emitter.ins(f"mul.f32 {inner}, {x}, 0f{float_bits(0.7071067811865476):08X};")
                e = self.emit_erf_f32(inner)
                one = self.emitter.regs.new("f")
                self.emitter.ins(f"add.f32 {one}, {e}, 0f3F800000;")
                half = self.emitter.regs.new("f")
                self.emitter.ins(f"mul.f32 {half}, 0f3F000000, {one};")
                out = self.emitter.regs.new("f")
                self.emitter.ins(f"mul.f32 {out}, {half}, {x};")
                return out, "f32"
            e = self.emit_erf_f32(x)
            return e, "f32"

        if name == "reciprocal":
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"rcp.rn.f32 {out}, {args[0]};")
            return out, "f32"

        if name in ("absolute", "abs"):
            if dt == "f64":
                out = self.emitter.regs.new("d")
                self.emitter.ins(f"abs.f64 {out}, {args[0]};")
                return out, "f64"
            if dt == "f16":
                raise SystemExit(f"unsupported float op abs for f16")
            if is_float(dt):
                out = self.emitter.regs.new("f")
                self.emitter.ins(f"abs.f32 {out}, {args[0]};")
                return out, "f32"
            out = self.emitter.regs.new("r")
            self.emitter.ins(f"abs.s32 {out}, {args[0]};")
            return out, "s32"

        if name == "neg":
            if dt == "f64":
                out = self.emitter.regs.new("d")
                self.emitter.ins(f"neg.f64 {out}, {args[0]};")
                return out, "f64"
            if dt == "f16":
                raise SystemExit(f"unsupported float op neg for f16")
            if is_float(dt):
                out = self.emitter.regs.new("f")
                self.emitter.ins(f"neg.f32 {out}, {args[0]};")
                return out, "f32"
            out = self.emitter.regs.new("r")
            self.emitter.ins(f"neg.s32 {out}, {args[0]};")
            return out, "s32"

        if name in ("copy", "pad"):
            out = self.emitter.regs.new(reg_kind(dt))
            self.emitter.ins(f"mov.{ptx_type(dt)} {out}, {args[0]};")
            return out, dt

        if name == "bitcast":
            # Same-width bit reinterpretation: the destination dtype is the
            # Assign's dtype, the source is the argument's.
            src_dt = adts[0] if adts else dt
            if src_dt == dt:
                out = self.emitter.regs.new(reg_kind(dt))
                self.emitter.ins(f"mov.{ptx_type(dt)} {out}, {args[0]};")
                return out, dt
            form = {
                ("f32", "s32"): ("r", "mov.b32"),
                ("s32", "f32"): ("f", "mov.f32"),
                ("f64", "s64"): ("rd", "mov.b64"),
                ("s64", "f64"): ("d", "mov.f64"),
            }.get((src_dt, dt))
            if form is None:
                raise SystemExit(f"unsupported bitcast {src_dt} -> {dt}")
            kind, ins = form
            out = self.emitter.regs.new(kind)
            self.emitter.ins(f"{ins} {out}, {args[0]};")
            return out, dt

        if name in ("sqrt", "rsqrt"):
            out = self.emitter.regs.new("f")
            ptx = "sqrt.rn.f32" if name == "sqrt" else "rsqrt.approx.f32"
            self.emitter.ins(f"{ptx} {out}, {args[0]};")
            return out, "f32"

        if name == "fma":
            out = self.emitter.regs.new("f")
            self.emitter.ins(f"fma.rn.f32 {out}, {args[0]}, {args[1]}, {args[2]};")
            return out, "f32"

        if name in ("from_f8e4m3", "from_f8e5m2"):
            raise SystemExit(f"unsupported fp8 decode op {name}")

        if is_float(dt):
            if dt == "f16":
                # This ptxas has no .b16 arithmetic or b16<->.f32 conversion.
                raise SystemExit(f"unsupported float op {name!r} for f16")
            # The operands may carry a different precision than the result:
            # emmy stamps the Assign with the promoted dtype but leaves the
            # operands alone, so an f32 + f64 add has to widen the f32 side
            # before ptxas sees it.
            args = [self.coerce(a, adt, dt) for a, adt in zip(args, arg_dts)]
            if name in ("floor_divide", "remainder", "mod"):
                # PTX has no float remainder.  q = floor(a / b), r = a - q * b;
                # this matches Python's float % (sign follows the divisor).
                quot = self.emitter.regs.new(reg_kind(dt))
                self.emitter.ins(f"div.rn.{dt} {quot}, {args[0]}, {args[1]};")
                fl = self.emitter.regs.new(reg_kind(dt))
                self.emitter.ins(f"cvt.rmi.{dt}.{dt} {fl}, {quot};")
                if name == "floor_divide":
                    return fl, dt
                prod = self.emitter.regs.new(reg_kind(dt))
                self.emitter.ins(f"mul.{dt} {prod}, {fl}, {args[1]};")
                out = self.emitter.regs.new(reg_kind(dt))
                self.emitter.ins(f"sub.{dt} {out}, {args[0]}, {prod};")
                return out, dt
            ptxop = {
                "f32": {
                    "add": "add.f32",
                    "sum": "add.f32",
                    "subtract": "sub.f32",
                    "sub": "sub.f32",
                    "multiply": "mul.f32",
                    "prod": "mul.f32",
                    "divide": "div.rn.f32",
                    "true_divide": "div.rn.f32",
                    "maximum": "max.f32",
                    "minimum": "min.f32",
                    "max": "max.f32",
                    "min": "min.f32",
                    "amax": "max.f32",
                    "amin": "min.f32",
                },
                "f64": {
                    "add": "add.f64",
                    "sum": "add.f64",
                    "subtract": "sub.f64",
                    "sub": "sub.f64",
                    "multiply": "mul.f64",
                    "prod": "mul.f64",
                    "divide": "div.rn.f64",
                    "true_divide": "div.rn.f64",
                    "maximum": "max.f64",
                    "minimum": "min.f64",
                    "max": "max.f64",
                    "min": "min.f64",
                    "amax": "max.f64",
                    "amin": "min.f64",
                },
            }[dt].get(name)
            if ptxop is None:
                raise SystemExit(f"unsupported float op {name!r}")
            out = dest or self.emitter.regs.new(reg_kind(dt))
            self.emitter.ins(f"{ptxop} {out}, {args[0]}, {args[1]};")
            return out, dt

        ptxop = {
            "add": "add.s32",
            "sum": "add.s32",
            "subtract": "sub.s32",
            "sub": "sub.s32",
            "multiply": "mul.lo.s32",
            "prod": "mul.lo.s32",
            "divide": "div.s32",
            "true_divide": "div.s32",
            "floor_divide": "div.s32",
            "remainder": "rem.s32",
            "mod": "rem.s32",
            "maximum": "max.s32",
            "minimum": "min.s32",
            "max": "max.s32",
            "min": "min.s32",
            "amax": "max.s32",
            "amin": "min.s32",
            "and": "and.b32",
            "or": "or.b32",
            "xor": "xor.b32",
            "left_shift": "shl.b32",
            "right_shift": "shr.s32",
            "bitwise_right_shift": "shr.s32",
        }.get(name)
        if ptxop is None:
            raise SystemExit(f"unsupported integer op {name!r}")
        args = [self.coerce(a, adt, "s32") for a, adt in zip(args, arg_dts)]
        out = dest or self.emitter.regs.new("r")
        self.emitter.ins(f"{ptxop} {out}, {args[0]}, {args[1]};")
        return out, "s32"

    # ------------------------------------------------------------------
    # statements
    # ------------------------------------------------------------------
    def stmt(self, s):
        if s is None:
            return
        if isinstance(s, (Body, list, tuple)):
            for child in s:
                self.stmt(child)
            return
        method = {
            "Tile": self.stmt_tile,
            "Load": self.stmt_load,
            "Assign": self.stmt_assign,
            "Let": self.stmt_let,
            "Accum": self.stmt_accum,
            "Write": self.stmt_write,
            "Init": self.stmt_init,
            "Cond": self.stmt_cond,
            "Loop": self.stmt_loop,
            "StridedLoop": self.stmt_strided_loop,
            "Smem": self.stmt_smem,
            "Sync": self.stmt_sync,
            "WarpShuffle": self.stmt_warp_shuffle,
            "TreeHalve": self.stmt_tree_halve,
            "CpAsyncCopy": self.stmt_cp_async,
            "CpAsyncCommit": self.stmt_cp_commit,
            "CpAsyncWait": self.stmt_cp_wait,
            "SetMaxNReg": self.stmt_setmaxnreg,
            "Select": self.stmt_select,
            "ZeroPrologue": self.stmt_zero_prologue,
        }.get(type(s).__name__)
        if method is None:
            raise SystemExit(f"unsupported statement {type(s).__name__}: {s!r}")
        method(s)

    def stmt_smem(self, s):
        return

    def stmt_tile(self, s):
        extents = []
        for axis in s.axes:
            if not axis.extent.is_static:
                raise SystemExit(f"dynamic tile axis {axis.name!r} is unsupported")
            extents.append(int(axis.extent.as_static()))
        n = 1
        for e in extents:
            n *= e

        tid = self.special.get("threadIdx.x")
        if tid is None:
            tid = self.emitter.regs.new("r")
            self.emitter.ins(f"mov.u32 {tid}, %tid.x;")
            self.special["threadIdx.x"] = tid

        cta = self.builtin(Builtin("blockIdx.x"))[0]
        gid = self.emitter.regs.new("r")
        if s.block_threads:
            width = int(s.block_threads)
            if s.aux_threads:
                t = self.emitter.regs.new("r")
                self.emitter.ins(f"rem.u32 {t}, {tid}, {width};")
            else:
                t = tid
            self.emitter.ins(f"mad.lo.u32 {gid}, {cta}, {width}, {t};")
        else:
            ntid = self.builtin(Builtin("blockDim.x"))[0]
            self.emitter.ins(f"mad.lo.u32 {gid}, {cta}, {ntid}, {tid};")

        end = self.emitter.label("TILE_END")
        guard = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.ge.u32 {guard}, {gid}, {n};")
        self.emitter.ins(f"@{guard} bra {end};")

        strides = [1] * len(extents)
        for i in range(len(extents) - 2, -1, -1):
            strides[i] = strides[i + 1] * extents[i + 1]

        raster = None
        if s.raster_group and s.raster_axes:
            names = [a.name for a in s.axes]
            if s.raster_axes[0] in names and s.raster_axes[1] in names:
                mi, ni = names.index(s.raster_axes[0]), names.index(s.raster_axes[1])
                if ni == mi + 1:
                    gi, oi = (mi, ni) if s.raster_orient == "m" else (ni, mi)
                    g = min(int(s.raster_group), extents[gi])
                    if extents[gi] > 1 and g > 1:
                        raster = (gi, oi, g)

        if raster is not None:
            gi, oi, g = raster
            ea, eb = extents[gi], extents[oi]
            mi, ni2 = min(gi, oi), max(gi, oi)
            sn = strides[ni2]
            self.emitter.raw("// raster decode (grouped CTA swizzle)")
            rg = self.emitter.regs.new("r")
            self.emitter.ins(f"mov.u32 {rg}, {g};")
            rgeb = self.emitter.regs.new("r")
            self.emitter.ins(f"mov.u32 {rgeb}, {g * eb};")
            reaeb = self.emitter.regs.new("r")
            self.emitter.ins(f"mov.u32 {reaeb}, {ea * eb};")
            rea = self.emitter.regs.new("r")
            self.emitter.ins(f"mov.u32 {rea}, {ea};")
            sub = gid if sn == 1 else None
            if sub is None:
                sub = self.emitter.regs.new("r")
                self.emitter.ins(f"div.u32 {sub}, {gid}, {sn};")
            if mi != 0:
                rsub = self.emitter.regs.new("r")
                self.emitter.ins(f"rem.u32 {rsub}, {sub}, {reaeb};")
                sub = rsub
            if ea % g == 0:
                t1 = self.emitter.regs.new("r")
                t2 = self.emitter.regs.new("r")
                self.emitter.ins(f"div.u32 {t1}, {sub}, {rgeb};")
                self.emitter.ins(f"mad.lo.u32 {t1}, {t1}, {rg}, 0;")
                self.emitter.ins(f"rem.u32 {t2}, {sub}, {rg};")
                a_expr_reg = self.emitter.regs.new("r")
                self.emitter.ins(f"add.u32 {a_expr_reg}, {t1}, {t2};")
                b_expr_reg = self.emitter.regs.new("r")
                self.emitter.ins(f"rem.u32 {b_expr_reg}, {sub}, {rgeb};")
                self.emitter.ins(f"div.u32 {b_expr_reg}, {b_expr_reg}, {rg};")
            else:
                rgrp = self.emitter.regs.new("r")
                self.emitter.ins(f"div.u32 {rgrp}, {sub}, {rgeb};")
                rsz = self.emitter.regs.new("r")
                self.emitter.ins(f"mad.lo.u32 {rsz}, {rgrp}, {rg}, 0;")
                self.emitter.ins(f"sub.u32 {rsz}, {rea}, {rsz};")
                self.emitter.ins(f"min.u32 {rsz}, {rsz}, {rg};")
                tmp = self.emitter.regs.new("r")
                self.emitter.ins(f"rem.u32 {tmp}, {sub}, {rgeb};")
                tmp2 = self.emitter.regs.new("r")
                self.emitter.ins(f"rem.u32 {tmp2}, {tmp}, {rsz};")
                a_expr_reg = self.emitter.regs.new("r")
                self.emitter.ins(f"mad.lo.u32 {a_expr_reg}, {rgrp}, {rg}, 0;")
                self.emitter.ins(f"add.u32 {a_expr_reg}, {a_expr_reg}, {tmp2};")
                b_expr_reg = self.emitter.regs.new("r")
                self.emitter.ins(f"div.u32 {b_expr_reg}, {tmp}, {rsz};")

        for i, (axis, extent, stride) in enumerate(zip(s.axes, extents, strides)):
            if extent == 1:
                reg = self.int_reg(0)
            elif raster is not None and i == gi:
                reg = a_expr_reg if raster is not None else None
            elif raster is not None and i == oi:
                reg = b_expr_reg if raster is not None else None
            elif i == len(extents) - 1:
                reg = self.emitter.regs.new("r")
                self.emitter.ins(f"rem.s32 {reg}, {gid}, {extent};")
            elif i == 0:
                reg = self.emitter.regs.new("r")
                self.emitter.ins(f"div.s32 {reg}, {gid}, {stride};")
            else:
                q = self.emitter.regs.new("r")
                self.emitter.ins(f"div.s32 {q}, {gid}, {stride};")
                reg = self.emitter.regs.new("r")
                self.emitter.ins(f"rem.s32 {reg}, {q}, {extent};")
            self.values[axis.name] = reg
            self.value_dtypes[axis.name] = "s32"

        self.stmt(s.body)
        self.emitter.raw(end + ":")

    def stmt_load(self, s):
        names = s.names if getattr(s, "names", None) else (getattr(s, "name", None),)
        input_name = s.input or getattr(s, "name", None)
        if input_name is None:
            raise SystemExit(f"Load without input: {s!r}")
        dtype = dtype_name(s.dtype) if s.dtype is not None else dtype_name(self.dtype_of(input_name))
        index = tuple(s.index or ())
        w = len(names)
        aligned = self._flat_index_aligned(input_name, index, w) and self._size_divisible(input_name, w)
        if (
            w in (2, 4)
            and dtype == "f32"
            and aligned
        ):
            flat = self.flatten(input_name, index)
            regs = [self.emitter.regs.new("f") for _ in names]
            if input_name in self.smem:
                addr = self.smem_elem_addr(input_name, flat)
                self.emitter.ins(f"ld.shared.v{w}.{dtype} {{{', '.join(regs)}}}, [{addr}];")
            else:
                addr = self.global_addr(input_name, flat)
                self.emitter.ins(f"ld.global.v{w}.{dtype} {{{', '.join(regs)}}}, [{addr}];")
            for name, reg in zip(names, regs):
                self.values[name] = reg
                self.value_dtypes[name] = dtype
            return
        for k, name in enumerate(names):
            if name is None:
                continue
            idx = list(index)
            if k and idx:
                idx[-1] = self.add_const_expr(idx[-1], k)
            flat = self.flatten(input_name, idx)
            if input_name in self.smem:
                addr = self.smem_elem_addr(input_name, flat)
                reg = self.emitter.regs.new(reg_kind(dtype))
                self.emitter.ins(f"ld.shared.{ptx_type(dtype)} {reg}, [{addr}];")
            else:
                addr = self.global_addr(input_name, flat)
                reg = self.emitter.regs.new(reg_kind(dtype))
                self.emitter.ins(f"ld.global.{ptx_type(dtype)} {reg}, [{addr}];")
            self.values[name] = reg
            self.value_dtypes[name] = dtype

    def stmt_assign(self, s):
        dtype = dtype_name(s.dtype) if s.dtype is not None else self.value_dtypes[s.args[0]]
        regs = []
        arg_dts = []
        for name in s.args:
            if name not in self.values:
                raise SystemExit(f"Assign {s.name}: unknown value {name!r}")
            regs.append(self.values[name])
            arg_dts.append(self.value_dtypes[name])
        reg, dt = self.op_on_regs(s.op, regs, dtype, arg_dtypes=arg_dts)
        self.values[s.name] = reg
        self.value_dtypes[s.name] = dt

    def stmt_let(self, s):
        reg, dt = self.expr(s.value)
        self.values[s.name] = reg
        self.value_dtypes[s.name] = dtype_name(s.dtype) if s.dtype is not None else dt

    def stmt_accum(self, s):
        dtype = dtype_name(s.dtype) if s.dtype is not None else self.value_dtypes[s.name]
        if s.name not in self.values:
            raise SystemExit(f"Accum {s.name}: accumulator was not seeded")
        acc = self.values[s.name]
        a = self.values[s.base] if s.base else acc
        b = self.values[s.value]
        arg_dts = [self.value_dtypes.get(s.base or s.name), self.value_dtypes[s.value]]
        reg, dt = self.op_on_regs(s.op, (a, b), dtype, dest=acc, arg_dtypes=arg_dts)
        self.values[s.name] = reg
        self.value_dtypes[s.name] = dt

    def stmt_write(self, s):
        values = s.values if getattr(s, "values", None) else (getattr(s, "value", None),)
        output = s.output or getattr(s, "name", None)
        if output is None:
            raise SystemExit(f"Write without output: {s!r}")
        dtype = dtype_name(s.value_dtype) if s.value_dtype is not None else dtype_name(self.dtype_of(output))
        # The destination's dtype decides the memory layout, so a value that
        # was computed at another precision is converted on the way out --
        # this is what emmy's own renderer does.  The value's own dtype is
        # used when the destination is unknown.
        if output not in self.smem:
            try:
                buf_dtype = dtype_name(self.dtype_of(output))
            except SystemExit:
                buf_dtype = None
            if buf_dtype is not None and buf_dtype != dtype:
                dtype = buf_dtype
        index = tuple(s.index or ())
        w = len(values)
        aligned = self._flat_index_aligned(output, index, w) and self._size_divisible(output, w)
        if (
            w in (2, 4)
            and not s.atomic
            and dtype == "f32"
            and all(n in self.value_dtypes and self.value_dtypes[n] == "f32" for n in values)
            and aligned
        ):
            flat = self.flatten(output, index)
            regs = [self.values[n] for n in values]
            if output in self.smem:
                addr = self.smem_elem_addr(output, flat)
                self.emitter.ins(f"st.shared.v{w}.{dtype} [{addr}], {{{', '.join(regs)}}};")
            else:
                addr = self.global_addr(output, flat)
                self.emitter.ins(f"st.global.v{w}.{dtype} [{addr}], {{{', '.join(regs)}}};")
            return
        for k, name in enumerate(values):
            if name is None:
                continue
            if name not in self.values:
                raise SystemExit(f"Write {output}: unknown value {name!r}")
            idx = list(index)
            if k and idx:
                idx[-1] = self.add_const_expr(idx[-1], k)
            flat = self.flatten(output, idx)
            reg = self.values[name]
            value_dtype = self.value_dtypes.get(name)
            if value_dtype is not None and value_dtype != dtype:
                reg = self.coerce(reg, value_dtype, dtype)
            if output in self.smem:
                addr = self.smem_elem_addr(output, flat)
                self.emitter.ins(f"st.shared.{ptx_type(dtype)} [{addr}], {reg};")
            else:
                addr = self.global_addr(output, flat)
                if s.atomic:
                    op = "add" if s.atomic is True else str(s.atomic)
                    self.emitter.ins(f"red.global.{op}.{ptx_type(dtype)} [{addr}], {reg};")
                else:
                    self.emitter.ins(f"st.global.{ptx_type(dtype)} [{addr}], {reg};")

    def stmt_init(self, s):
        dtype = dtype_name(s.dtype) if s.dtype is not None else "f32"
        if is_float(dtype):
            reg = self.fconst(float(s.identity), dtype)
        else:
            reg = self.int_reg(int(s.identity))
        self.values[s.name] = reg
        self.value_dtypes[s.name] = dtype

    def stmt_cond(self, s):
        p = self.to_pred(s.cond)
        els = self.emitter.label("ELSE")
        end = self.emitter.label("COND_END")
        self.emitter.ins(f"@!{p} bra {els};")
        self.stmt(s.body)
        self.emitter.ins(f"bra {end};")
        self.emitter.raw(els + ":")
        self.stmt(s.else_body)
        self.emitter.raw(end + ":")

    def seed_accums(self, body, loop_axis: str | None = None):
        seen = set()
        for st in body:
            if isinstance(st, Accum) and st.name not in seen:
                # Axis-driven: only seed accumulators whose axes reduce over
                # the enclosing loop axis (or are scalar/axis-less).
                if loop_axis is not None and st.axes and loop_axis not in st.axes:
                    continue
                seen.add(st.name)
                dtype = dtype_name(st.dtype) if st.dtype is not None else "f32"
                identity = st.op.identity
                if identity is None:
                    raise SystemExit(f"no identity for accumulator op {st.op.name}")
                if is_float(dtype):
                    reg = self.fconst(float(identity), dtype)
                else:
                    reg = self.int_reg(int(identity))
                self.values[st.name] = reg
                self.value_dtypes[st.name] = dtype

    def stmt_loop(self, s):
        if not s.axis.extent.is_static:
            raise SystemExit(f"dynamic loop axis {s.axis.name!r} is unsupported")
        extent = int(s.axis.extent.as_static())
        if s.seed:
            self.seed_accums(s.body, loop_axis=s.axis.name)
        old = self.values.get(s.axis.name)
        self.value_dtypes[s.axis.name] = "s32"
        if s.unroll and extent <= MAX_UNROLL:
            # Fully unroll a static trip-count loop: emit the body once per
            # iteration with the axis bound to a constant, removing branch
            # and counter overhead.  Wide loops stay rolled (see MAX_UNROLL).
            for i in range(extent):
                self.values[s.axis.name] = self.int_reg(i)
                self.stmt(s.body)
        else:
            counter = self.int_reg(0)
            head = self.emitter.label("LOOP")
            tail = self.emitter.label("LOOP_END")
            self.emitter.raw(head + ":")
            p = self.emitter.regs.new("p")
            self.emitter.ins(f"setp.ge.s32 {p}, {counter}, {extent};")
            self.emitter.ins(f"@{p} bra {tail};")
            self.values[s.axis.name] = counter
            self.stmt(s.body)
            self.emitter.ins(f"add.s32 {counter}, {counter}, 1;")
            self.emitter.ins(f"bra {head};")
            self.emitter.raw(tail + ":")
        if old is not None:
            self.values[s.axis.name] = old
        else:
            self.values.pop(s.axis.name, None)
            self.value_dtypes.pop(s.axis.name, None)

    def _strided_trip_count(self, s):
        """Static trip count of a StridedLoop, or None when it is not known."""
        start = self.const_int(s.start)
        step = self.const_int(s.step)
        if start is None or step is None or step == 0:
            return None
        if s.end is None:
            if not s.axis.extent.is_static:
                return None
            end = int(s.axis.extent.as_static())
        else:
            end = self.const_int(s.end)
            if end is None:
                return None
        if step > 0:
            return max(0, -(-(end - start) // step))
        if step < 0:
            return max(0, -(-(start - end) // (-step)))
        return None

    def _unroll_strided(self, s, trip):
        """Emit a StridedLoop body once per statically-known iteration."""
        start = self.const_int(s.start)
        step = self.const_int(s.step)
        axis = s.axis.name
        old = self.values.get(axis)
        self.value_dtypes[axis] = "s32"
        for i in range(trip):
            self.values[axis] = self.int_reg(start + i * step)
            self.stmt(s.body)
        if old is not None:
            self.values[axis] = old
        else:
            self.values.pop(axis, None)
            self.value_dtypes.pop(axis, None)

    def stmt_strided_loop(self, s):
        loop_axis = getattr(s, "axis", None)
        loop_axis = loop_axis.name if loop_axis is not None else None
        if s.seed:
            self.seed_accums(s.body, loop_axis=loop_axis)
        trip = self._strided_trip_count(s)
        if getattr(s, "unroll", False) and trip is not None and 0 < trip <= MAX_UNROLL:
            self._unroll_strided(s, trip)
            return
        start, _ = self.expr(s.start)
        counter = self.emitter.regs.new("r")
        self.emitter.ins(f"mov.s32 {counter}, {start};")
        step, _ = self.expr(s.step)
        if s.end is None:
            if not s.axis.extent.is_static:
                raise SystemExit(f"dynamic strided loop axis {s.axis.name!r} is unsupported")
            end = self.int_reg(int(s.axis.extent.as_static()))
        else:
            end, _ = self.expr(s.end)
        head = self.emitter.label("SL")
        tail = self.emitter.label("SL_END")
        self.emitter.raw(head + ":")
        p = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.ge.s32 {p}, {counter}, {end};")
        self.emitter.ins(f"@{p} bra {tail};")
        old = self.values.get(s.axis.name)
        self.values[s.axis.name] = counter
        self.value_dtypes[s.axis.name] = "s32"
        self.stmt(s.body)
        self.emitter.ins(f"add.s32 {counter}, {counter}, {step};")
        self.emitter.ins(f"bra {head};")
        self.emitter.raw(tail + ":")
        if old is not None:
            self.values[s.axis.name] = old
        else:
            self.values.pop(s.axis.name, None)
            self.value_dtypes.pop(s.axis.name, None)

    def stmt_sync(self, s):
        if s.warp:
            self.emitter.ins("bar.warp.sync 0xffffffff;")
        elif s.count is not None:
            self.emitter.ins(f"bar.sync {int(s.barrier_id)}, {int(s.count)};")
        else:
            self.emitter.ins(f"bar.sync {int(s.barrier_id)};")

    def stmt_warp_shuffle(self, s):
        dtype = dtype_name(s.dtype)
        mask = self.emitter.regs.new("r")
        self.emitter.ins(f"activemask.b32 {mask};")
        step = int(s.length) // 2
        while step > 0:
            for state, state_b in zip(s.state, s.state_b):
                a = self.values[state]
                if dtype == "f64":
                    reg = self._f64_lane_exchange(a, step, mask)
                else:
                    bits = self.emitter.regs.new("r")
                    self.emitter.ins(f"mov.b32 {bits}, {a};")
                    other = self.emitter.regs.new("r")
                    self.emitter.ins(
                        f"shfl.sync.bfly.b32 {other}, {bits}, {step}, 0x1f, {mask};"
                    )
                    if is_float(dtype):
                        reg = self.emitter.regs.new("f")
                        self.emitter.ins(f"mov.b32 {reg}, {other};")
                    else:
                        reg = other
                self.values[state_b] = reg
                self.value_dtypes[state_b] = dtype
            self.stmt(s.combine_states)
            step //= 2

    def _f64_lane_exchange(self, reg: str, step: int, mask: str) -> str:
        """Exchange a 64-bit value with the lane at ``lane ^ step``.

        The value travels as two 32-bit halves through the register shuffle.
        A shared-memory bounce cannot be used here: its slot index only knows
        the lane, so two warps of the same block would overwrite each other's
        value, and ptxas rejects ``mov.b32`` from a .f64 register.
        """
        bits = self.emitter.regs.new("rd")
        self.emitter.ins(f"mov.b64 {bits}, {reg};")
        lo = self.emitter.regs.new("r")
        self.emitter.ins(f"cvt.u32.u64 {lo}, {bits};")
        hi64 = self.emitter.regs.new("rd")
        self.emitter.ins(f"shr.u64 {hi64}, {bits}, 32;")
        hi = self.emitter.regs.new("r")
        self.emitter.ins(f"cvt.u32.u64 {hi}, {hi64};")
        lo2 = self.emitter.regs.new("r")
        self.emitter.ins(f"shfl.sync.bfly.b32 {lo2}, {lo}, {step}, 0x1f, {mask};")
        hi2 = self.emitter.regs.new("r")
        self.emitter.ins(f"shfl.sync.bfly.b32 {hi2}, {hi}, {step}, 0x1f, {mask};")
        pair = self.emitter.regs.new("rd")
        self.emitter.ins(f"mov.b64 {pair}, {{{lo2}, {hi2}}};")
        out = self.emitter.regs.new("d")
        self.emitter.ins(f"mov.f64 {out}, {pair};")
        return out

    def _smem_index(self, buf: str, index_expr: int) -> str:
        return self.int_reg(index_expr)

    def stmt_tree_halve(self, s):
        dtype = dtype_name(s.dtype)
        tid = self.values[s.tid_var]
        inner = s.inner
        bufs = list(s.bufs)
        states = list(s.state)
        state_bs = list(s.state_b)
        if len(bufs) == 1 and len(states) > 1:
            bufs = bufs * len(states)

        # A segment-indexed tree (emmy's transposed ``coop-t`` combine) keeps
        # ``length`` SEGMENTS of ``scale`` slots per slab: the slot is
        # ``t*scale + iv`` and the broadcast root is ``iv`` -- the lane's own
        # slot 0 inside its segment.  ``iv`` is a runtime value, not a
        # component index: each state component has its own slab.
        inner_reg = None
        scale = 0
        if inner is not None:
            iv_name, scale = inner
            scale = int(scale)
            if iv_name not in self.values:
                raise SystemExit(
                    f"TreeHalve inner variable {iv_name!r} is not in scope"
                )
            inner_reg = self.values[iv_name]

        def slot_expr(component: int, plus: int):
            if inner is None:
                return self.add_int(tid, plus) if plus else tid
            base = self.mul_int(tid, scale)
            base = self.add_int(base, inner_reg)
            if plus:
                base = self.add_int(base, plus * scale)
            return base

        step = int(s.length) // 2
        while step > 0:
            p = self.emitter.regs.new("p")
            skip = self.emitter.label("TREE_SKIP")
            self.emitter.ins(f"setp.lt.s32 {p}, {tid}, {step};")
            self.emitter.ins(f"@!{p} bra {skip};")
            for comp, (buf, state, state_b) in enumerate(zip(bufs, states, state_bs)):
                a_idx = slot_expr(comp, 0)
                b_idx = slot_expr(comp, step)
                a_addr = self.smem_elem_addr(buf, a_idx)
                b_addr = self.smem_elem_addr(buf, b_idx)
                a = self.emitter.regs.new(reg_kind(dtype))
                b = self.emitter.regs.new(reg_kind(dtype))
                self.emitter.ins(f"ld.shared.{ptx_type(dtype)} {a}, [{a_addr}];")
                self.emitter.ins(f"ld.shared.{ptx_type(dtype)} {b}, [{b_addr}];")
                self.values[state] = a
                self.value_dtypes[state] = dtype
                self.values[state_b] = b
                self.value_dtypes[state_b] = dtype
            self.stmt(s.combine_states)
            for comp, (buf, state) in enumerate(zip(bufs, states)):
                a_idx = slot_expr(comp, 0)
                a_addr = self.smem_elem_addr(buf, a_idx)
                self.emitter.ins(
                    f"st.shared.{ptx_type(dtype)} [{a_addr}], {self.values[state]};"
                )
            self.emitter.raw(skip + ":")
            self._tree_barrier(s)
            step //= 2

        for comp, (buf, state) in enumerate(zip(bufs, states)):
            # the segment root is the lane's own slot 0 (``iv``), the flat
            # tree's root is slot 0
            root = inner_reg if inner is not None else self.int_reg(0)
            addr = self.smem_elem_addr(buf, root)
            reg = self.emitter.regs.new(reg_kind(dtype))
            self.emitter.ins(f"ld.shared.{ptx_type(dtype)} {reg}, [{addr}];")
            self.values[state] = reg
            self.value_dtypes[state] = dtype

    def _tree_barrier(self, s):
        if s.barrier_count is not None:
            self.emitter.ins(f"bar.sync {int(s.barrier_id)}, {int(s.barrier_count)};")
        else:
            self.emitter.ins(f"bar.sync {int(s.barrier_id)};")

    def stmt_cp_async(self, s):
        if s.swizzle not in (None, "NONE"):
            raise SystemExit(f"unsupported cp.async swizzle {s.swizzle!r}")
        if s.smem_index:
            flat = self.flatten(s.smem, s.smem_index)
        else:
            flat = self.int_reg(0)
        if s.lane_index is not None:
            cols = self.shape_of(s.smem)[-1]
            row, _ = self.expr(s.lane_index[0])
            col, _ = self.expr(s.lane_index[1])
            lane = self.add_int(self.mul_int(row, cols), col)
            flat = self.add_int(flat, lane)
        dst = self.smem_elem_addr(s.smem, flat)
        src_flat = self.flatten(s.src, s.src_index)
        src = self.global_addr(s.src, src_flat)
        nbytes = int(s.nbytes)
        op = "cg" if nbytes == 16 else "ca"
        self.emitter.ins(
            f"cp.async.{op}.shared.global [{dst}], [{src}], {nbytes};"
        )

    def stmt_cp_commit(self, s):
        self.emitter.ins("cp.async.commit_group;")

    def stmt_cp_wait(self, s):
        self.emitter.ins(f"cp.async.wait_group {int(s.group)};")

    def stmt_setmaxnreg(self, s):
        """``SetMaxNReg`` is a register-budget hint, not a semantic operation.

        It asks Hopper+ to move registers between the producer and consumer
        halves of a warp-specialized kernel.  The instruction is unknown to
        this runner's ``ptxas`` ("Not a name of any known instruction"), so
        emitting it would turn a working kernel into a compile error.  The
        statement is dropped; only the register/occupancy trade-off is lost.
        """
        print(
            f"warning: ignoring SetMaxNReg({int(s.count)}, {s.direction!r}); "
            "this ptxas cannot assemble setmaxnreg",
            file=sys.stderr,
        )

    def stmt_select(self, s):
        branches = list(s.branches)
        if not branches:
            raise SystemExit(f"Select {s.name} has no branches")
        last_val, dt = self.expr(branches[-1].value)
        dtype = dtype_name(dt)
        out = self.emitter.regs.new(reg_kind(dtype))
        self.emitter.ins(f"mov.{ptx_type(dtype)} {out}, {last_val};")
        self.values[s.name] = out
        self.value_dtypes[s.name] = dtype
        for branch in reversed(branches[:-1]):
            p = self.to_pred(branch.select)
            v, vdt = self.expr(branch.value)
            v = self.coerce(v, vdt, dtype)
            self.emitter.ins(f"@{p} mov.{ptx_type(dtype)} {out}, {v};")

    def stmt_zero_prologue(self, s):
        dst = s.dst
        words = int(s.words)
        if words <= 0:
            return
        cta = self.builtin(Builtin("blockIdx.x"))[0]
        guard = self.emitter.regs.new("p")
        self.emitter.ins(f"setp.ne.u32 {guard}, {cta}, 0;")
        end = self.emitter.label("ZPRO_END")
        self.emitter.ins(f"@{guard} bra {end};")
        tid = self.special.get("threadIdx.x")
        if tid is None:
            tid = self.emitter.regs.new("r")
            self.emitter.ins(f"mov.u32 {tid}, %tid.x;")
            self.special["threadIdx.x"] = tid
        ntid = self.emitter.regs.new("r")
        self.emitter.ins(f"mov.u32 {ntid}, %ntid.x;")
        base = self.global_addr(dst, (0,))
        wreg = self.emitter.regs.new("r")
        self.emitter.ins(f"mov.u32 {wreg}, {tid};")
        loop = self.emitter.label("ZPRO_LOOP")
        done = self.emitter.label("ZPRO_DONE")
        p2 = self.emitter.regs.new("p")
        self.emitter.raw(loop + ":")
        self.emitter.ins(f"setp.ge.u32 {p2}, {wreg}, {words};")
        self.emitter.ins(f"@{p2} bra {done};")
        addr = self.emitter.regs.new("r")
        self.emitter.ins(f"mad.lo.u32 {addr}, {wreg}, 4;")
        self.emitter.ins(f"st.global.u32 [{base} + {addr}], 0;")
        self.emitter.ins(f"add.u32 {wreg}, {wreg}, {ntid};")
        self.emitter.ins(f"bra {loop};")
        self.emitter.raw(end + ":")

    # ------------------------------------------------------------------
    # module
    # ------------------------------------------------------------------
    def _param_decl(self, name: str, arg: dict) -> str:
        dtype = dtype_name(arg.get("dtype", "f32"))
        scalar = bool(arg.get("scalar")) or len(arg.get("shape", [])) == 0
        if scalar:
            return f"        .param .{ptx_type(dtype)} p_{sanitize(name)}"
        return f"        .param .u64 p_{sanitize(name)}"

    def emit_prologue(self):
        for name in self.arg_order:
            arg = self.args.get(name, {"name": name, "dtype": self.dtype_of(name), "shape": self.shape_of(name)})
            dtype = dtype_name(arg.get("dtype", "f32"))
            scalar = bool(arg.get("scalar")) or len(arg.get("shape", [])) == 0
            pname = f"p_{sanitize(name)}"
            if scalar:
                if dtype == "f32":
                    reg = self.emitter.regs.new("f")
                    self.emitter.ins(f"ld.param.f32 {reg}, [{pname}];")
                elif dtype == "f64":
                    reg = self.emitter.regs.new("d")
                    self.emitter.ins(f"ld.param.f64 {reg}, [{pname}];")
                elif dtype == "f16":
                    reg = self.emitter.regs.new("h")
                    self.emitter.ins(f"ld.param.b16 {reg}, [{pname}];")
                else:
                    reg = self.emitter.regs.new("r")
                    self.emitter.ins(f"ld.param.s32 {reg}, [{pname}];")
                self.values[name] = reg
                self.value_dtypes[name] = dtype
            else:
                raw = self.emitter.regs.new("rd")
                base = self.emitter.regs.new("rd")
                self.emitter.ins(f"ld.param.u64 {raw}, [{pname}];")
                self.emitter.ins(f"cvta.to.global.u64 {base}, {raw};")
                self.base[name] = base

        tid = self.emitter.regs.new("r")
        self.emitter.ins(f"mov.u32 {tid}, %tid.x;")
        self.special["threadIdx.x"] = tid
        lane = self.emitter.regs.new("r")
        self.emitter.ins(f"and.b32 {lane}, {tid}, 31;")
        warp = self.emitter.regs.new("r")
        self.emitter.ins(f"shr.u32 {warp}, {tid}, 5;")
        self.values["lane"] = lane
        self.value_dtypes["lane"] = "s32"
        self.values["warp"] = warp
        self.value_dtypes["warp"] = "s32"

        for name in self.smem:
            self._materialize_smem_base(name)

    def check(self) -> list[str]:
        """Validate the IR without producing PTX. Returns a list of issues (empty = OK)."""
        issues = []
        if not self.ir_path.exists():
            return [f"IR file not found: {self.ir_path}"]
        try:
            self.raw = json.loads(self.ir_path.read_text())
        except json.JSONDecodeError as exc:
            return [f"invalid JSON: {exc}"]
        try:
            self.graph = Graph.from_dict(self.raw)
        except Exception as exc:
            return [f"IR parse error: {exc}"]
        try:
            self.kernel = self._find_kernel()
        except SystemExit as exc:
            return [str(exc)]
        if getattr(self.kernel, "serial", None):
            issues.append("KernelOp.serial is not empty (not implemented)")
        try:
            self.meta = self._find_meta()
            self.kmeta = self._kernel_meta()
        except SystemExit as exc:
            issues.append(str(exc))
        if not self.kmeta:
            issues.append("no meta.json found — argument order and launch geometry are unknown")
        # Validate target
        try:
            ptx_header(self.ptx_version, self.target)
        except ValueError as exc:
            issues.append(str(exc))
        # Check for unsupported ops
        try:
            body = self.kernel.body
            statements = body if isinstance(body, list) else [body]
            for stmt in statements:
                # Quick scan for known-unsupported patterns
                if "SetMaxNReg" in str(stmt):
                    issues.append("SetMaxNReg will be dropped (ptxas does not support it)")
        except Exception:
            pass
        return issues

    def compile(self) -> str:
        self.plan_smem()
        self.emit_prologue()
        self.stmt(self.kernel.body)
        self.emitter.ins("ret;")

        params = ",\n".join(
            self._param_decl(name, self.args.get(name, {"dtype": self.dtype_of(name), "shape": self.shape_of(name)}))
            for name in self.arg_order
        )
        out = [ptx_header(self.ptx_version, self.target)]
        if self.shared_decls:
            out.append("\n".join(self.shared_decls) + "\n")
        out.append(f".visible .entry {self.entry}(\n{params}\n)\n{{")
        out.extend(self.emitter.regs.declarations())
        out.append("")
        out.extend(self.emitter.lines)
        out.append("}\n")
        return "\n".join(out)


def compile_file(path, *, target: str = "sm_89", ptx_version: str = "8.0",
            cache_dir=None) -> str:
    if cache_dir is not None:
        import os
        os.makedirs(cache_dir, exist_ok=True)
        key = _cache_key(path, target, ptx_version)
        cache_path = os.path.join(cache_dir, key + ".ptx")
        if os.path.exists(cache_path):
            return open(cache_path).read()
    result = Compiler(path, target=target, ptx_version=ptx_version).compile()
    if cache_dir is not None:
        with open(cache_path, "w") as f:
            f.write(result)
    return result

def _cache_key(path, target, ptx_version) -> str:
    h = hashlib.sha256()
    h.update(open(path, "rb").read())
    meta = Path(path).with_name("meta.json")
    if meta.exists():
        h.update(meta.read_bytes())
    h.update(f"{target}:{ptx_version}".encode())
    return h.hexdigest()[:16]


def main():
    if len(sys.argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json", file=sys.stderr)
        sys.exit(2)
    try:
        ptx = compile_file(sys.argv[1])
    except Exception as exc:  # noqa: BLE001 - report the real failure on stderr
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    sys.stdout.write(ptx)


if __name__ == "__main__":
    main()
