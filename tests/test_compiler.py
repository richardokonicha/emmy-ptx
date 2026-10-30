import contextlib
import importlib.util
import io
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path("/Users/ro/Fugoku/RESEARCHPROJECT/emmyhackathon")


def load_compiler():
    import emmy_ptx.compiler as _c
    return _c


def ir_path(problem: str) -> Path:
    return next((ROOT / "problems" / problem).glob("*.ir.json"))


def dev_ir_path(problem: str) -> Path:
    for directory in (ROOT / "dev" / "cases", ROOT / "dev" / "cases_oops"):
        found = list(directory.glob(problem + "/*.ir.json"))
        if found:
            return found[0]
    raise AssertionError(f"no dev IR for {problem}")


def rewrite_write_index(src: Path, dst: Path, old: str, new: str) -> Path:
    """Copy a problem package, rewriting only the Write statement's index text.

    The IR ``body`` field is a list of statement strings; locate the statement
    containing ``Write(`` and patch its index arguments textually.
    """
    shutil.copytree(src, dst, dirs_exist_ok=True)
    ir = next(dst.glob("*.ir.json"))
    doc = json.loads(ir.read_text())
    for node in doc["nodes"].values():
        if node["op"] != "KernelOp":
            continue
        body = node["op_fields"]["body"]
        statements = body if isinstance(body, list) else [body]
        for i, stmt in enumerate(statements):
            if "Write(" not in stmt:
                continue
            head, sep, tail = stmt.rpartition("Write(")
            assert sep, "no Write statement in the IR"
            statements[i] = head + "Write(" + tail.replace(old, new, 1)
        node["op_fields"]["body"] = statements if isinstance(body, list) else statements[0]
    ir.write_text(json.dumps(doc, indent=1))
    return ir


class CompilerTests(unittest.TestCase):
    def test_float_bits(self):
        mod = load_compiler()
        self.assertEqual(mod.float_bits(1.0), 0x3F800000)
        self.assertEqual(mod.float_bits(float("-inf")), 0xFF800000)
        self.assertEqual(mod.float_bits(0.0), 0)

    def test_p00_ptx_shape(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("p00_scalar_add"))
        self.assertIn(".version 8.0", ptx)
        self.assertIn(".entry k_add_pointwise", ptx)
        self.assertIn(".param .u64 p_x0", ptx)
        self.assertIn("ld.global.f32", ptx)
        self.assertIn("add.f32", ptx)
        self.assertIn("st.global.v2.f32", ptx)
        self.assertTrue(ptx.endswith("}\n"))

    def test_r01_has_shared_and_shuffle(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("r01_array_sum"))
        self.assertIn(".shared .align 4 .b8 acc0_smem[16];", ptx)
        self.assertIn("shfl.sync.bfly.b32", ptx)
        self.assertIn("bar.sync 0;", ptx)

    def test_m01_uses_dynamic_shared_and_cp_async(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("m01_matmul_512"))
        self.assertIn(".extern .shared", ptx)
        self.assertIn("cp.async.cg.shared.global", ptx)
        self.assertIn("cp.async.wait_group 2;", ptx)

    def test_all_open_problems_compile(self):
        mod = load_compiler()
        for problem in sorted((ROOT / "problems").iterdir()):
            if not (problem / "meta.json").exists():
                continue
            ptx = mod.compile_file(ir_path(problem.name))
            self.assertIn(".visible .entry", ptx, problem.name)
            self.assertIn("ret;", ptx, problem.name)


class VectorMemoryTests(unittest.TestCase):
    """TASK.md: a grouped write ``buf[i] = (v0, v1, v2, v3)`` is a vector store."""

    def test_grouped_write_becomes_one_vector_store(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("p01_vector_add"))
        self.assertEqual(ptx.count("st.global"), 1)
        self.assertIn("st.global.v2.f32", ptx)

    def test_vector_store_carries_both_values(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("p01_vector_add"))
        self.assertRegex(ptx, r"st\.global\.v2\.f32 \[%rd\d+\], \{%f\d+, %f\d+\};")

    def test_two_dimensional_grouped_write_becomes_vector_store(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("m01_matmul_512"))
        self.assertIn("st.global.v2.f32", ptx)
        self.assertNotIn("st.global.f32", ptx)

    def test_grouped_load_becomes_vector_load(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("a01_attention"))
        self.assertIn("ld.global.v2.f32", ptx)
        self.assertRegex(ptx, r"ld\.global\.v2\.f32 \{%f\d+, %f\d+\}, \[%rd\d+\];")

    def test_single_value_write_stays_scalar(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("a01_attention"))
        self.assertIn("st.global.f32", ptx)
        self.assertNotIn("st.global.v1.f32", ptx)

    def test_atomic_grouped_write_stays_scalar(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("m02_matmul_rect"))
        self.assertIn("red.global.add.f32", ptx)
        self.assertNotIn("red.global.add.v2.f32", ptx)

    def test_unaligned_grouped_write_stays_scalar(self):
        mod = load_compiler()
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "case"
            ir = rewrite_write_index(
                ROOT / "problems" / "p01_vector_add",
                dst,
                "right=Literal(value=0, dtype='int')",
                "right=Literal(value=1, dtype='int')",
            )
            ptx = mod.compile_file(ir)
        self.assertNotIn("st.global.v2.f32", ptx)
        self.assertIn("st.global.f32", ptx)

    def test_odd_row_extent_keeps_grouped_write_scalar(self):
        mod = load_compiler()
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "case"
            ir = rewrite_write_index(
                ROOT / "problems" / "p01_vector_add",
                dst,
                "right=Literal(value=0, dtype='int')",
                "right=Literal(value=1, dtype='int')",
            )
            meta = json.loads((dst / "meta.json").read_text())
            for arg in meta["kernels"][0]["args"]:
                if arg["name"] == "add":
                    arg["shape"] = [1023]
            (dst / "meta.json").write_text(json.dumps(meta, indent=1))
            ptx = mod.compile_file(ir)
        self.assertNotIn("st.global.v2.f32", ptx)


class WideVectorTests(unittest.TestCase):
    """TASK.md: a grouped write of 4 values is a v4 vector store."""

    ALIGNED4 = "BinaryExpr(op='*', left=Var(name='a0'), right=Literal(value=4, dtype='int'))"
    UNALIGNED4 = (
        "BinaryExpr(op='+', left=BinaryExpr(op='*', left=Var(name='a0'), "
        "right=Literal(value=4, dtype='int')), right=Literal(value=1, dtype='int'))"
    )

    def test_four_wide_group_is_one_v4_store(self):
        mod = load_compiler()
        ptx = mod.compile_file(_wide_ir(4, self.ALIGNED4, 256))
        self.assertIn("st.global.v4.f32", ptx)
        self.assertEqual(ptx.count("st.global"), 1)
        self.assertNotIn("st.global.f32", ptx)

    def test_four_wide_group_is_one_v4_load(self):
        mod = load_compiler()
        ptx = mod.compile_file(_wide_ir(4, self.ALIGNED4, 256))
        self.assertIn("ld.global.v4.f32", ptx)
        self.assertEqual(ptx.count("ld.global"), 2)

    def test_unaligned_four_wide_group_stays_scalar(self):
        mod = load_compiler()
        ptx = mod.compile_file(_wide_ir(4, self.UNALIGNED4, 256))
        self.assertNotIn("st.global.v4.f32", ptx)
        self.assertIn("st.global.f32", ptx)

    def test_size_not_divisible_by_width_stays_scalar(self):
        mod = load_compiler()
        ptx = mod.compile_file(_wide_ir(4, self.ALIGNED4, 254))
        self.assertNotIn("st.global.v4.f32", ptx)
        self.assertIn("st.global.f32", ptx)


class ContractTests(unittest.TestCase):
    def test_entry_name_comes_from_the_ir(self):
        mod = load_compiler()
        ptx = mod.compile_file(ir_path("r01_array_sum"))
        self.assertIn(".visible .entry k_sum_1_reduce(", ptx)

    def test_parameter_order_comes_from_meta_json(self):
        mod = load_compiler()
        meta = json.loads((ROOT / "problems" / "p01_vector_add" / "meta.json").read_text())
        order = meta["kernels"][0]["arg_order"]
        mod_ = load_compiler()
        ptx = mod_.compile_file(ir_path("p01_vector_add"))
        params = re.findall(r"\.param \.\w+ (p_\w+)", ptx)
        self.assertEqual(params, [f"p_{name}" for name in order])

    def test_output_is_pure_ascii(self):
        mod = load_compiler()
        for problem in sorted((ROOT / "problems").iterdir()):
            if not (problem / "meta.json").exists():
                continue
            ptx = mod.compile_file(ir_path(problem.name))
            ptx.encode("ascii")

    def test_unsupported_statement_fails_loudly(self):
        mod = load_compiler()
        comp = mod.Compiler(ir_path("p01_vector_add"))

        class _CarryStatement:
            pass

        with self.assertRaises(SystemExit):
            comp.stmt(_CarryStatement())


F32_DTYPE = "DataType(name='f32', np=dtype('float32'), nbytes=4)"

BODY_TMPL = (
    "Load(names=('in0__u0',), input='x0', index=({idx},), dtype=" + F32_DTYPE + "), "
    "Assign(name='v0__u0', op=ElementwiseImpl('add'), args=('in0__u0', 'in0__u0'), "
    "dtype=" + F32_DTYPE + "), "
    "Write(output='add', index=({idx},), values=('v0__u0',), value_dtype=" + F32_DTYPE
    + ", atomic=False, swizzle='NONE')"
)


def _body(idx):
    return ", ".join(
        [
            BODY_TMPL.format(idx=idx),
        ]
    )


def _tile(inner, axis_extent=4):
    return (
        "Tile(axes=(Axis(name='a0', extent=Dim(%d), window=None),), body=(%s,))"
        % (axis_extent, inner)
    )


def _wide_body(width, idx, out_len=256):
    """A width-N grouped load/add/write over one buffer."""
    parts = []
    for buf in ("x0", "x1"):
        names = ", ".join("'in%s__u%d'" % (buf, k) for k in range(width)) + ","
        parts.append(
            "Load(names=(%s), input='%s', index=(%s,), dtype=%s)"
            % (names, buf, idx, F32_DTYPE)
        )
    for k in range(width):
        parts.append(
            "Assign(name='v0__u%d', op=ElementwiseImpl('add'), "
            "args=('inx0__u%d', 'inx1__u%d'), dtype=%s)" % (k, k, k, F32_DTYPE)
        )
    values = ", ".join("'v0__u%d'" % k for k in range(width)) + ","
    parts.append(
        "Write(output='add', index=(%s,), values=(%s), value_dtype=%s, atomic=False, swizzle='NONE')"
        % (idx, values, F32_DTYPE)
    )
    return ", ".join(parts)


def _wide_ir(width, index_expr, out_len=256):
    """Build a synthetic package whose 'add' output has `out_len` elements."""
    td = Path(tempfile.mkdtemp())
    dst = td / "case"
    shutil.copytree(ROOT / "problems" / "p01_vector_add", dst, dirs_exist_ok=True)
    ir = next(dst.glob("*.ir.json"))
    doc = json.loads(ir.read_text())
    for node in doc["nodes"].values():
        if node["op"] == "KernelOp":
            node["op_fields"]["body"] = [
                _tile(_wide_body(width, index_expr, out_len), axis_extent=out_len // width)
            ]
        for o in node.get("outputs", []):
            if o["name"] == "add":
                o["shape"] = [out_len]
    ir.write_text(json.dumps(doc, indent=1))
    meta_path = dst / "meta.json"
    meta = json.loads(meta_path.read_text())
    for kern in meta["kernels"]:
        for arg in kern["args"]:
            arg["shape"] = [out_len]
    meta_path.write_text(json.dumps(meta, indent=1))
    return ir


def synthetic_ir(body_str: str) -> Path:
    """Copy p01 into a temp dir with the KernelOp body replaced."""
    td = Path(tempfile.mkdtemp())
    dst = td / "case"
    shutil.copytree(ROOT / "problems" / "p01_vector_add", dst, dirs_exist_ok=True)
    ir = next(dst.glob("*.ir.json"))
    doc = json.loads(ir.read_text())
    for node in doc["nodes"].values():
        if node["op"] == "KernelOp":
            node["op_fields"]["body"] = [body_str]
    ir.write_text(json.dumps(doc, indent=1))
    return ir


class UnrollTests(unittest.TestCase):
    """Unrolling must be honoured AND bounded (a wide loop must stay rolled)."""

    def _loop_body(self, extent):
        inner = (
            "Loop(axis=Axis(name='i', extent=Dim(%d), window=None), body=(%s,), "
            "unroll=True, seed=False)" % (extent, _body("Var(name='i')"))
        )
        return _tile(inner)

    def test_small_loop_is_unrolled(self):
        mod = load_compiler()
        ptx = mod.compile_file(synthetic_ir(self._loop_body(8)))
        self.assertEqual(ptx.count("ld.global"), 8)
        self.assertNotIn("LOOP_", ptx)

    def test_wide_loop_stays_rolled(self):
        mod = load_compiler()
        ptx = mod.compile_file(synthetic_ir(self._loop_body(4096)))
        self.assertEqual(ptx.count("ld.global"), 1)
        self.assertIn("LOOP_", ptx)

    def test_strided_loop_honours_unroll(self):
        mod = load_compiler()
        inner = (
            "StridedLoop(axis=Axis(name='i', extent=Dim(8), window=None), "
            "start=Literal(value=0, dtype='int'), step=Literal(value=1, dtype='int'), "
            "body=(%s,), unroll=True, end=None, seed=False)" % _body("Var(name='i')")
        )
        ptx = mod.compile_file(synthetic_ir(_tile(inner)))
        self.assertEqual(ptx.count("ld.global"), 8)
        self.assertNotIn("SL_", ptx)


class DtypeTests(unittest.TestCase):
    def test_float16_bits_are_half_precision(self):
        mod = load_compiler()
        self.assertEqual(mod.float16_bits(1.0), 0x3C00)
        self.assertEqual(mod.float16_bits(0.5), 0x3800)
        self.assertEqual(mod.float16_bits(-2.0), 0xC000)

    def test_float64_bits(self):
        mod = load_compiler()
        self.assertEqual(mod.float64_bits(1.0), 0x3FF0000000000000)

    def test_f64_setp_uses_a_register_immediate(self):
        """ptxas rejects a 64-bit immediate in setp.ne.f64."""
        mod = load_compiler()
        comp = mod.Compiler(ir_path("p01_vector_add"))
        comp.values["probe"] = comp.emitter.regs.new("d")
        comp.value_dtypes["probe"] = "f64"
        comp.to_pred(mod.Var("probe"))
        emitted = [l.strip() for l in comp.emitter.lines if "setp" in l]
        self.assertEqual(len(emitted), 1, emitted)
        self.assertIn("setp.ne.f64", emitted[0])
        self.assertNotIn("0x0000000000000000", emitted[0])
        self.assertTrue(any("mov.f64" in l for l in comp.emitter.lines), comp.emitter.lines)

    def test_zero_prologue_uses_full_address_register(self):
        """global_addr returns a register name, not a sequence: [0] took one char."""
        from emmy.compiler.ir.stmt.leaves import ZeroPrologue

        mod = load_compiler()
        comp = mod.Compiler(ir_path("p01_vector_add"))
        comp.emit_prologue()
        comp.stmt_zero_prologue(ZeroPrologue(dst="add", words=16))
        stores = [l.strip() for l in comp.emitter.lines if "st.global.u32" in l]
        self.assertTrue(stores, comp.emitter.lines)
        # the address operand must be a whole register, e.g. "[%rd9 + %r3]"
        self.assertRegex(stores[0], r"st\.global\.u32 \[%rd\d+( \+ %r\d+)?\], 0;")


class HiddenSetInsuranceTests(unittest.TestCase):
    """Coverage for IR shapes the open set never exercises but the hidden set may."""

    def _compiler(self):
        mod = load_compiler()
        comp = mod.Compiler(ir_path("p01_vector_add"))
        comp.emit_prologue()
        return mod, comp

    def _regs(self, mod, comp, count=2, dtype="f32"):
        regs = []
        for _ in range(count):
            reg = comp.emitter.regs.new(mod.reg_kind(dtype))
            regs.append(reg)
        return regs

    def test_float_coercion_widens_and_narrows_the_right_way(self):
        """Widening f32->f64 takes no rounding modifier; narrowing needs one."""
        mod, comp = self._compiler()
        src = comp.emitter.regs.new("f")
        wide = comp.coerce(src, "f32", "f64")
        self.assertEqual(comp.emitter.lines[-1],
                         f"        cvt.f64.f32 {wide}, {src};")
        src64 = comp.emitter.regs.new("d")
        narrow = comp.coerce(src64, "f64", "f32")
        self.assertEqual(comp.emitter.lines[-1],
                         f"        cvt.rn.f32.f64 {narrow}, {src64};")

    def test_pow_keeps_the_sign_of_a_negative_base(self):
        """torch.pow accepts a negative base when the exponent is an integer."""
        mod = load_compiler()
        ptx = mod.compile_file(dev_ir_path("z12_pow_neg"))
        self.assertIn("setp.lt.f32", ptx)
        self.assertIn("ex2.approx.f32", ptx)
        self.assertIn("selp.f32", ptx)

    def test_integer_division_by_a_literal_uses_a_shift(self):
        """A power-of-two divisor becomes a shift instead of div.s32."""
        mod, comp = self._compiler()
        src = comp.emitter.regs.new("r")
        comp._sdiv_const(src, 8)
        self.assertIn("shr.s32", comp.emitter.lines[-1])
        comp.emitter.lines.clear()
        comp._srem_const(src, 8)
        self.assertIn("sub.s32", comp.emitter.lines[-1])

    def test_setmaxnreg_is_dropped_because_ptxas_cannot_assemble_it(self):
        """setmaxnreg is a Hopper+ register hint; this runner's ptxas rejects it."""
        mod, comp = self._compiler()

        class _SetMaxNReg:
            count = 40
            direction = "inc"

        before = len(comp.emitter.lines)
        with contextlib.redirect_stderr(io.StringIO()):
            comp.stmt_setmaxnreg(_SetMaxNReg())
        self.assertEqual(len(comp.emitter.lines), before)
        self.assertFalse(any("setmaxnreg" in line for line in comp.emitter.lines))

    def test_float_remainder_is_floor_based(self):
        """PTX has no float remainder: r = a - floor(a / b) * b."""
        mod, comp = self._compiler()
        a, b = self._regs(mod, comp)
        comp.op_on_regs("remainder", (a, b), "f32", arg_dtypes=["f32", "f32"])
        text = "\n".join(comp.emitter.lines)
        self.assertIn("div.rn.f32", text)
        self.assertIn("cvt.rmi.f32.f32", text)
        self.assertIn("mul.f32", text)
        self.assertIn("sub.f32", text)

    def test_float_floor_divide_rounds_down(self):
        mod, comp = self._compiler()
        a, b = self._regs(mod, comp)
        comp.op_on_regs("floor_divide", (a, b), "f32", arg_dtypes=["f32", "f32"])
        text = "\n".join(comp.emitter.lines)
        self.assertIn("div.rn.f32", text)
        self.assertIn("cvt.rmi.f32.f32", text)
        self.assertNotIn("mul.f32", text)

    def test_integer_shifts(self):
        mod, comp = self._compiler()
        a, b = self._regs(mod, comp, dtype="s32")
        comp.op_on_regs("right_shift", (a, b), "s32", arg_dtypes=["s32", "s32"])
        comp.op_on_regs("left_shift", (a, b), "s32", arg_dtypes=["s32", "s32"])
        text = "\n".join(comp.emitter.lines)
        self.assertIn("shr.s32", text)
        self.assertIn("shl.b32", text)

    def test_bitcast_moves_bits_not_values(self):
        mod, comp = self._compiler()
        a, _ = self._regs(mod, comp)
        comp.op_on_regs("bitcast", (a,), "s32", arg_dtypes=["f32"])
        text = "\n".join(comp.emitter.lines)
        self.assertIn("mov.b32", text)
        self.assertNotIn("cvt.", text)

    def test_pad_is_a_plain_copy(self):
        mod, comp = self._compiler()
        a, _ = self._regs(mod, comp)
        comp.op_on_regs("pad", (a,), "f32", arg_dtypes=["f32"])
        text = "\n".join(comp.emitter.lines)
        self.assertIn("mov.f32", text)

    def test_missing_meta_falls_back_to_the_ir_order(self):
        """A kernel shipped without meta.json must still compile."""
        mod = load_compiler()
        src = ir_path("p01_vector_add")
        td = Path(tempfile.mkdtemp())
        dst = td / "lonely"
        shutil.copytree(src.parent, dst, dirs_exist_ok=True)
        (dst / "meta.json").unlink()
        ir = next(dst.glob("*.ir.json"))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            ptx = mod.compile_file(ir)
        self.assertIn("warning: no meta.json found", stderr.getvalue())
        self.assertIn(".param .u64 p_x0", ptx)
        self.assertIn(".param .u64 p_x1", ptx)
        self.assertIn(".param .u64 p_add", ptx)

    def test_segment_indexed_tree_uses_the_inner_slot(self):
        """emmy's segment tree indexes buf[t*scale + iv] and roots at buf[iv].

        The flat tree (inner=None) is the only one the open problems reach,
        so the segment arithmetic is pinned here: the slot must add the lane
        register and the broadcast must read the lane's own slot.
        """
        mod = load_compiler()
        ir = dev_ir_path("z17_segment_tree")
        ptx = mod.compile_file(ir)

        lane = re.search(r"and\.b32 (%r\d+), %r\d+, 31;", ptx)
        self.assertIsNotNone(lane, "the lane register is not computed")
        lane_reg = lane.group(1)
        # the slot is warp*32 + lane, so the lane register must be added to a
        # scaled product: once by the staging write and twice per tree step
        # (the slot and its partner)
        lane_adds = re.findall(rf"add\.s32 %r\d+, %r\d+, {re.escape(lane_reg)};", ptx)
        self.assertGreaterEqual(len(lane_adds), 5, ptx)
        # the broadcast root is the lane's own slot: lane*4 is the address
        self.assertTrue(
            re.search(rf"mul\.lo\.s32 %r\d+, {re.escape(lane_reg)}, 4;", ptx),
            ptx,
        )

    def test_mixed_f32_f64_operands_are_converted(self):
        """emmy stamps the Assign with the promoted dtype, not the operands'.

        An f32 + f64 add has to widen the f32 side, otherwise ptxas sees a
        type-mismatched instruction.
        """
        mod = load_compiler()
        src = ir_path("p01_vector_add")
        td = Path(tempfile.mkdtemp())
        dst = td / "mix"
        shutil.copytree(src.parent, dst, dirs_exist_ok=True)
        ir = next(dst.glob("*.ir.json"))
        doc = json.loads(ir.read_text())
        f32 = "DataType(name='f32', np=dtype('float32'), nbytes=4)"
        f64 = "DataType(name='f64', np=dtype('float64'), nbytes=8)"
        for node in doc["nodes"].values():
            if node["op"] != "KernelOp":
                continue
            body = node["op_fields"]["body"]
            parts = body if isinstance(body, list) else [body]
            out = []
            for part in parts:
                for tag in ("in1__u0", "in1__u1"):
                    start = part.index(f"Load(names=('{tag}',)")
                    marker = "), dtype="
                    end = part.index(marker, start)
                    part = (part[:end] + marker + f64
                            + part[end + len(marker) + len(f32):])
                part = part.replace(
                    f"Assign(name='v0__u0', op=ElementwiseImpl('add'), "
                    f"args=('in0__u0', 'in1__u0'), dtype={f32}",
                    f"Assign(name='v0__u0', op=ElementwiseImpl('add'), "
                    f"args=('in0__u0', 'in1__u0'), dtype={f64}")
                out.append(part)
            node["op_fields"]["body"] = out if isinstance(body, list) else out[0]
        ir.write_text(json.dumps(doc, indent=2))

        ptx = mod.compile_file(ir)
        self.assertIn("cvt.f64.f32", ptx)
        self.assertIn("add.f64", ptx)
        self.assertIn("cvt.rn.f32.f64", ptx)
        # the f64 result is narrowed again for the f32 destination
        self.assertIn("st.global.f32", ptx)
        self.assertNotIn("st.global.f64", ptx)
        for line in ptx.splitlines():
            self.assertNotIn("('", line)

    def test_f64_reduction_shuffles_both_halves_in_registers(self):
        """ptxas rejects mov.b32 from a .f64 register, so the value is split.

        The two 32-bit halves travel through the register shuffle: a shared
        memory bounce would race between the warps of one block.
        """
        mod = load_compiler()
        src = ir_path("r01_array_sum")
        td = Path(tempfile.mkdtemp())
        dst = td / "f64"
        shutil.copytree(src.parent, dst, dirs_exist_ok=True)
        ir = next(dst.glob("*.ir.json"))
        doc = json.loads(ir.read_text())
        for node in doc["nodes"].values():
            if node["op"] != "KernelOp":
                continue
            body = node["op_fields"]["body"]
            parts = body if isinstance(body, list) else [body]
            parts = [
                p.replace(
                    "DataType(name='f32', np=dtype('float32'), nbytes=4)",
                    "DataType(name='f64', np=dtype('float64'), nbytes=8)",
                ).replace("dtype='float'", "dtype='double'")
                for p in parts
            ]
            node["op_fields"]["body"] = parts if isinstance(body, list) else parts[0]
        ir.write_text(json.dumps(doc, indent=2))
        ptx = mod.compile_file(ir)
        self.assertIn("mov.b64", ptx)
        self.assertIn("cvt.u32.u64", ptx)
        self.assertIn("shr.u64", ptx)
        self.assertEqual(ptx.count("shfl.sync.bfly.b32"), 10)
        self.assertNotIn("bar.warp.sync 0;", ptx)
        self.assertNotIn("f64_shfl", ptx)


if __name__ == "__main__":
    unittest.main()
