# Owner(s): ["module: cuda"]

import json
from hashlib import sha256
from unittest import mock

from torch._inductor.runtime._cudagraph._sdk import activate


activate()

from cutlass._mlir import ir

from torch._inductor.runtime._cudagraph._compiler.cfg_values import read_cfg_function
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.artifact import (
    _Payload,
    ARTIFACT_VERSION,
    ArtifactSite,
    BinaryImage,
    Consumer,
    FieldSource,
    Formal,
    IntegerField,
    Leaf,
    NodeFields,
    Parameter,
    ParameterExpression,
    PointerField,
    Registration,
    Stream,
)
from torch._inductor.runtime._cudagraph._compiler.cudagraph_cute_runtime.persistence import (
    dump_payload,
    load_payload,
)
from torch._inductor.runtime._cudagraph._compiler.decoded_values import (
    prepare_decodings,
)
from torch._inductor.runtime._cudagraph._compiler.overflow_properties import (
    bind_properties,
)
from torch._inductor.runtime._cudagraph._compiler.owned_numeric import (
    dump_numeric,
    evaluate_owned,
    freeze_numeric,
    load_numeric,
)
from torch._inductor.runtime._cudagraph._compiler.values import scalar
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def numeric(source, order=()):
    with ir.Context(), ir.Location.unknown(), ir.raw_values():
        module = ir.Module.parse(source)
        cfg = read_cfg_function(module, "probe")
        return freeze_numeric(bind_properties(cfg), prepare_decodings(cfg), order)


def literal(typ, value):
    return numeric(f"""module {{ llvm.func @probe() -> {typ} {{
        %value = llvm.mlir.constant({value} : {typ}) : {typ}
        llvm.return %value : {typ}
    }} }}""")


def payload_fixture():
    tensor = Formal(
        5,
        0,
        0,
        "x",
        (),
        "Tensor",
        "tensor",
        "!llvm.ptr",
        (),
        (),
        None,
        None,
        None,
        None,
        None,
        8,
        8,
        (Leaf((), "!llvm.ptr", "pointer", (), 0, 8, 8),),
        (),
    )
    integer = Formal(
        9,
        1,
        1,
        "n",
        (),
        "Var",
        "i64",
        "i64",
        (),
        (),
        None,
        None,
        None,
        None,
        None,
        8,
        8,
        (Leaf((), "i64", "value", (), 0, 8, 8),),
        (),
    )
    stream = Formal(
        12,
        2,
        2,
        "stream",
        (),
        "EnvStream",
        "!llvm.ptr",
        "!llvm.ptr",
        (),
        (),
        None,
        None,
        None,
        None,
        None,
        8,
        8,
        (),
        (),
    )
    image = BinaryImage(
        0, "cubin", sha256(b"compiler image").hexdigest(), b"compiler image"
    )
    registration = Registration("kernel", "handle", 0, image.global_name, image.sha256)
    leaf = ParameterExpression("argument", "i64", 9, (), None, (), ())
    distinct = ParameterExpression(*tuple(leaf))
    shared = ParameterExpression("llvm.add", "i64", None, (), None, (leaf, leaf), ())
    expression = ParameterExpression(
        "llvm.add", "i64", None, (), None, (shared, distinct), ()
    )
    pointer = FieldSource("tensor_property", 5, "x", (), "pointer", (), "!llvm.ptr", ())
    computed = FieldSource(
        "compiler_expression",
        None,
        None,
        (),
        "value",
        (),
        "i64",
        (),
        expression=expression,
    )
    fields = NodeFields(
        0,
        "kernel",
        (8, 8),
        (PointerField(0, 0, pointer),),
        (IntegerField(1, 0, "i64", computed),),
        (),
        (),
    )
    consumers = []
    i32, i64 = literal("i32", 1), literal("i64", 0)
    for role, index, typ in [
        (role, axis, "i32") for role in ("grid", "block") for axis in range(3)
    ] + [("shared", 0, "i64"), ("kernel_smem", 0, "i64")]:
        consumers.append(
            Consumer(
                len(consumers),
                f"{role}{index}",
                0,
                role,
                index,
                typ,
                (),
                i32 if typ == "i32" else i64,
            )
        )
    site = ArtifactSite(
        0,
        None,
        0,
        ("module", "kernel"),
        registration,
        (Parameter(0, "!llvm.ptr", 5, 0, 8, 8), Parameter(1, "i64", None, None, 8, 8)),
        fields,
        tuple(range(8)),
        (),
        12,
        (),
    )
    digest = sha256(b"compiler source").hexdigest()
    return _Payload(
        ARTIFACT_VERSION,
        "host",
        "sm_103a",
        "aarch64",
        digest,
        digest,
        digest,
        digest,
        ("!llvm.ptr", "i64", "!llvm.ptr"),
        (),
        ((5, 0), (9, 1), (12, 2)),
        (tensor, integer, stream),
        Stream(12, 2, 2, "!llvm.ptr", 8, 8),
        (image,),
        tuple(consumers),
        (site,),
    )


@instantiate_parametrized_tests
class TestCuTePersistence(TestCase):
    def test_numeric_preserves_properties_order_and_evaluation(self):
        program = numeric(
            """module { llvm.func @probe(%x: i64, %y: i64) -> i64 {
          %v = llvm.add %x, %y overflow<nsw> : i64
          llvm.return %v : i64
        } }""",
            (7, 2),
        )
        with mock.patch.object(
            ir.Module,
            "parse",
            side_effect=AssertionError("Transport reparsed compiler IR"),
        ):
            restored = load_numeric(dump_numeric(program))
        self.assertEqual(restored.source_order, (7, 2))
        self.assertEqual(restored._flags, program._flags)
        self.assertEqual(restored._predicates, program._predicates)
        self.assertEqual(
            evaluate_owned(restored, (scalar("i64", 17), scalar("i64", 9)))[
                0
            ].integer(),
            26,
        )
        with self.assertRaises(ValueError):
            evaluate_owned(restored, (scalar("i64", (1 << 63) - 1), scalar("i64", 1)))

    @parametrize(
        "typ,value", (("f32", "-0.0"), ("f64", "0x7FF8000000000042"), ("i64", "-1"))
    )
    def test_exact_constant_bytes(self, typ, value):
        program = literal(typ, value)
        restored = load_numeric(dump_numeric(program))
        before = program._cfg.blocks[0].instructions[0].expression.constant
        after = restored._cfg.blocks[0].instructions[0].expression.constant
        self.assertEqual(before, after)
        self.assertEqual(dump_numeric(restored), dump_numeric(program))

    def test_multiblock_numeric_keeps_control_flow(self):
        program = numeric(
            """module { llvm.func @probe(%choose: i1, %x: i64, %y: i64) -> i64 {
          llvm.cond_br %choose, ^left, ^right
        ^left:
          llvm.br ^merge(%x : i64)
        ^right:
          llvm.br ^merge(%y : i64)
        ^merge(%value: i64):
          llvm.return %value : i64
        } }""",
            (9, 3, 1),
        )
        restored = load_numeric(dump_numeric(program))
        for choose, expected in ((0, 23), (1, 17)):
            self.assertEqual(
                evaluate_owned(
                    restored,
                    (scalar("i1", choose), scalar("i64", 17), scalar("i64", 23)),
                )[0].integer(),
                expected,
            )

    @parametrize(
        "change",
        (
            "version",
            "forward_reference",
            "unknown_tag",
            "source_order",
            "flags",
            "flag_bits",
            "constant_width",
            "instruction",
        ),
    )
    def test_malformed_numeric_rejected(self, change):
        program = numeric(
            """module { llvm.func @probe(%x: i64, %y: i64) -> i64 {
          %one = llvm.mlir.constant(1 : i64) : i64
          %v = llvm.add %x, %one overflow<nsw> : i64
          llvm.return %v : i64
        } }""",
            (7, 2),
        )
        data = json.loads(dump_numeric(program))
        root = data["nodes"][data["root"][1]][1]
        if change == "version":
            data["version"] += 1
        elif change == "forward_reference":
            data["nodes"][0] = ["tuple", [["ref", 0]]]
        elif change == "unknown_tag":
            data["nodes"][0][0] = "Executable"
        elif change == "source_order":
            data["nodes"][root[3][1]][1] = [7, 7]
        elif change == "flags":
            data["nodes"][root[1][1]][1] = []
        elif change == "flag_bits":
            (flag_ref,) = data["nodes"][root[1][1]][1]
            data["nodes"][flag_ref[1]][1][1] = 1024
        elif change == "constant_width":
            flow = next(
                row
                for row in data["nodes"]
                if row[0] == "_Flow" and row[1][0] == "constant"
            )
            data["nodes"][flow[1][6][1]][1][1] = ["bytes", "00"]
        else:
            flow = next(
                row
                for row in data["nodes"]
                if row[0] == "_Flow" and row[1][0] == "llvm.add"
            )
            flow[1][0] = "llvm.call"
        with self.assertRaises(ValueError):
            load_numeric(json.dumps(data).encode())

    def test_payload_keeps_shared_and_distinct_records(self):
        payload = payload_fixture()
        restored = load_payload(dump_payload(payload))
        expression = restored.sites[0].fields.integers[0].source.expression
        shared, distinct = expression.operands
        self.assertIs(shared.operands[0], shared.operands[1])
        self.assertIsNot(shared.operands[0], distinct)
        self.assertEqual(shared.operands[0], distinct)
        self.assertIs(restored.consumers[0].numeric, restored.consumers[1].numeric)
        self.assertEqual(restored.binaries, payload.binaries)
        self.assertEqual(restored.operand_bindings, payload.operand_bindings)
        self.assertEqual(dump_payload(restored), dump_payload(payload))

    def test_declared_f32_field_keeps_its_physical_type(self):
        payload = payload_fixture()
        formal = payload.formals[1]._replace(
            source_type="f32",
            llvm_type="f32",
            size=4,
            alignment=4,
            leaves=(Leaf((), "f32", "value", (), 0, 4, 4),),
        )
        source = FieldSource("scalar_formal", 9, "n", (), "value", (), "f32", ())
        site = payload.sites[0]
        site = site._replace(
            parameters=(site.parameters[0], Parameter(1, "f32", 9, 1, 4, 4)),
            fields=site.fields._replace(
                parameter_sizes=(8, 4), integers=(IntegerField(1, 0, "f32", source),)
            ),
        )
        payload = payload._replace(
            host_types=("!llvm.ptr", "f32", "!llvm.ptr"),
            formals=(payload.formals[0], formal, payload.formals[2]),
            sites=(site,),
        )
        restored = load_payload(dump_payload(payload))
        self.assertEqual(restored.formals[1], formal)
        self.assertEqual(
            restored.sites[0].fields.integers[0], IntegerField(1, 0, "f32", source)
        )
        self.assertEqual(restored.sites[0].fields.parameter_sizes, (8, 4))
        data = json.loads(dump_payload(payload))
        field = next(row[1] for row in data["nodes"] if row[0] == "IntegerField")
        field[2] = "i32"
        with self.assertRaisesRegex(ValueError, "inconsistent native scalar"):
            load_payload(json.dumps(data).encode())

    def test_repeated_compiled_launch_index_is_preserved(self):
        payload = payload_fixture()
        original = payload.sites[0]
        extra = tuple(
            item._replace(consumer_id=item.consumer_id + 8, site_id=1)
            for item in payload.consumers
        )
        site = original._replace(site_id=1, consumer_ids=tuple(range(8, 16)))
        payload = payload._replace(
            sites=(original, site), consumers=(*payload.consumers, *extra)
        )
        restored = load_payload(dump_payload(payload))
        self.assertEqual(tuple(item.launch_index for item in restored.sites), (0, 0))
        self.assertIs(restored.sites[0].fields, restored.sites[1].fields)

    @parametrize(
        "change",
        ("image", "registration", "formal", "field", "consumer", "source_mapping"),
    )
    def test_payload_association_mismatch_rejected(self, change):
        data = json.loads(dump_payload(payload_fixture()))
        tag = {
            "image": "BinaryImage",
            "registration": "Registration",
            "formal": "Formal",
            "field": "IntegerField",
            "consumer": "Consumer",
            "source_mapping": "Consumer",
        }[change]
        row = next(row for row in data["nodes"] if row[0] == tag)[1]
        if change == "image":
            row[3] = ["bytes", b"different image".hex()]
        elif change == "registration":
            row[4] = "0" * 64
        elif change == "formal":
            row[1] = 8
        elif change == "field":
            row[1] = 1
        elif change == "consumer":
            row[3] = "unknown"
        else:
            data["nodes"][row[6][1]][1] = [5]
        with self.assertRaises(ValueError):
            load_payload(json.dumps(data).encode())


if __name__ == "__main__":
    run_tests()
