"""CPU checks for files/patch_qsa_fp8_kv_v030.py (vllm#55557 backport onto vLLM 0.30.0).

QSA_V030_SRC points at the v0.30.0 models/qwen4_exp/nvidia directory. Inside the image:

    docker run --name fp8kv-test --network none --entrypoint python3 -v "$PWD:/r" -w /r \\
        -e QSA_V030_SRC=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia \\
        vllm/vllm-openai:v0.30.0 tests/test_qsa_fp8_kv_v030.py
"""
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "files", "patch_qsa_fp8_kv_v030.py")
SRC = os.environ.get(
    "QSA_V030_SRC",
    "/tmp/claude-1000/-home-jvr0x/a5ecbb00-2895-4187-83fa-bb98b246ccc2/scratchpad"
    "/v030src/vllm_pkg/models/qwen4_exp/nvidia",
)
HAVE_SRC = all(os.path.isfile(os.path.join(SRC, p)) for p in ("qsa.py", "ops/qsa.py"))
LOCAL = ("qsa.py", "ops/qsa.py")


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, SCRIPT, *args], capture_output=True, text=True
    )


def read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


@unittest.skipUnless(HAVE_SRC, f"v0.30.0 QSA sources not found under {SRC}, set QSA_V030_SRC")
class Backport(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.orig = os.path.join(self.tmp.name, "orig")
        self.out = os.path.join(self.tmp.name, "out")
        os.makedirs(os.path.join(self.orig, "ops"))
        for local in LOCAL:
            shutil.copyfile(os.path.join(SRC, local), os.path.join(self.orig, local))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def patch(self, orig: str | None = None, out: str | None = None) -> subprocess.CompletedProcess:
        return run(orig or self.orig, out or self.out)

    def test_applies_to_v030(self) -> None:
        result = self.patch()
        self.assertEqual(result.returncode, 0, result.stderr)
        nvidia = read(os.path.join(self.out, "qsa.py"))
        ops = read(os.path.join(self.out, "ops/qsa.py"))
        for text in (nvidia, ops):
            ast.parse(text)
        self.assertIn('"Qwen4Exp QSA requires a BF16 main KV cache"', read(os.path.join(self.orig, "qsa.py")))
        self.assertNotIn('"Qwen4Exp QSA requires a BF16 main KV cache"', nvidia)
        self.assertIn('"Qwen4Exp QSA requires a BF16 or FP8-e4m3 main KV cache"', nvidia)
        self.assertIn('"fp8_e4m3",\n    ]', nvidia)
        self.assertIn("key_cache = key_cache.view(torch.float8_e4m3fn)", nvidia)
        self.assertIn("k_scale = layer._k_scale_float", nvidia)
        self.assertIn("(torch.bfloat16, torch.uint8)", nvidia)
        self.assertIn("IS_FP8: tl.constexpr", ops)
        self.assertIn("softmax_scale = (head_dim**-0.5) * float(k_scale)", ops)
        self.assertIn("is_fp8 = kv_cache.dtype == torch.uint8", ops)
        self.assertNotIn("assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16", ops)

    def test_idempotent(self) -> None:
        second = os.path.join(self.tmp.name, "second")
        self.assertEqual(self.patch().returncode, 0)
        self.assertEqual(self.patch(out=second).returncode, 0)
        for local in LOCAL:
            self.assertEqual(read(os.path.join(self.out, local)), read(os.path.join(second, local)))

    def test_moved_context_fails_and_writes_nothing(self) -> None:
        path = os.path.join(self.orig, "ops/qsa.py")
        text = read(path)
        first = "    num_kv_heads = key_cache.shape[2]\n"
        second = "    group_size = num_query_heads // num_kv_heads\n"
        self.assertEqual(text.count(first + second), 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(first + second, second + first))
        result = self.patch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not apply", result.stderr)
        self.assertFalse(os.path.exists(self.out))

    def test_already_patched_passes_through(self) -> None:
        self.assertEqual(self.patch().returncode, 0)
        again = os.path.join(self.tmp.name, "again")
        result = self.patch(orig=self.out, out=again)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already present", result.stdout)
        for local in LOCAL:
            self.assertEqual(read(os.path.join(self.out, local)), read(os.path.join(again, local)))


class Cli(unittest.TestCase):
    def test_list(self) -> None:
        result = run("--list")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            result.stdout.splitlines(),
            ["models/qwen4_exp/nvidia/qsa.py", "models/qwen4_exp/nvidia/ops/qsa.py"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
