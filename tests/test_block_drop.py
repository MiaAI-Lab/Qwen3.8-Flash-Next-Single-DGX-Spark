"""CPU checks for the opt-in vllm#53388 block-drop backport and its start.sh wiring.

- files/patch_block_drop.py (MTP_DISABLE_BLOCK_DROP=1): the vllm#53388 anchors
  apply to the six pinned sources, the KV connectors use the new check, the
  output is the same on each run, and a failed anchor writes no files.
- start.sh: the knob section and the docker run text, cut out of start.sh and
  run in bash with a stub extract(), give no change with the knob off and the
  expected env and mounts with the knob on.

Most of BlockDropPatch and one StartShWiring test read the vLLM sources of the
pinned image (they carry @NEED_SRC below); the rest of StartShWiring needs no
image at all. Inside the image the sources are found automatically:

    docker run --rm --network none --entrypoint python3 -v "$PWD:/r" -w /r \\
        vllm/vllm-openai:qwen38-flash-next tests/test_block_drop.py

(Run the file, not "-m unittest tests/...": the image has its own "tests" package.)

On the host, set VLLM_SRC to a copy of the image's vllm package directory:

    VLLM_SRC=/path/to/vllm python3 tests/test_block_drop.py

Without VLLM_SRC, run only the tests that do not need it:

    python3 -m unittest tests.test_block_drop.StartShWiring.test_both_off_changes_nothing \\
        tests.test_block_drop.StartShWiring.test_block_drop_needs_mtp \\
        tests.test_block_drop.StartShWiring.test_bad_value_stops_the_launch \\
        tests.test_block_drop.StartShWiring.test_knob_honours_environment_over_env_file
"""
import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

REPO = Path(__file__).resolve().parent.parent
FILES = REPO / "files"
PKG = "/usr/local/lib/python3.12/dist-packages/vllm"
SRC = Path(os.environ.get("VLLM_SRC", PKG))
# The six vllm files that vllm#53388 changes, less single_type_kv_cache_manager.py
# (sliding-window cache hits; the QSA attention of this model has no sliding window).
BLOCK_DROP_RELS = (
    "config/speculative.py",
    "v1/core/kv_cache_utils.py",
    "v1/core/sched/scheduler.py",
    "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py",
    "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
    "v1/simple_kv_offload/manager.py",
)
CONNECTOR_RELS = BLOCK_DROP_RELS[3:]
HAVE_SRC = all((SRC / r).is_file() for r in BLOCK_DROP_RELS)
NEED_SRC = unittest.skipUnless(HAVE_SRC, f"no pinned vLLM sources at {SRC}; set VLLM_SRC")


def run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


def load_functions(path: Path, names, ns, cls=None):
    """Exec the named module-level functions (or methods of cls) of path into ns."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names), [n.name for n in nodes]
    for n in nodes:
        n.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


@NEED_SRC
class BlockDropPatch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.orig, self.out = self.tmp / "orig", self.tmp / "out"
        self.orig.mkdir()
        for rel in BLOCK_DROP_RELS:
            (self.orig / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(SRC / rel, self.orig / rel)

    def patch(self):
        return run([sys.executable, FILES / "patch_block_drop.py", self.orig, self.out])

    def text(self, rel):
        return (self.out / rel).read_text()

    def outputs(self):
        if not self.out.exists():
            return {}
        return {str(p.relative_to(self.out)): p.read_text() for p in self.out.rglob("*") if p.is_file()}

    def test_list_names_the_files(self):
        out = run([sys.executable, FILES / "patch_block_drop.py", "--list"])
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(tuple(out.stdout.split()), BLOCK_DROP_RELS)

    def test_applies_to_pinned_sources(self):
        out = self.patch()
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(sorted(self.outputs()), sorted(BLOCK_DROP_RELS))
        for rel in BLOCK_DROP_RELS:
            ast.parse(self.text(rel))
        spec = self.text("config/speculative.py")
        self.assertIn("    disable_eagle_block_drop: bool = False\n", spec)
        self.assertIn("    def use_eagle_block_drop(self) -> bool:\n", spec)
        kv = self.text("v1/core/kv_cache_utils.py")
        self.assertIn("not spec_config.use_eagle_block_drop():", kv)
        self.assertNotIn("not spec_config.use_eagle():", kv)
        sched = self.text("v1/core/sched/scheduler.py")
        self.assertIn("use_eagle=self.use_eagle_block_drop,", sched)
        self.assertIn("        if self.use_eagle_block_drop:\n"
                      "            last_cache_position = max(last_cache_position - block_size, 0)\n", sched)
        self.assertIn("block dropping is disabled", sched)
        self.assertRegex(sched, r"(?m)^logger = init_logger\(__name__\)$")
        # The prefill lookahead still follows use_eagle: the drafter still runs.
        self.assertIn("            if self.use_eagle:\n                self.num_prefill_lookahead", sched)

    def test_connectors_follow_the_new_check(self):
        # A KV connector that still drops the block while the scheduler keeps it
        # would disagree with the scheduler on the cached length.
        self.assertEqual(self.patch().returncode, 0)
        for rel in CONNECTOR_RELS:
            src = self.text(rel)
            self.assertIn(".use_eagle_block_drop()", src, rel)
            self.assertNotIn(".use_eagle()", src, rel)
            # The pinned file has one call of the old check, and the patch replaces it.
            self.assertEqual((self.orig / rel).read_text().count(".use_eagle()"), 1, rel)

    def test_use_eagle_block_drop_logic(self):
        self.assertEqual(self.patch().returncode, 0)
        ns = load_functions(self.out / "config/speculative.py", ["use_eagle", "use_eagle_block_drop"], {},
                            cls="SpeculativeConfig")
        for method, disable, want in (("mtp", False, True), ("mtp", True, False),
                                      ("eagle3", False, True), ("ngram", False, False),
                                      ("ngram", True, False)):
            cfg = types.SimpleNamespace(method=method, disable_eagle_block_drop=disable)
            cfg.use_eagle = types.MethodType(ns["use_eagle"], cfg)
            self.assertIs(ns["use_eagle_block_drop"](cfg), want, (method, disable))

    def test_output_is_the_same_on_each_run(self):
        self.assertEqual(self.patch().returncode, 0)
        first = self.outputs()
        self.assertEqual(self.patch().returncode, 0)
        self.assertEqual(self.outputs(), first)

    def test_image_with_the_option_mounts_nothing(self):
        self.assertEqual(self.patch().returncode, 0)
        for rel in BLOCK_DROP_RELS:
            shutil.copy(self.out / rel, self.orig / rel)
        out = self.patch()
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("nothing to mount", out.stdout)
        self.assertEqual(self.outputs(), {})

    def test_failed_anchor_writes_nothing(self):
        # The last file fails, so the first five must not be written either.
        mgr = self.orig / "v1/simple_kv_offload/manager.py"
        mgr.write_text(mgr.read_text().replace("spec_config.use_eagle()", "spec_config.use_eagle_x()"))
        out = self.patch()
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("v1/simple_kv_offload/manager.py: anchor", out.stderr)
        self.assertEqual(self.outputs(), {})


def section(text, start, end_count=1):
    """Lines from the first line matching start through the end_count-th "fi" line after it."""
    lines = text.splitlines()
    i = next(n for n, line in enumerate(lines) if re.match(start, line))
    j, seen = i, 0
    while seen < end_count:
        j += 1
        if lines[j] == "fi":
            seen += 1
    return "\n".join(lines[i:j + 1])


def section_to(text, start, end):
    """Lines from the first line matching start through the first line at/after it matching end."""
    lines = text.splitlines()
    i = next(n for n, line in enumerate(lines) if re.match(start, line))
    j = next(n for n in range(i, len(lines)) if re.match(end, lines[n]))
    return "\n".join(lines[i:j + 1])


START = (REPO / "start.sh").read_text()
KNOBS = section_to(START, r'^MTP_DISABLE_BLOCK_DROP="\$\{MTP_DISABLE_BLOCK_DROP:-0\}"$',
                    r'^    \|\| err "MTP_DISABLE_BLOCK_DROP must be 0 or 1"$')
STEP4 = section(START, r'^BLOCK_DROP_MOUNTS=""$')
SPEC = section(START, r'^    _SPEC_ARGMAX=""$')
SPEC = 'if [[ "$MTP_NUM_SPECULATIVE_TOKENS" -gt 0 ]]; then\n' + SPEC
LAUNCH = re.search(r'(?ms)^cat > "\$LAUNCH_SCRIPT" <<LAUNCH_EOF$.*?^LAUNCH_EOF$', START).group(0)

HARNESS = r'''
set -euo pipefail
err()  { echo "ERR: $*" >&2; exit 3; }
warn() { echo "WARN: $*" >&2; }
info() { :; }
extract() { [[ -f "$2" ]] || cp "$VLLM_SRC/${1#"$VLLM_PKG"/}" "$2"; }
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
MTP_PKG="$VLLM_PKG/models/qwen3_8_flash_next/nvidia/mtp.py"
PATCHED_MTP="$SCRIPT_DIR/files/mtp_patched.py"
VLLM_ARGS=()
MTP_K_SCHEDULE=""
%(knobs)s
%(step4)s
%(spec)s
echo "SPEC=${VLLM_ARGS[*]:-}"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch.sh"
set +u
VLLM_ARGS_STR="${VLLM_ARGS[*]:-}"
%(launch)s
cat "$LAUNCH_SCRIPT"
'''


class StartShWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        files = self.tmp / "files"
        files.mkdir()
        shutil.copy(FILES / "patch_block_drop.py", files / "patch_block_drop.py")
        self.script = self.tmp / "harness.sh"
        self.script.write_text(HARNESS % dict(knobs=KNOBS, step4=STEP4, spec=SPEC, launch=LAUNCH))

    def launch(self, mtp="3", vocab="files/draft_vocab_en_code_47k.txt", **knobs):
        env = dict(os.environ, SCRIPT_DIR=str(self.tmp), VLLM_SRC=str(SRC),
                   MTP_NUM_SPECULATIVE_TOKENS=mtp, MTP_DRAFT_VOCAB=vocab, **knobs)
        return run(["bash", self.script], env=env)

    def mounts(self, text):
        return re.findall(r"-v (\S+):(\S+):ro", text)

    def test_both_off_changes_nothing(self):
        # The knob defaults to 0: no VLLM_SRC needed, the mount block never runs.
        out = self.launch()
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn("block_drop", out.stdout)
        self.assertNotIn("disable_eagle_block_drop", out.stdout)
        self.assertFalse((self.tmp / "files" / "block_drop").exists())

    @NEED_SRC
    def test_block_drop_on_mounts_the_backport(self):
        out = self.launch(MTP_DISABLE_BLOCK_DROP="1")
        self.assertEqual(out.returncode, 0, out.stderr)
        bd = self.tmp / "files" / "block_drop"
        want = [(str(bd / rel), f"{PKG}/{rel}") for rel in BLOCK_DROP_RELS]
        got = [m for m in self.mounts(out.stdout) if "block_drop" in m[0]]
        self.assertEqual(got, want)
        # Two files share the name scheduler.py: each mount needs its own host file.
        self.assertEqual(len({host for host, _ in got}), len(BLOCK_DROP_RELS))
        for host, _ in got:
            self.assertIn("use_eagle_block_drop", Path(host).read_text())
        # The key and the backport go together: the image rejects the key alone.
        self.assertIn('"disable_eagle_block_drop":true', out.stdout)

    def test_block_drop_needs_mtp(self):
        # MTP off: no VLLM_SRC needed, the mount block never runs.
        out = self.launch(mtp="0", MTP_DISABLE_BLOCK_DROP="1")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertNotIn("block_drop", out.stdout)
        self.assertFalse((self.tmp / "files" / "block_drop").exists())

    def test_bad_value_stops_the_launch(self):
        # Validation runs before the mount block: no VLLM_SRC needed.
        out = self.launch(MTP_DISABLE_BLOCK_DROP="true")
        self.assertEqual(out.returncode, 3, out.stderr)
        self.assertIn("must be 0 or 1", out.stderr)

    def test_knob_honours_environment_over_env_file(self):
        snap = re.search(r"(?s)_ENV_SNAPSHOT_VARS=\((.*?)\)", START).group(1).split()
        self.assertIn("MTP_DISABLE_BLOCK_DROP", snap)


if __name__ == "__main__":
    unittest.main(verbosity=2)
