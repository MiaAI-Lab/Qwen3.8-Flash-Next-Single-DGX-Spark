"""CPU checks for FAST_LOAD: files/patch_default_loader.py and its start.sh wiring.

- files/patch_default_loader.py: the three anchors apply once to the pinned
  default_loader.py, the output compiles, the same input gives the same output,
  and a failed anchor exits non-zero without writing the output.
- The shard selection the patch uses: the drafter keys pick exactly the shards
  holding mtp.* / embed_tokens / lm_head, the lazy keys pick the PLE shards, and
  a missing index selects nothing.
- start.sh: FAST_LOAD accepts only 0/1; FAST_LOAD=0 leaves the default lane's
  env and mounts exactly as they were, and FAST_LOAD=1 adds -e VLLM_FAST_LOAD=1
  and the loader mount on the default lane only (tests/test_v030_lane.py
  separately pins the docker run template itself).

GeneratorOnPinnedSource reads the pinned image's default_loader.py (it carries
@NEED_SRC); the rest needs no image. Inside the image the source is found
automatically:

    docker run --rm --network none --entrypoint python3 -v "$PWD:/r" -w /r \\
        vllm/vllm-openai:qwen38-flash-next tests/test_default_loader.py

On the host, set VLLM_SRC to a copy of the image's vllm package directory, or
run without it to skip GeneratorOnPinnedSource:

    python3 tests/test_default_loader.py
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import re
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
FILES = REPO / "files"
GENERATOR = FILES / "patch_default_loader.py"
PKG = "/usr/local/lib/python3.12/dist-packages/vllm"
SRC = Path(os.environ.get("VLLM_SRC", PKG))
LOADER_REL = "model_executor/model_loader/default_loader.py"
HAVE_SRC = (SRC / LOADER_REL).is_file()
NEED_SRC = unittest.skipUnless(HAVE_SRC, f"no pinned vLLM sources at {SRC}; set VLLM_SRC")


def generator_module():
    spec = importlib.util.spec_from_file_location("patch_default_loader", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_generator(orig_text: str):
    """Run a copy of the generator next to orig_text; return (rc, output or None, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(GENERATOR, tmp)
        Path(tmp, "default_loader_patched.py.orig").write_text(orig_text)
        proc = subprocess.run(
            ["python3", str(Path(tmp, GENERATOR.name))], capture_output=True, text=True
        )
        out = Path(tmp, "default_loader_patched.py")
        return proc.returncode, (out.read_text() if out.exists() else None), proc.stderr


@NEED_SRC
class GeneratorOnPinnedSource(unittest.TestCase):
    def setUp(self):
        self.orig = (SRC / LOADER_REL).read_text()

    def test_applies_and_compiles(self):
        rc, out, err = run_generator(self.orig)
        self.assertEqual(rc, 0, err)
        self.assertIn('os.environ.get("VLLM_FAST_LOAD") == "1"', out)
        self.assertIn("_fast_load_bulk_iterator", out)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(out)
        try:
            py_compile.compile(f.name, doraise=True)
        finally:
            os.unlink(f.name)

    def test_same_input_same_output(self):
        self.assertEqual(run_generator(self.orig)[1], run_generator(self.orig)[1])

    def test_failed_anchor_writes_nothing(self):
        broken = self.orig.replace("class DefaultModelLoader(BaseModelLoader):", "class X:")
        rc, out, err = run_generator(broken)
        self.assertNotEqual(rc, 0)
        self.assertIsNone(out)
        self.assertIn("anchor", err)


class ShardSelection(unittest.TestCase):
    """_files_with_keys from the generator's HELPERS text, run on a fake index."""

    @classmethod
    def setUpClass(cls):
        mod = generator_module()
        tree = ast.parse(mod.HELPERS)
        ns = {"os": os, "SAFE_WEIGHTS_INDEX_NAME": "model.safetensors.index.json"}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_files_with_keys":
                exec(compile(ast.Module(body=[node], type_ignores=[]), "HELPERS", "exec"), ns)
            elif isinstance(node, ast.Assign) and node.targets[0].id in ("_FAST_LOAD_DRAFT_KEYS", "_FAST_LOAD_LAZY_KEYS"):
                exec(compile(ast.Module(body=[node], type_ignores=[]), "HELPERS", "exec"), ns)
        cls.ns = ns

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.files = [os.path.join(self.tmp, f"model-0000{i}-of-00004.safetensors") for i in range(1, 5)]
        weight_map = {
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": "model-00001-of-00004.safetensors",
            "model.language_model.layers.3.mlp.experts.0.down_proj.weight": "model-00002-of-00004.safetensors",
            "model.language_model.embed_tokens.weight": "model-00003-of-00004.safetensors",
            "lm_head.weight": "model-00003-of-00004.safetensors",
            "mtp.layers.0.mlp.experts.0.down_proj.weight": "model-00004-of-00004.safetensors",
        }
        with open(os.path.join(self.tmp, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": weight_map}, f)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def pick(self, keys):
        return sorted(os.path.basename(p) for p in self.ns["_files_with_keys"](self.tmp, self.files, keys))

    def test_drafter_keys(self):
        self.assertEqual(
            self.pick(self.ns["_FAST_LOAD_DRAFT_KEYS"]),
            ["model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors"],
        )

    def test_lazy_keys_pick_ple_shards(self):
        self.assertEqual(self.pick(self.ns["_FAST_LOAD_LAZY_KEYS"]), ["model-00001-of-00004.safetensors"])

    def test_missing_index_selects_nothing(self):
        os.unlink(os.path.join(self.tmp, "model.safetensors.index.json"))
        self.assertEqual(self.pick(self.ns["_FAST_LOAD_DRAFT_KEYS"]), [])


class StartShWiring(unittest.TestCase):
    """The FAST_LOAD knob and the lane block of start.sh, cut out and run in bash."""

    @classmethod
    def setUpClass(cls):
        src = (REPO / "start.sh").read_text()
        lines = src.splitlines()
        cls.knob = [l for l in lines if l.startswith('FAST_LOAD="${FAST_LOAD:-0}"')
                    or l.startswith('[[ "$FAST_LOAD" == 0 || "$FAST_LOAD" == 1 ]]')]
        # Same cut as tests/test_v030_lane.py RenderedCommand.render().
        cls.block = re.search(r'^if \[\[ "\$V030" == "true" \]\]; then\n    PLE_ENV=.*?^fi\n',
                              src, re.S | re.M).group(0)

    def bash(self, script, env):
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                              env={"PATH": os.environ["PATH"], **env})

    def lane_vars(self, fast, v030):
        script = (f"FAST_LOAD={fast}\nV030={v030}\nPATCHED_LOADER=@PL@\nLOADER_PKG=@LP@\n"
                  + self.block + 'printf "%s\\n--\\n%s" "$PLE_ENV" "$OVERLAY_MOUNTS"')
        r = self.bash(script, {})
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def test_knob_lines_present(self):
        self.assertEqual(len(self.knob), 2, self.knob)

    def test_bad_value_stops_the_launch(self):
        script = "err() { echo \"ERR $*\"; exit 1; }\n" + "\n".join(self.knob)
        self.assertEqual(self.bash(script, {"FAST_LOAD": "2"}).returncode, 1)
        self.assertEqual(self.bash(script, {"FAST_LOAD": "1"}).returncode, 0)
        self.assertEqual(self.bash(script, {}).returncode, 0)

    def test_knob_off_changes_nothing(self):
        self.assertEqual(self.lane_vars(0, "false"),
                         self.lane_vars(1, "false").replace(" \\\n    -e VLLM_FAST_LOAD=1", "")
                         .replace(" \\\n    -v @PL@:@LP@:ro", ""))
        self.assertNotIn("VLLM_FAST_LOAD", self.lane_vars(0, "false"))
        self.assertNotIn("@PL@", self.lane_vars(0, "false"))

    def test_knob_on_adds_env_and_mount_on_default_lane_only(self):
        on = self.lane_vars(1, "false")
        self.assertIn("-e VLLM_FAST_LOAD=1", on)
        self.assertIn("-v @PL@:@LP@:ro", on)
        lane = self.lane_vars(1, "true")
        self.assertNotIn("VLLM_FAST_LOAD", lane)
        self.assertNotIn("@PL@", lane)


if __name__ == "__main__":
    unittest.main(verbosity=2)
