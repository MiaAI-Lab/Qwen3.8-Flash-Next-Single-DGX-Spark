import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parent.parent
FILES = REPO / "files"
START = REPO / "start.sh"
WRAPPER = REPO / "start-v030.sh"
SUPERVISE = REPO / "scripts" / "supervise.sh"
MTP_ORIG = Path(os.environ.get("V030_MTP_ORIG", "/m/qwen-dual-wt/files/mtp_v030_patched.py.orig"))

LANE_ON = re.compile(r'"\$(V030|\{V030:-\})" == "true"')
LANE_OFF = '"$V030" != "true"'
LANE_TOKENS = ("v030_fp8kv", "v030_ple", "VLLM_PLE_MMAP_DIR", "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB",
               "index_share_for_mtp_iteration",
               "--kv-cache-memory-bytes", "--engram-config", "patch_ple_mmap_v030.py",
               "patch_mtp_draft_vocab_v030.py", "patch_qsa_fp8_kv_v030.py", "fi_autotune")
DAY0_TOKENS = ("patch_ple_layer.py", "patch_modelopt_mxfp8.py", "patch_qsa_fp8_kv.py\"",
               "patch_determinism.py", "patch_mtp_draft_vocab.py\"", "patch_block_drop.py",
               "patch_ple_offload.py", "build_ple_packed_table.py", "VLLM_PLE_PACKED_TABLE_DIR",
               "VLLM_PLE_OFFLOAD_STEP_TIMEOUT")

DAY0_HEREDOC = r"""#!/bin/bash
docker run \\
    -d --name $CONTAINER_NAME \\
    --gpus all --network host --ipc host \\
    --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \\
    --memory ${CONTAINER_MEM_GIB}g --memory-swap ${CONTAINER_MEM_GIB}g \\
    --log-opt max-size=50m --log-opt max-file=3 \\
    -e HF_HUB_OFFLINE=1 \\
    -e TRANSFORMERS_OFFLINE=1 \\
    -e VLLM_PLE_CPU_OFFLOAD=1 \\
    -e VLLM_PLE_PACKED_TABLE_DIR=$PLE_CACHE_CTR \\
    -e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300 \\
    -e MAX_JOBS=2 \\
    -e FLASHINFER_NVCC_THREADS=1 \\
    ${VLLM_QSA_DET_TOPK:+-e VLLM_QSA_DET_TOPK=$VLLM_QSA_DET_TOPK} \\
    ${VLLM_MOE_DET_FINALIZE:+-e VLLM_MOE_DET_FINALIZE=$VLLM_MOE_DET_FINALIZE} \\
    $( [[ "$VLLM_MOE_DET_FINALIZE" == 1 ]] && echo "-e VLLM_FLASHINFER_MOE_FUSED_FINALIZE=0 -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/vllm/flashinfer_autotune_cache_unfused" ) \\
    ${GDN_DECODE_KERNEL:+-e VLLM_GDN_DECODE_KERNEL=$GDN_DECODE_KERNEL} \\
    ${MTP_DRAFT_VOCAB:+-v $MTP_DRAFT_VOCAB:/root/draft_vocab.txt:ro} \\
    ${MTP_DRAFT_VOCAB:+-e VLLM_MTP_DRAFT_VOCAB=/root/draft_vocab.txt} \\
    ${CHAT_TEMPLATE:+-v $CHAT_TEMPLATE:/root/chat_template.jinja:ro} \\
    -e HF_HOME=/root/.cache/huggingface \\
    ${HF_TOKEN:+-e HF_TOKEN=\$HF_TOKEN} \\
    -v $PATCHED_PLE:$PLE_PKG:ro \\
    -v $PATCHED_MODELOPT:$MODELOPT_PKG:ro \\
    -v $PATCHED_QSA_OPS:$QSA_OPS_PKG:ro \\
    -v $PATCHED_QSA_NVIDIA:$QSA_NVIDIA_PKG:ro \\
    -v $PATCHED_MTP:$MTP_PKG:ro \\
    -v $DET_DIR/flashinfer_cutlass_moe.py:$MOE_CUTLASS_PKG:ro \\
    $BLOCK_DROP_MOUNTS \\
    -v $OFFLOAD_DIR/ple_offload_layer.py:$VLLM_PKG/model_executor/layers/ple_offload_layer.py:ro \\
    -v $OFFLOAD_DIR/connector.py:$VLLM_PKG/v1/ple_offload/connector.py:ro \\
    -v $OFFLOAD_DIR/worker.py:$VLLM_PKG/v1/ple_offload/worker.py:ro \\
    -v $OFFLOAD_DIR/protocol.py:$VLLM_PKG/v1/ple_offload/protocol.py:ro \\
    -v $HF_CACHE_DIR:/root/.cache/huggingface \\
    -v $HOME/.cache/vllm:/root/.cache/vllm \\
    $EXTRA_DOCKER_ARGS \\
    $IMAGE \\
    $MODEL_ID \\
    $VLLM_ARGS_STR \\
    --host $BIND \\
    --port $PORT \\
    ${API_KEY:+--api-key \$API_KEY} \\
"""


def lane_state_per_line(lines):
    stack = []
    states = []
    for line in lines:
        s = line.strip()
        if s.startswith("if ") and not s.endswith(":"):
            stack.append(True if LANE_ON.search(s) else False if LANE_OFF in s else None)
        elif s == "else" and stack and stack[-1] is not None:
            stack[-1] = not stack[-1]
        elif re.fullmatch(r"fi\b.*", s) and stack:
            stack.pop()
        inline = True if LANE_ON.search(s) else False if LANE_OFF in s else None
        known = [x for x in stack if x is not None]
        states.append(inline if inline is not None else (known[-1] if known else None))
    return states


class StartGates(unittest.TestCase):
    def setUp(self):
        self.lines = START.read_text().splitlines()
        self.states = lane_state_per_line(self.lines)

    def test_scripts_parse(self):
        r = subprocess.run(["bash", "-n", str(START), str(WRAPPER)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_lane_tokens_only_inside_lane_gates(self):
        for i, (line, state) in enumerate(zip(self.lines, self.states), 1):
            if not line.lstrip().startswith("#") and any(t in line for t in LANE_TOKENS):
                self.assertIs(state, True, f"start.sh:{i} lane token outside a V030 gate: {line}")

    def test_day0_steps_skipped_on_the_lane(self):
        for i, (line, state) in enumerate(zip(self.lines, self.states), 1):
            if not line.lstrip().startswith("#") and any(t in line for t in DAY0_TOKENS):
                self.assertIs(state, False, f"start.sh:{i} day-0 step not gated off the lane: {line}")

    def test_lane_rejects_unsupported_knobs(self):
        src = START.read_text()
        for needle in ('V030: ABLIT=1', 'V030: YARN=1', 'V030: MTP_K_SCHEDULE',
                       'V030: the determinism knobs', 'V030: the vLLM 0.30 lane serves only'):
            self.assertIn(needle, src)

    def test_indexer_logits_cap_defaults_and_validates(self):
        src = START.read_text()
        self.assertIn('V030_INDEXER_LOGITS_MB="${V030_INDEXER_LOGITS_MB:-128}"', src)
        self.assertIn('V030_INDEXER_LOGITS_MB must be a positive integer', src)
        # The validation sits inside the V030 gate, so the default lane never sees it.
        lines, states = self.lines, self.states
        for i, (line, state) in enumerate(zip(lines, states), 1):
            if 'V030_INDEXER_LOGITS_MB must be a positive integer' in line:
                self.assertIs(state, True, f"start.sh:{i} validation outside the V030 gate")


class RenderedCommand(unittest.TestCase):
    def render(self, template_body, lane):
        src = START.read_text()
        block = re.search(r'^if \[\[ "\$V030" == "true" \]\]; then\n    PLE_ENV=.*?^fi\n', src, re.S | re.M).group(0)
        names = sorted(set(re.findall(r"\$\{?([A-Z_][A-Z0-9_]*)", block + template_body)) - {"V030", "HOME", "HF_TOKEN", "API_KEY"})
        setup = "".join(f'{n}="@{n}@"\n' for n in names)
        script = (f"V030={'true' if lane else 'false'}\nHF_TOKEN=x\nAPI_KEY=x\nVLLM_MOE_DET_FINALIZE=1\n"
                  + setup + block + "cat <<LAUNCH_EOF\n" + template_body + "LAUNCH_EOF\n")
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def current_template(self):
        src = START.read_text()
        return re.search(r'<<LAUNCH_EOF\n(.*?)^LAUNCH_EOF\n', src, re.S | re.M).group(1)

    def test_day0_command_unchanged(self):
        self.assertEqual(self.render(self.current_template(), lane=False), self.render(DAY0_HEREDOC, lane=False))

    def test_lane_command_swaps_overlays(self):
        out = self.render(self.current_template(), lane=True)
        for needle in ("VLLM_PLE_MMAP_DIR=/root/.cache/vllm/ple_mmap_v030", "VLLM_PLE_MMAP_ADVICE=1",
                       "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=@V030_INDEXER_LOGITS_MB@",
                       "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/fi_autotune", "@V030_MOUNTS@"):
            self.assertIn(needle, out)
        for needle in ("VLLM_PLE_PACKED_TABLE_DIR", "@PATCHED_PLE@", "@OFFLOAD_DIR@", "@BLOCK_DROP_MOUNTS@"):
            self.assertNotIn(needle, out)


class LaunchWrapper(unittest.TestCase):
    def test_wrapper_sets_the_lane(self):
        src = WRAPPER.read_text()
        for needle in ("export V030=true", "export IMAGE=vllm/vllm-openai:v0.30.0",
                       'TP1_MODEL_ID="${TP1_MODEL_ID:-nvidia/Qwen3.8-Flash-Next-NVFP4}"',
                       'KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"', 'PLE_GIB="${PLE_GIB:-47.68}"',
                       'MEMWATCH_MIN_GIB="${MEMWATCH_MIN_GIB:-3}"', 'MEMWATCH_MIN_FREE_GIB="${MEMWATCH_MIN_FREE_GIB:-1}"',
                       'V030_INDEXER_LOGITS_MB="${V030_INDEXER_LOGITS_MB:-128}"',
                       'exec "$SCRIPT_DIR/start.sh"'):
            self.assertIn(needle, src)

    def test_watchdog_floors_are_integers_below_the_soak_gate(self):
        src = WRAPPER.read_text()
        avail = int(re.search(r'MEMWATCH_MIN_GIB:-([^}]+)\}', src).group(1))
        free = int(re.search(r'MEMWATCH_MIN_FREE_GIB:-([^}]+)\}', src).group(1))
        self.assertLessEqual(avail, 3)
        self.assertLess(free, 1.5)

    def test_new_knobs_beat_dotenv(self):
        snapshot = re.search(r"_ENV_SNAPSHOT_VARS=\((.*?)\)", START.read_text(), re.S).group(1).split()
        for name in ("V030", "V030_KV_GIB", "IMAGE", "PLE_GIB", "MEMWATCH_MIN_GIB", "MEMWATCH_MIN_FREE_GIB",
                     "MTP_DISABLE_BLOCK_DROP", "V030_INDEXER_LOGITS_MB"):
            self.assertIn(name, snapshot)


LANE_LOADER = REPO / "scripts" / "launch-lane.sh"
MAINT = REPO / "scripts" / "maintenance-relaunch.sh"
LANE_RECORD = "V030=true\nMEMWATCH_MIN_GIB=3\nMEMWATCH_MIN_FREE_GIB=1\n"
DAY0_RECORD = "V030=false\nMEMWATCH_MIN_GIB=3\nMEMWATCH_MIN_FREE_GIB=1\n"


class SuperviseLane(unittest.TestCase):
    def decide(self, record):
        src = SUPERVISE.read_text()
        tick = re.search(r"^    load_launch_lane\n(    MEMWATCH_MIN_GIB=.*\n    MEMWATCH_MIN_FREE_GIB=.*\n)", src, re.M).group(0)
        with tempfile.TemporaryDirectory() as repo:
            (Path(repo) / "logs").mkdir()
            if record is not None:
                (Path(repo) / "logs" / "launch-lane").write_text(record)
            script = (f'REPO_DIR="{repo}"\n_DEFAULT_MEMWATCH_MIN_GIB=6\n_DEFAULT_MEMWATCH_MIN_FREE_GIB=2\n'
                      f'source "{LANE_LOADER}"\n' + tick
                      + 'echo "${START_SCRIPT#$REPO_DIR/} $MEMWATCH_MIN_GIB $MEMWATCH_MIN_FREE_GIB"\n')
            r = subprocess.run(["bash", "-uc", "set -o pipefail\n" + script], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout.split()

    def test_no_record_keeps_start_sh_and_defaults(self):
        self.assertEqual(self.decide(None), ["start.sh", "6", "2"])

    def test_day0_record_ignores_its_floors(self):
        self.assertEqual(self.decide(DAY0_RECORD), ["start.sh", "6", "2"])

    def test_lane_record_relaunches_the_wrapper_with_its_floors(self):
        self.assertEqual(self.decide(LANE_RECORD), ["start-v030.sh", "3", "1"])

    def test_lane_record_rejects_non_integer_floors(self):
        rec = "V030=true\nMEMWATCH_MIN_GIB=2.5\nMEMWATCH_MIN_FREE_GIB=$(false)\n"
        self.assertEqual(self.decide(rec), ["start-v030.sh", "6", "2"])

    def test_supervisor_uses_the_decision(self):
        src = SUPERVISE.read_text()
        self.assertIn('source "$REPO_DIR/scripts/launch-lane.sh"', src)
        self.assertIn('if "$START_SCRIPT" >"$_attempt_log" 2>&1; then', src)
        self.assertNotIn('"$REPO_DIR/start.sh" >"$_attempt_log"', src)
        self.assertRegex(src, r"while true; do\n    load_launch_lane\n")

    def test_start_records_the_lane_after_the_watchdog(self):
        src = START.read_text()
        at = src.index('bash "$SCRIPT_DIR/scripts/start-memwatch.sh"')
        rec = src.index('> "$SCRIPT_DIR/logs/launch-lane"')
        self.assertLess(at, rec)
        self.assertIn("V030=%s\\nMEMWATCH_MIN_GIB=%s\\nMEMWATCH_MIN_FREE_GIB=%s", src)


class MaintenanceLane(unittest.TestCase):
    STUB = "#!/bin/bash\necho \"$(basename \"$0\") ${MEMWATCH_MIN_GIB-unset} ${MEMWATCH_MIN_FREE_GIB-unset}\" >> \"$(dirname \"$0\")/../calls\"\n"

    def run_maintenance(self, record):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "scripts").mkdir(parents=True)
            (repo / "logs").mkdir()
            shutil.copy(MAINT, repo / "scripts" / MAINT.name)
            shutil.copy(LANE_LOADER, repo / "scripts" / LANE_LOADER.name)
            for stub in ("start.sh", "start-v030.sh", "stop.sh"):
                (repo / stub).write_text(self.STUB.replace("/..", ""))
                (repo / stub).chmod(0o755)
            for stub in ("smoke-test.sh", "alert.sh"):
                (repo / "scripts" / stub).write_text(self.STUB)
                (repo / "scripts" / stub).chmod(0o755)
            (repo / ".env").write_text("PORT=1\n")
            if record is not None:
                (repo / "logs" / "launch-lane").write_text(record)
            env = {k: v for k, v in os.environ.items() if not k.startswith("MEMWATCH_")}
            r = subprocess.run(["bash", str(repo / "scripts" / MAINT.name)], capture_output=True, text=True,
                               env=env, timeout=60)
            calls = (repo / "calls").read_text().split("\n")
            return r.returncode, [c for c in calls if c], (repo / "logs" / "stopping").exists()

    def test_no_record_relaunches_start_sh_as_today(self):
        rc, calls, flag = self.run_maintenance(None)
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["stop.sh unset unset", "start.sh unset unset", "smoke-test.sh unset unset"])
        self.assertFalse(flag)

    def test_day0_record_relaunches_start_sh_without_floors(self):
        rc, calls, _ = self.run_maintenance(DAY0_RECORD)
        self.assertEqual(rc, 0)
        self.assertIn("start.sh unset unset", calls)

    def test_lane_record_relaunches_the_wrapper_with_its_floors(self):
        rc, calls, flag = self.run_maintenance(LANE_RECORD)
        self.assertEqual(rc, 0)
        self.assertIn("start-v030.sh 3 1", calls)
        self.assertNotIn("start.sh", " ".join(c.split()[0] for c in calls).split())
        self.assertFalse(flag)


@unittest.skipUnless(MTP_ORIG.is_file(), f"no extracted v0.30 mtp.py at {MTP_ORIG}; set V030_MTP_ORIG")
class DraftVocabV030(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        for name in ("patch_mtp_draft_vocab_v030.py", "patch_mtp_draft_vocab.py"):
            shutil.copy(FILES / name, self.tmp / name)
        shutil.copy(MTP_ORIG, self.tmp / "mtp_v030_patched.py.orig")

    def tearDown(self):
        self._tmp.cleanup()

    def run_patch(self):
        return subprocess.run([sys.executable, str(self.tmp / "patch_mtp_draft_vocab_v030.py")],
                              capture_output=True, text=True)

    def test_patches_get_top_tokens_and_attach(self):
        r = self.run_patch()
        self.assertEqual(r.returncode, 0, r.stderr)
        src = (self.tmp / "mtp_v030_patched.py").read_text()
        ast.parse(src)
        self.assertIn("def get_top_tokens(self, hidden_states", src)
        self.assertIn("_attach_draft_vocab(self)", src)
        self.assertIn('getattr(lm_head, "tp_size", 1) != 1', src)
        self.assertEqual(src.count("logger = init_logger(__name__)"), 1)
        self.assertEqual(src.count("\nimport os\n"), 1)

    def test_rerun_gives_same_output(self):
        self.run_patch()
        first = (self.tmp / "mtp_v030_patched.py").read_text()
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual((self.tmp / "mtp_v030_patched.py").read_text(), first)

    def test_moved_anchor_fails_without_writing(self):
        orig = self.tmp / "mtp_v030_patched.py.orig"
        orig.write_text(orig.read_text().replace("return loader.load_weights(", "return  loader.load_weights("))
        r = self.run_patch()
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((self.tmp / "mtp_v030_patched.py").exists())

    def test_patched_original_is_refused(self):
        self.assertEqual(self.run_patch().returncode, 0)
        shutil.copy(self.tmp / "mtp_v030_patched.py", self.tmp / "mtp_v030_patched.py.orig")
        self.assertNotEqual(self.run_patch().returncode, 0)


if __name__ == "__main__":
    unittest.main()
