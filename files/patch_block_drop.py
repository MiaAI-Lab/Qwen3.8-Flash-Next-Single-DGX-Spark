#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Backport vllm-project/vllm#53388 (disable_eagle_block_drop) to the image.

With EAGLE-style drafters (MTP included), the prefix cache drops the last
matched block of a request and computes it again. On a multi-turn chat that is
one full cache block (1,664 tokens at MTP 3) of extra prefill on each turn.
The speculative-config key "disable_eagle_block_drop": true keeps that block.
start.sh already merges the key when MTP_DISABLE_BLOCK_DROP=1, but the pinned
image does not know the key, and its SpeculativeConfig rejects unknown keys.
This backport adds it.

The drafter still runs. The change can move acceptance only: the target model
verifies every draft token, so the output does not change.

vllm#53388 changes seven vllm files. This backport changes six of them: the
config, the KV cache utils, the scheduler, and the three KV transfer and
offload users of the old check (mooncake store, offloading connector, simple
CPU offload). This recipe does not configure a KV connector. But a connector
that drops the block while the scheduler keeps it disagrees on the cached
length, so the connectors follow the new check too. The seventh file,
single_type_kv_cache_manager.py, gets a sliding-window fix in vllm#53388.
This backport leaves it out: the QSA attention of this model does not
support sliding windows. The upstream tests are not part of the image.

start.sh runs this only when MTP_DISABLE_BLOCK_DROP=1 and MTP is on.

Inputs:  files/block_drop/orig/<path>   (start.sh extracts them from the image)
Outputs: files/block_drop/<path>        (start.sh mounts them over the package)
<path> is the path under the vllm package, for example v1/core/sched/scheduler.py.

When the image already has the option, the script removes old outputs and
writes nothing, and start.sh mounts nothing.

    python3 files/patch_block_drop.py [orig_dir] [out_dir]
    python3 files/patch_block_drop.py --list    # the paths, one on each line
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Paths under the vllm package. start.sh reads this list through --list.
SPEC = "config/speculative.py"
FILES = (
    SPEC,
    "v1/core/kv_cache_utils.py",
    "v1/core/sched/scheduler.py",
    "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py",
    "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
    "v1/simple_kv_offload/manager.py",
)
MARK = "disable_eagle_block_drop"

EDITS = {
    SPEC: [
        ("    use_local_argmax_reduction: bool = False\n",
         "    disable_eagle_block_drop: bool = False\n"
         '    """Disable dropping the trailing prefix-cache block for EAGLE-like\n'
         "    speculative methods (backport of vllm#53388). The drafter still runs;\n"
         '    only prefix-cache reuse of that block changes."""\n'
         "    use_local_argmax_reduction: bool = False\n"),
        ('        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")\n',
         '        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")\n\n'
         "    def use_eagle_block_drop(self) -> bool:\n"
         '        """Whether volatile trailing cache blocks should be discarded."""\n'
         "        return self.use_eagle() and not self.disable_eagle_block_drop\n"),
    ],
    "v1/core/kv_cache_utils.py": [
        ("    if spec_config is None or not spec_config.use_eagle():\n",
         "    if spec_config is None or not spec_config.use_eagle_block_drop():\n"),
    ],
    "v1/core/sched/scheduler.py": [
        ("        self.use_eagle = False\n",
         "        self.use_eagle = False\n        self.use_eagle_block_drop = False\n"),
        ("            self.use_eagle = speculative_config.use_eagle()\n",
         "            self.use_eagle = speculative_config.use_eagle()\n"
         "            self.use_eagle_block_drop = speculative_config.use_eagle_block_drop()\n"
         "            if self.use_eagle and not self.use_eagle_block_drop:\n"
         "                logger.warning(\n"
         '                    "EAGLE trailing prefix-cache block dropping is disabled "\n'
         '                    "(vllm#53388 backport)."\n'
         "                )\n"),
        ("            use_eagle=self.use_eagle,\n",
         "            use_eagle=self.use_eagle_block_drop,\n"),
        ("        if self.use_eagle:\n"
         "            last_cache_position = max(last_cache_position - block_size, 0)\n",
         "        if self.use_eagle_block_drop:\n"
         "            last_cache_position = max(last_cache_position - block_size, 0)\n"),
    ],
    # The KV transfer and offload users of the old check follow the new one.
    "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py": [
        ("            spec_cfg.use_eagle()\n"
         '            if spec_cfg is not None and callable(getattr(spec_cfg, "use_eagle", None))\n',
         "            spec_cfg.use_eagle_block_drop()\n"
         "            if spec_cfg is not None\n"
         '            and callable(getattr(spec_cfg, "use_eagle_block_drop", None))\n'),
    ],
    "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py": [
        ("            and vllm_config.speculative_config.use_eagle()\n"
         "        )\n"
         "        if use_eagle and not eagle_groups:\n",
         "            and vllm_config.speculative_config.use_eagle_block_drop()\n"
         "        )\n"
         "        if use_eagle and not eagle_groups:\n"),
    ],
    "v1/simple_kv_offload/manager.py": [
        ("        use_eagle = spec_config is not None and spec_config.use_eagle()\n",
         "        use_eagle = spec_config is not None and spec_config.use_eagle_block_drop()\n"),
    ],
}


def patch(name: str, src: str) -> str:
    for old, new in EDITS[name]:
        count = src.count(old)
        if count != 1:
            raise SystemExit(
                f"patch_block_drop: {name}: anchor not unique/missing (count={count}):\n{old[:200]}"
            )
        src = src.replace(old, new)
    return src


def main() -> None:
    if sys.argv[1:] == ["--list"]:
        print("\n".join(FILES))
        return
    orig = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "block_drop", "orig")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "block_drop")
    sources = {}
    for name in FILES:
        path = os.path.join(orig, name)
        if not os.path.isfile(path):
            raise SystemExit(f"patch_block_drop: missing {path}")
        sources[name] = open(path).read()
    if MARK in sources[SPEC]:
        for name in FILES:
            try:
                os.remove(os.path.join(out, name))
            except FileNotFoundError:
                pass
        print("patch_block_drop: the image already has disable_eagle_block_drop; nothing to mount")
        return
    # Patch all files before writing any, so a failed anchor leaves no partial set.
    patched = {name: patch(name, src) for name, src in sources.items()}
    for name, src in patched.items():
        path = os.path.join(out, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(src)
        print(f"patched {name}")


if __name__ == "__main__":
    main()
