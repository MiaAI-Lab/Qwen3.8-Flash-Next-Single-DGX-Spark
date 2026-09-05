#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 MiaAI Lab (https://x.com/MiaAI_lab)
"""Patch vLLM's PLE CPU-offload machinery so it works on DGX Spark (GB10).

Why this exists (measured on this box, see docs/HANDOFF-single-spark.md):

1. GB10 reports CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 0. The stock
   offload design synchronises the CPU worker and the GPU stream with
   cuStreamWaitValue32 / cuStreamWriteValue32. On GB10 the wait call returns
   CUDA_SUCCESS but the *next* kernel launch on that stream blocks the host
   thread until the value is written, and nothing inside a captured CUDA
   graph waits at all. Result: the GPU worker hangs forever right after
   graph capture (the symptom the previous session hit).

   Fix: a host-side handshake. The GPU worker sends the request *and blocks*
   until the CPU worker has finished its H2D copy (copy_stream.synchronize())
   and written a sequence number into a shared-memory flag. The PLE output
   buffer is therefore complete before the forward (eager or graph replay)
   starts, and every stream-memory op becomes a no-op. The PLE layer sits at
   layer 1, so the lost CPU/GPU overlap is negligible.

2. The 26.8 GiB NVFP4 PLE table is memory-mapped from a pre-packed file
   (files/build_ple_packed_table.py) instead of being copied into anonymous
   RAM. On unified memory that is the difference between ~104 GiB and
   ~77 GiB of non-evictable footprint. Set VLLM_PLE_PACKED_TABLE_DIR to the
   directory holding "<layer_name>.ngram_embedding.packed_u8".

Inputs:  files/ple_offload/orig/*.py   (extracted from the image)
Outputs: files/ple_offload/*.py        (bind-mounted over the package)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "ple_offload", "orig")
OUT = os.path.join(HERE, "ple_offload")


def patch(name: str, edits: list[tuple[str, str]]) -> None:
    src = open(os.path.join(ORIG, name)).read()
    for old, new in edits:
        count = src.count(old)
        if count != 1:
            raise SystemExit(
                f"{name}: anchor not unique/missing (count={count}):\n{old[:300]}"
            )
        src = src.replace(old, new)
    open(os.path.join(OUT, name), "w").write(src)
    print(f"patched {name}")


# --------------------------------------------------------------------------
# protocol.py: sequence numbers + a per-worker shared-memory done flag.
# --------------------------------------------------------------------------
patch("protocol.py", [
    (
        "    input_ids_buf: torch.Tensor\n"
        "    query_start_loc_buf: torch.Tensor\n"
        "    ngram_context_buf: torch.Tensor | None\n",
        "    input_ids_buf: torch.Tensor\n"
        "    query_start_loc_buf: torch.Tensor\n"
        "    ngram_context_buf: torch.Tensor | None\n"
        "    # GB10 host-side handshake: shared CPU int64[1]; the CPU worker\n"
        "    # writes the request sequence number here once this worker's GPU\n"
        "    # output buffer holds the complete result.\n"
        "    done_flag: torch.Tensor | None = None\n",
    ),
    (
        "    dp_rank: int\n"
        "    num_tokens: int\n"
        "    num_reqs: int\n",
        "    dp_rank: int\n"
        "    num_tokens: int\n"
        "    num_reqs: int\n"
        "    seq: int = 0\n",
    ),
])

# --------------------------------------------------------------------------
# ple_offload_layer.py: every stream-memory op becomes a no-op.
# --------------------------------------------------------------------------
patch("ple_offload_layer.py", [
    (
        '    def reset(self, stream: torch.cuda.Stream | None = None) -> None:\n'
        '        """Enqueue ``WriteValue32(flag=0)`` on ``stream``."""\n'
        '        if stream is None:\n',
        '    def reset(self, stream: torch.cuda.Stream | None = None) -> None:\n'
        '        """No-op on GB10 (CAN_USE_STREAM_MEM_OPS=0): host handshake."""\n'
        '        return\n'
        '        if stream is None:\n',
    ),
    (
        '    def signal(self, stream: torch.cuda.Stream | None = None) -> None:\n'
        '        """Enqueue ``WriteValue32(flag=1)`` on ``stream``."""\n'
        '        if stream is None:\n',
        '    def signal(self, stream: torch.cuda.Stream | None = None) -> None:\n'
        '        """No-op on GB10 (CAN_USE_STREAM_MEM_OPS=0): host handshake."""\n'
        '        return\n'
        '        if stream is None:\n',
    ),
    (
        '    def wait_reset(self, stream: torch.cuda.Stream | None = None) -> None:\n'
        '        """Enqueue ``WaitValue32(flag==0)`` on ``stream``."""\n'
        '        if stream is None:\n',
        '    def wait_reset(self, stream: torch.cuda.Stream | None = None) -> None:\n'
        '        """No-op on GB10 (CAN_USE_STREAM_MEM_OPS=0): host handshake."""\n'
        '        return\n'
        '        if stream is None:\n',
    ),
    (
        '    """Wait for the CPU result without releasing its output buffer."""\n'
        '    stream = torch.cuda.current_stream()\n',
        '    """No-op: the connector already blocked the host until the CPU\n'
        '    result landed in gpu_output_buffer (GB10 has no stream mem ops)."""\n'
        '    return\n'
        '    stream = torch.cuda.current_stream()\n',
    ),
])

# --------------------------------------------------------------------------
# connector.py: synchronous request + done-flag wait on the model thread.
# --------------------------------------------------------------------------
patch("connector.py", [
    (
        "import os\n"
        "import queue\n"
        "import threading\n",
        "import os\n"
        "import queue\n"
        "import threading\n"
        "import time\n",
    ),
    # Allocate the done flag before registration.
    (
        "        # Runner input allocations are address-stable, so bind them once and\n"
        "        # pass only batch sizes through the per-forward request queue.\n",
        "        # Host-side completion flag (GB10 has no CUDA stream memory ops).\n"
        "        self._done_flag = torch.zeros(1, dtype=torch.int64, device=\"cpu\")\n"
        "        self._done_flag.share_memory_()\n"
        "        self._seq = 0\n"
        "        self._request_socket: zmq.Socket | None = None\n"
        "        self._wait_timeout_s = float(\n"
        "            os.environ.get(\"VLLM_PLE_OFFLOAD_STEP_TIMEOUT\", \"300\")\n"
        "        )\n"
        "\n"
        "        # Runner input allocations are address-stable, so bind them once and\n"
        "        # pass only batch sizes through the per-forward request queue.\n",
    ),
    # Replace the background request thread with a main-thread socket.
    (
        "                self._start_request_thread(ipc_addr)\n",
        "                # Requests are sent from the model thread so the wait for\n"
        "                # the CPU result can immediately follow them.\n"
        "                self._request_socket = self._zmq_ctx.socket(zmq.PUSH)\n"
        "                self._request_socket.connect(ipc_addr)\n",
    ),
    # Registration carries the done flag.
    (
        "            input_ids_buf=self._input_ids_buf,\n"
        "            query_start_loc_buf=self._query_start_loc_buf,\n"
        "            ngram_context_buf=self._ngram_context_buf,\n"
        "        )\n",
        "            input_ids_buf=self._input_ids_buf,\n"
        "            query_start_loc_buf=self._query_start_loc_buf,\n"
        "            ngram_context_buf=self._ngram_context_buf,\n"
        "            done_flag=self._done_flag,\n"
        "        )\n",
    ),
    # Synchronous launch.
    (
        "        # Inputs are replicated across TP ranks. One request per DP rank drives\n"
        "        # the CPU result fan-out to every registered TP output buffer.\n"
        "        if self.tp_rank != 0:\n"
        "            return\n"
        "\n"
        "        if self._uses_cuda_inputs:\n"
        "            assert self._input_ready_event is not None\n"
        "            # The background copy stream waits for runner input production\n"
        "            # without making the model stream wait for D2H completion.\n"
        "            self._input_ready_event.record(torch.cuda.current_stream(self.device))\n"
        "        request = PleOffloadRequest(\n"
        "            dp_rank=self.dp_rank,\n"
        "            num_tokens=num_tokens,\n"
        "            num_reqs=num_reqs,\n"
        "        )\n"
        "        self._request_queue.put_nowait(request)\n",
        "        # Inputs are replicated across TP ranks. One request per DP rank drives\n"
        "        # the CPU result fan-out to every registered TP output buffer. Every\n"
        "        # rank then blocks until its own output buffer is complete.\n"
        "        self._seq += 1\n"
        "        seq = self._seq\n"
        "        if self.tp_rank == 0:\n"
        "            if self._uses_cuda_inputs:\n"
        "                assert self._input_ready_event is not None\n"
        "                # The D2H stream waits for runner input production (and for\n"
        "                # the previous forward, so the output buffer is free).\n"
        "                self._input_ready_event.record(\n"
        "                    torch.cuda.current_stream(self.device)\n"
        "                )\n"
        "            request = PleOffloadRequest(\n"
        "                dp_rank=self.dp_rank,\n"
        "                num_tokens=num_tokens,\n"
        "                num_reqs=num_reqs,\n"
        "                seq=seq,\n"
        "            )\n"
        "            assert self._request_socket is not None\n"
        "            self._process_request(request, self._request_socket)\n"
        "        self._wait_done(seq)\n"
        "\n"
        "    def _wait_done(self, seq: int) -> None:\n"
        "        \"\"\"Block until the CPU worker reports ``seq`` complete.\"\"\"\n"
        "        flag = self._done_flag\n"
        "        deadline = time.monotonic() + self._wait_timeout_s\n"
        "        spins = 0\n"
        "        while int(flag[0]) < seq:\n"
        "            spins += 1\n"
        "            if spins < 2000:\n"
        "                continue\n"
        "            time.sleep(0.00005)\n"
        "            if time.monotonic() > deadline:\n"
        "                raise RuntimeError(\n"
        "                    f\"PLE offload worker did not complete step {seq} within \"\n"
        "                    f\"{self._wait_timeout_s}s (flag={int(flag[0])})\"\n"
        "                )\n",
    ),
    # Dummy runs: just zero the buffers; no stream memory ops.
    (
        "        stream = torch.cuda.current_stream(self.device)\n"
        "        for layer in self._layers.values():\n"
        "            layer._gpu_output_buffer[:num_tokens].zero_()\n"
        "            layer._sem.signal(stream)\n",
        "        for layer in self._layers.values():\n"
        "            layer._gpu_output_buffer[:num_tokens].zero_()\n",
    ),
    (
        "        stream = torch.cuda.current_stream(self.device)\n"
        "        for layer in self._layers.values():\n"
        "            layer.release_offloaded_output(stream)\n",
        "        # No-op with the host-side handshake: the next request is only\n"
        "        # sent after the previous forward completed on the model stream.\n"
        "        return\n",
    ),
    # close(): the request socket.
    (
        "        if self._registration_socket is not None:\n"
        "            self._registration_socket.close(linger=0)\n"
        "            self._registration_socket = None\n",
        "        if self._request_socket is not None:\n"
        "            self._request_socket.close(linger=0)\n"
        "            self._request_socket = None\n"
        "        if self._registration_socket is not None:\n"
        "            self._registration_socket.close(linger=0)\n"
        "            self._registration_socket = None\n",
    ),
])

# --------------------------------------------------------------------------
# worker.py: done-flag signalling + memory-mapped packed PLE table.
# --------------------------------------------------------------------------
patch("worker.py", [
    (
        "import contextlib\n"
        "import multiprocessing.process\n",
        "import contextlib\n"
        "import json\n"
        "import os\n"
        "import multiprocessing.process\n",
    ),
    (
        "    sem: CpuGpuSemaphore  # semaphore paired with gpu_output_buffer\n"
        "    copy_stream: torch.cuda.Stream\n",
        "    sem: CpuGpuSemaphore  # semaphore paired with gpu_output_buffer\n"
        "    copy_stream: torch.cuda.Stream\n"
        "    done_flag: torch.Tensor | None = None  # shared CPU int64[1]\n",
    ),
    # Memory-mapped packed table: skip shard tensors, attach mmap after load.
    (
        "        offload_prefixes = tuple(f\"{name}.\" for name in offload_layers)\n",
        "        offload_prefixes = tuple(f\"{name}.\" for name in offload_layers)\n"
        "\n"
        "        # Pre-packed, memory-mapped PLE tables (see patch_ple_offload.py).\n"
        "        packed_dir = os.environ.get(\"VLLM_PLE_PACKED_TABLE_DIR\", \"\")\n"
        "        packed_tables: dict[str, str] = {}\n"
        "        if packed_dir:\n"
        "            for name, layer in offload_layers.items():\n"
        "                path = os.path.join(packed_dir, f\"{name}.ngram_embedding.packed_u8\")\n"
        "                if os.path.exists(path) and os.path.exists(path + \".json\"):\n"
        "                    packed_tables[name] = path\n"
        "                    logger.info(\"PLE %s: using packed mmap table %s\", name, path)\n"
        "                else:\n"
        "                    logger.warning(\n"
        "                        \"PLE %s: no packed table at %s; loading shards into RAM\",\n"
        "                        name, path,\n"
        "                    )\n"
        "        packed_prefixes = tuple(\n"
        "            f\"{name}.ngram_embedding.shard_\" for name in packed_tables\n"
        "        )\n",
    ),
    (
        "                if mapped_name is not None and mapped_name.startswith(offload_prefixes):\n"
        "                    matched_checkpoint_tensors += 1\n"
        "                    yield weight_name, tensor\n",
        "                if mapped_name is not None and mapped_name.startswith(offload_prefixes):\n"
        "                    if packed_prefixes and mapped_name.startswith(packed_prefixes):\n"
        "                        # Served from the mmap table; never touch RAM.\n"
        "                        matched_checkpoint_tensors += 1\n"
        "                        continue\n"
        "                    matched_checkpoint_tensors += 1\n"
        "                    yield weight_name, tensor\n",
    ),
    (
        "            expected_offload_params = {\n"
        "                f\"{layer_name}.{param_name}\"\n"
        "                for layer_name, layer in offload_layers.items()\n"
        "                for param_name, _ in layer.named_parameters()\n"
        "            }\n",
        "            expected_offload_params = {\n"
        "                f\"{layer_name}.{param_name}\"\n"
        "                for layer_name, layer in offload_layers.items()\n"
        "                for param_name, _ in layer.named_parameters()\n"
        "                if not (\n"
        "                    layer_name in packed_tables\n"
        "                    and param_name\n"
        "                    in (\"ngram_embedding.weight\", \"ngram_embedding.weight_scale\")\n"
        "                )\n"
        "            }\n",
    ),
    (
        "        for layer in offload_layers.values():\n"
        "            process_weights_after_loading(layer, model_config, torch.device(\"cpu\"))\n",
        "        for name, layer in offload_layers.items():\n"
        "            if name in packed_tables:\n"
        "                self._attach_packed_table(name, layer, packed_tables[name])\n"
        "            process_weights_after_loading(layer, model_config, torch.device(\"cpu\"))\n",
    ),
    (
        "    def accept_registrations(\n",
        "    @staticmethod\n"
        "    def _attach_packed_table(name: str, layer: PleOffloadLayer, path: str) -> None:\n"
        "        \"\"\"Replace the shard-loaded table with a read-only memory map.\"\"\"\n"
        "        import numpy as np\n"
        "\n"
        "        meta = json.load(open(path + \".json\"))\n"
        "        rows, width = int(meta[\"total_rows\"]), int(meta[\"row_width\"])\n"
        "        emb = layer.ngram_embedding\n"
        "        quant_method = getattr(emb, \"quant_method\", None)\n"
        "        expected_width = getattr(quant_method, \"packed_row_width\", None)\n"
        "        if expected_width is not None and expected_width != width:\n"
        "            raise RuntimeError(\n"
        "                f\"PLE {name}: packed table row width {width} != \"\n"
        "                f\"expected {expected_width}\"\n"
        "            )\n"
        "        if emb.weight.shape[0] != rows:\n"
        "            raise RuntimeError(\n"
        "                f\"PLE {name}: packed table has {rows} rows, model expects \"\n"
        "                f\"{emb.weight.shape[0]}\"\n"
        "            )\n"
        "        if os.path.getsize(path) != rows * width:\n"
        "            raise RuntimeError(f\"PLE {name}: packed table size mismatch\")\n"
        "        mm = np.memmap(path, dtype=np.uint8, mode=\"r\", shape=(rows, width))\n"
        "        # 27 GiB of randomly-accessed rows against a far smaller page\n"
        "        # cache. Default mmap behaviour faults in a ~64 KiB window per\n"
        "        # touched row (fault-around); nearly all of it is never read.\n"
        "        # Declaring the access random makes each fault cost one page.\n"
        "        try:\n"
        "            import mmap as _mmap_mod\n"
        "            mm._mmap.madvise(_mmap_mod.MADV_RANDOM)\n"
        "            _advice = \"MADV_RANDOM\"\n"
        "        except Exception as _exc:  # advisory only, never fatal\n"
        "            _advice = f\"no madvise ({_exc})\"\n"
        "        table = torch.from_numpy(mm)  # zero-copy, file-backed, evictable\n"
        "        emb._packed_table = table\n"
        "        emb._packed_table_mmap = mm\n"
        "        # Same file, opened for posix_fadvise(WILLNEED): the gather in\n"
        "        # ple_layer batches its page reads through this fd so the NVMe\n"
        "        # sees them together instead of one fault at a time.\n"
        "        emb._packed_table_fd = os.open(path, os.O_RDONLY)\n"
        "        # Release the never-touched anonymous allocations.\n"
        "        emb.weight.data = torch.empty(0, dtype=emb.weight.dtype)\n"
        "        ws = getattr(emb, \"weight_scale\", None)\n"
        "        if ws is not None and ws.dim() == 2:\n"
        "            ws.data = torch.empty(0, dtype=ws.dtype)\n"
        "        logger.info(\n"
        "            \"PLE %s: mmap table attached (%d rows x %d B = %.2f GiB) [%s]\",\n"
        "            name, rows, width, rows * width / 2**30, _advice,\n"
        "        )\n"
        "\n"
        "    def accept_registrations(\n",
    ),
    (
        "                    copy_stream=torch.cuda.Stream(device=gpu_buffer.device),\n"
        "                )\n",
        "                    copy_stream=torch.cuda.Stream(device=gpu_buffer.device),\n"
        "                    done_flag=registration.done_flag,\n"
        "                )\n",
    ),
    # No stream-memop reset wait; signal completion through the done flag.
    (
        "                for target in targets:\n"
        "                    target.copy_stream.synchronize()\n"
        "                    target.sem.wait_reset(target.copy_stream)\n"
        "\n",
        "",
    ),
    (
        "                for target in targets:\n"
        "                    with torch.cuda.stream(target.copy_stream):\n"
        "                        target.gpu_output_buffer[slices].copy_(\n"
        "                            result[slices], non_blocking=True\n"
        "                        )\n"
        "                        target.sem.signal(target.copy_stream)\n",
        "                for target in targets:\n"
        "                    with torch.cuda.stream(target.copy_stream):\n"
        "                        target.gpu_output_buffer[slices].copy_(\n"
        "                            result[slices], non_blocking=True\n"
        "                        )\n"
        "\n"
        "        # Host-side handshake (GB10 has no stream memory ops): wait for\n"
        "        # every layer's DMA to land, then publish the sequence number so\n"
        "        # the blocked GPU worker(s) can start their forward.\n"
        "        for dp_rank, request in requests_by_dp.items():\n"
        "            flags = []\n"
        "            for layer_name in self._layers:\n"
        "                for target in self._worker_targets[dp_rank][layer_name]:\n"
        "                    target.copy_stream.synchronize()\n"
        "                    if target.done_flag is not None:\n"
        "                        flags.append(target.done_flag)\n"
        "            for flag in flags:\n"
        "                flag[0] = request.seq\n",
    ),
])
print("ok")
