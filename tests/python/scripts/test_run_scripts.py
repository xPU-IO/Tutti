"""Validate launch configuration without loading a model or touching devices."""

import importlib.util
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVE_SCRIPT = REPO_ROOT / "scripts/vllm/run/serve-8gpu-striped.sh"
PROFILE_ENV = REPO_ROOT / "scripts/vllm/run/profile-env.sh"


@pytest.fixture
def launch_env(tmp_path):
    config = tmp_path / 'daemon with "quotes".yaml'
    config.write_text("nvmes:\n  - device_id: 0\n  - device_id: 1\n")
    return {
        **os.environ,
        "MODEL": "example/model",
        "TUTTI_PYTHON": sys.executable,
        "TUTTI_PROFILE_ROOT": str(tmp_path / "scratch"),
        "DAEMON_CONFIG": str(config),
        "DEVICE_GROUPS": "0,1",
        "KV_MOUNT": str(tmp_path / 'kv with "quotes"'),
        "TP_SIZE": "2",
        "CHUNK_TOKENS": "512",
        "BLOCK_SIZE": "512",
        "TOTAL_CAPACITY": "2 KiB",
        "POOL_TAG": "launch-test",
        "CUDA_VISIBLE_DEVICES": "3,5",
        "LOAD_FORMAT": "auto",
        "PHXLOADER_DIR": "",
        "NUM_QUEUES": "8",
    }


def preview(env):
    return subprocess.run(
        ["bash", str(SERVE_SCRIPT), "--dry-run"],
        env=env, text=True, capture_output=True, timeout=30,
    )


def test_serve_preview_has_real_connector_config(launch_env):
    result = preview(launch_env)
    assert result.returncode == 0, result.stderr
    config_line = next(line for line in result.stdout.splitlines()
                       if line.startswith("[serve] kv_config="))
    config = json.loads(config_line.split("=", 1)[1])
    assert config["kv_connector_module_path"] == "tutti.integration.vllm.connector"
    assert config["kv_load_failure_policy"] == "fail"
    options = config["kv_connector_extra_config"]["store"]["options"]
    assert options["capacity"] == "1024 B"
    assert "capacity_bytes" not in options and "num_chunks" not in options
    assert options["root"] == str(Path(launch_env["KV_MOUNT"]) /
                                   "tutti-kv-online-launch-test-{LOCAL_RANK}")
    assert options["preset"]["daemon_config"] == [launch_env["DAEMON_CONFIG"]]
    assert options["preset"]["device_groups"] == [[0, 1]]
    assert options["preset"]["type"] == "raid0"
    assert "--load-format auto" in result.stdout


@pytest.mark.parametrize("eager", ["0", "1"])
def test_serve_preview_compute_mode(launch_env, eager):
    result = preview({**launch_env, "ENFORCE_EAGER": eager})
    assert result.returncode == 0, result.stderr
    command = next(line for line in result.stdout.splitlines()
                   if line.startswith("[serve] command:"))
    args = shlex.split(command.split(":", 1)[1])
    if eager == "1":
        assert "--enforce-eager" in args
        assert "--compilation-config" not in args
    else:
        assert "--enforce-eager" not in args
        assert json.loads(args[args.index("--compilation-config") + 1]) == {
            "mode": 3, "cudagraph_mode": "PIECEWISE",
        }


@pytest.mark.parametrize("overrides, message", [
    ({"TP_SIZE": "0"}, "positive integers"),
    ({"TOTAL_CAPACITY": "1 B"}, "at least one byte per rank"),
    ({"TOTAL_CAPACITY": "0 GiB"}, "正整数字节"),
    ({"TOTAL_CAPACITY_BYTES": "2048"}, "已删除"),
    ({"CHUNK_TOKENS": "513"}, "divisible by BLOCK_SIZE"),
    ({"DEVICE_GROUPS": "0;1;0"}, "group count must divide TP_SIZE"),
    ({"DEVICE_GROUPS": "0;9"}, "devices missing"),
    ({"KV_MOUNT": "relative"}, "must be absolute"),
    ({"POOL_TAG": "../unsafe"}, "directory name"),
])
def test_serve_preview_rejects_invalid_config(launch_env, overrides, message):
    result = preview({**launch_env, **overrides})
    assert result.returncode != 0
    assert message in result.stderr


def test_serve_preview_applies_queue_override(launch_env):
    result = preview({**launch_env, "NUM_QUEUES": "4"})
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("[serve] kv_config="))
    config = json.loads(line.split("=", 1)[1])
    assert config["kv_connector_extra_config"]["store"]["options"]["preset"]["num_queues"] == 4


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_serve_preview_rejects_invalid_queue_count(launch_env, value):
    result = preview({**launch_env, "NUM_QUEUES": value})
    assert result.returncode != 0
    assert "positive integers" in result.stderr


def test_only_module_config_templates_exist():
    assert (REPO_ROOT / "config/device_manager/nvme_daemon.yaml").is_file()
    assert (REPO_ROOT / "config/storage_runtime/local_nvme.yaml").is_file()
    assert not (REPO_ROOT / "config/local_nvme_config.yaml").exists()
    assert not (REPO_ROOT / "config/tutti_config.yaml").exists()


def test_serve_rejects_unmounted_metadata_root(launch_env):
    result = subprocess.run(
        ["bash", str(SERVE_SCRIPT)], env=launch_env,
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode != 0
    assert "KV_MOUNT is not mounted" in result.stderr


def test_profile_env_preserves_gpu_and_toolchain(launch_env):
    env = {**launch_env, "CC": "chosen-cc", "CXX": "chosen-cxx"}
    command = '''source "$1"; "$TUTTI_PYTHON" -c 'import json,os
print(json.dumps({k:os.environ.get(k) for k in
("PATH","CC","CXX","CUDA_VISIBLE_DEVICES","TUTTI_PYTHON","TMPDIR")}))' '''
    result = subprocess.run(
        ["bash", "-e", "-c", command, "bash", str(PROFILE_ENV)],
        env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    for key in ("PATH", "CC", "CXX", "CUDA_VISIBLE_DEVICES", "TUTTI_PYTHON"):
        assert output[key] == env[key]
    assert output["TMPDIR"] == str(Path(env["TUTTI_PROFILE_ROOT"]) / "tmp")


@pytest.fixture
def bench_runner(launch_env, tmp_path):
    capture = tmp_path / "driver-arguments.txt"
    interpreter = tmp_path / "capture-python"
    interpreter.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$TUTTI_TEST_ARGV\"\n"
        "printf 'PYTHONPATH=%s\\n' \"$PYTHONPATH\"\n"
        "printf 'PAYLOAD_TAG=%s\\n' \"${TUTTI_E2E_PAYLOAD_TAG:-}\"\n"
    )
    interpreter.chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    mountpoint = tools / "mountpoint"
    mountpoint.write_text("#!/bin/sh\nexit 0\n")
    mountpoint.chmod(0o755)
    env = {**launch_env, "TUTTI_PYTHON": str(interpreter),
           "TUTTI_TEST_ARGV": str(capture), "PATH": f"{tools}:{launch_env['PATH']}"}
    env.pop("CAPACITY", None)
    env.pop("MAX_IN_FLIGHT", None)
    env.pop("TUTTI_E2E_PAYLOAD_TAG", None)

    def run(args, overrides=None):
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts/vllm/run/bench-8gpu-striped.sh"), *args],
            env={**env, **(overrides or {})}, text=True, capture_output=True, timeout=30,
        )
        arguments = capture.read_text().splitlines() if capture.exists() else []
        return result, arguments

    return run


@pytest.mark.parametrize("args, overrides", [
    (["--capacity", "64 GiB"], {}),
    ([], {"CAPACITY": "64 GiB"}),
])
def test_bench_passes_per_rank_capacity_without_loading_model(bench_runner, args, overrides):
    result, arguments = bench_runner(args, overrides)
    assert result.returncode == 0, result.stderr
    assert arguments[arguments.index("--capacity") + 1] == "64 GiB"
    assert "--direct-transfer-strict" not in arguments
    assert "--without-tutti" not in arguments


def test_bench_passes_minimax_block_and_load_mode(bench_runner):
    result, arguments = bench_runner(["--capacity", "8 GiB", "--block-size", "128",
                                     "--kv-load-mode", "sync"])
    assert result.returncode == 0, result.stderr
    assert arguments[arguments.index("--block-size") + 1] == "128"
    assert arguments[arguments.index("--kv-load-mode") + 1] == "sync"


@pytest.mark.parametrize("baseline", [False, True])
def test_bench_passes_minimax_model_options(bench_runner, baseline):
    flags = ["--enable-expert-parallel", "--enable-ep-weight-filter",
             "--language-model-only"]
    values = {"--kv-cache-dtype": "fp8", "--indexer-kv-dtype": "bf16",
              "--max-tokens": "1",
              "--max-model-len": "131072", "--gpu-memory-utilization": "0.8",
              "--max-num-seqs": "8", "--max-num-batched-tokens": "8192"}
    args = ["--baseline"] if baseline else ["--capacity", "8 GiB"]
    args += flags + [item for pair in values.items() for item in pair]
    args += ["--replica-io"]
    result, arguments = bench_runner(args, {"TP_SIZE": "8"})
    assert result.returncode == 0, result.stderr
    assert arguments[arguments.index("--tensor-parallel-size") + 1] == "8"
    for flag in flags:
        assert flag in arguments
    for option, value in values.items():
        assert arguments[arguments.index(option) + 1] == value
    assert ("--replica-io" in arguments) is not baseline


def test_bench_passes_large_block_chunk_and_exact_shared_tokens(bench_runner):
    result, arguments = bench_runner(["--capacity", "8 GiB", "--block-size", "512",
                                     "--chunk-tokens", "512", "--shared-tokens", "259072"])
    assert result.returncode == 0, result.stderr
    assert arguments[arguments.index("--chunk-tokens") + 1] == "512"
    assert arguments[arguments.index("--shared-tokens") + 1] == "259072"
    assert arguments[arguments.index("--block-size") + 1] == "512"


def test_bench_default_keeps_model_options_unset(bench_runner):
    result, arguments = bench_runner(["--capacity", "8 GiB"])
    assert result.returncode == 0, result.stderr
    for option in ("--enable-expert-parallel", "--enable-ep-weight-filter",
                   "--kv-cache-dtype", "--indexer-kv-dtype", "--max-model-len",
                   "--gpu-memory-utilization", "--language-model-only", "--replica-io",
                   "--max-num-seqs", "--max-num-batched-tokens"):
        assert option not in arguments
    assert arguments[arguments.index("--load-format") + 1] == "auto"
    assert arguments[arguments.index("--max-tokens") + 1] == "8"
    assert "PAYLOAD_TAG=\n" in result.stdout


@pytest.mark.parametrize("from_env", [False, True])
def test_bench_phxsafetensors_preserves_caller_pythonpath(bench_runner, from_env):
    args = ["--baseline"]
    overrides = {"PYTHONPATH": "/chosen/phxloader:/chosen/other"}
    if from_env:
        overrides["LOAD_FORMAT"] = "phxsafetensors"
    else:
        args += ["--load-format", "phxsafetensors"]
    result, arguments = bench_runner(args, overrides)
    assert result.returncode == 0, result.stderr
    assert arguments[arguments.index("--load-format") + 1] == "phxsafetensors"
    assert "/chosen/phxloader:/chosen/other" in result.stdout


def test_bench_phxsafetensors_prepends_explicit_loader(bench_runner, tmp_path):
    loader = tmp_path / "loader with spaces"
    (loader / "phxloader").mkdir(parents=True)
    (loader / "phxloader/__init__.py").write_text("")
    result, arguments = bench_runner(
        ["--baseline", "--load-format", "phxsafetensors"],
        {"PHXLOADER_DIR": str(loader), "PYTHONPATH": "/chosen/other"},
    )
    assert result.returncode == 0, result.stderr
    assert f"PYTHONPATH={loader}:" in result.stdout
    assert "/chosen/other" in result.stdout


def test_bench_rejects_missing_explicit_phxloader(bench_runner, tmp_path):
    result, arguments = bench_runner(
        ["--baseline", "--load-format", "phxsafetensors"],
        {"PHXLOADER_DIR": str(tmp_path / "missing")},
    )
    assert result.returncode != 0
    assert "phxloader missing" in result.stderr
    assert arguments == []


@pytest.mark.parametrize("from_env", [False, True])
def test_bench_passes_isolated_payload_tag(bench_runner, from_env):
    args = ["--capacity", "8 GiB", "--pool-tag", "metadata-only"]
    overrides = {}
    if from_env:
        overrides["TUTTI_E2E_PAYLOAD_TAG"] = "minimax-replica"
    else:
        args += ["--payload-tag", "minimax-replica"]
    result, arguments = bench_runner(args, overrides)
    assert result.returncode == 0, result.stderr
    assert "PAYLOAD_TAG=minimax-replica\n" in result.stdout
    assert "metadata-only-{LOCAL_RANK}" in arguments[arguments.index("--kv-root") + 1]
    assert "--payload-tag" not in arguments


def test_bench_requires_capacity_for_tutti(bench_runner):
    result, arguments = bench_runner([])
    assert result.returncode != 0
    assert "CAPACITY" in result.stderr
    assert arguments == []


def test_bench_baseline_needs_no_storage_capacity(bench_runner):
    result, arguments = bench_runner(["--baseline"])
    assert result.returncode == 0, result.stderr
    assert "--without-tutti" in arguments
    assert "--capacity" not in arguments


@pytest.fixture
def offline_driver(monkeypatch):
    captured = {}

    def llm(**kwargs):
        captured["llm"] = kwargs
        return SimpleNamespace(start_profile=lambda: None)

    def sampling(**kwargs):
        captured["sampling"] = kwargs
        return kwargs

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=object))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=llm, SamplingParams=sampling))
    monkeypatch.setitem(sys.modules, "vllm.config.kv_transfer",
                        SimpleNamespace(KVTransferConfig=lambda **kwargs: kwargs))
    monkeypatch.setenv("TUTTI_NVTX", "0")
    spec = importlib.util.spec_from_file_location(
        "tutti_offline_driver_test", REPO_ROOT / "scripts/vllm/vllm_profile_offline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_tokens", lambda *args: ([1] * 16, [1] * 16))

    def run(args):
        monkeypatch.setattr(sys, "argv", [str(spec.origin), "--model", "example/model",
                                        "--tokens", "16", "--rounds", "0", *args])
        return module.main()

    return module, run, captured


@pytest.mark.parametrize("block,chunk", [(128, 512), (256, 512), (512, 512)])
def test_offline_driver_large_blocks_and_exact_prefix(offline_driver, monkeypatch, block, chunk):
    module, run, captured = offline_driver
    seen = []
    monkeypatch.setattr(module, "_tokens", lambda *args: (seen.append(args), ([1] * 16, [1] * 16))[1])
    assert run(["--capacity", "8 GiB", "--block-size", str(block),
                "--chunk-tokens", str(chunk), "--shared-tokens", "8"]) == 0
    assert captured["llm"]["block_size"] == block
    assert captured["llm"]["kv_transfer_config"]["kv_connector_extra_config"]["chunk_tokens"] == chunk
    assert seen == [("example/model", 16, 80, 8)]


@pytest.mark.parametrize("args", [["--block-size", "512"], ["--chunk-tokens", "0"],
                                  ["--shared-tokens", "16"], ["--shared-tokens", "0"]])
def test_offline_driver_rejects_unaligned_chunk_or_invalid_exact_prefix(offline_driver, args):
    module, run, captured = offline_driver
    with pytest.raises(SystemExit):
        run(["--without-tutti", *args])
    assert "llm" not in captured


def test_exact_tokens_end_common_prefix_at_requested_boundary(offline_driver, monkeypatch):
    module, _, _ = offline_driver
    spec = importlib.util.spec_from_file_location("exact_tokens_test", REPO_ROOT / "scripts/vllm/vllm_profile_offline.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3])
    monkeypatch.setattr(original, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer))
    a, b = original._tokens("example/model", 16, 99, 8)
    assert a[:8] == b[:8] and a[8] != b[8]
    assert len(a) == len(b) == 16


def test_offline_driver_keeps_defaults_and_deterministic_sampling(offline_driver):
    module, run, captured = offline_driver
    assert run(["--capacity", "8 GiB"]) == 0
    options = captured["llm"]
    assert options["max_model_len"] == 24
    assert options["max_num_seqs"] == 256
    assert options["load_format"] == "auto"
    for name in ("enable_expert_parallel", "enable_ep_weight_filter", "kv_cache_dtype",
                 "attention_config", "gpu_memory_utilization", "language_model_only",
                 "max_num_batched_tokens"):
        assert name not in options
    extra = options["kv_transfer_config"]["kv_connector_extra_config"]
    assert "replica_io" not in extra
    assert captured["sampling"] == {"temperature": 0.0, "max_tokens": 8, "ignore_eos": True}


@pytest.mark.parametrize("flag", ["--enable-compile", "--enable-compiler"])
@pytest.mark.parametrize("graph_mode", ["NONE", "PIECEWISE"])
def test_offline_driver_compile_graph_mode(offline_driver, monkeypatch, flag, graph_mode):
    _, run, captured = offline_driver
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    assert run(["--capacity", "8 GiB", flag, "--cudagraph-mode", graph_mode]) == 0
    assert captured["llm"]["enforce_eager"] is False
    assert captured["llm"]["compilation_config"] == {"mode": 3, "cudagraph_mode": graph_mode}
    assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"


@pytest.mark.parametrize("baseline", [False, True])
@pytest.mark.parametrize("replica_io", [False, True])
def test_offline_driver_maps_minimax_options(offline_driver, baseline, replica_io):
    module, run, captured = offline_driver
    args = ["--without-tutti"] if baseline else ["--capacity", "8 GiB"]
    args += ["--tensor-parallel-size", "8", "--enable-expert-parallel",
             "--enable-ep-weight-filter", "--kv-cache-dtype", "fp8",
             "--indexer-kv-dtype", "bf16", "--max-model-len", "131072",
             "--gpu-memory-utilization", "0.8", "--language-model-only",
             "--max-num-seqs", "8", "--max-num-batched-tokens", "8192"]
    if replica_io:
        args += ["--replica-io"]
    assert run(args) == 0
    options = captured["llm"]
    assert options["tensor_parallel_size"] == 8
    assert options["enable_expert_parallel"] is True
    assert options["enable_ep_weight_filter"] is True
    assert options["kv_cache_dtype"] == "fp8"
    assert options["attention_config"] == {"indexer_kv_dtype": "bf16"}
    assert options["max_model_len"] == 131072
    assert options["max_num_seqs"] == 8
    assert options["max_num_batched_tokens"] == 8192
    assert options["gpu_memory_utilization"] == 0.8
    assert options["language_model_only"] is True
    if baseline:
        assert options["kv_transfer_config"] is None
    else:
        extra = options["kv_transfer_config"]["kv_connector_extra_config"]
        assert extra.get("replica_io", False) is replica_io


@pytest.mark.parametrize("reuse_pct", [80, 95])
@pytest.mark.parametrize("rounds", [1, 2])
def test_offline_driver_b_labels_follow_reuse_pct(
    offline_driver, monkeypatch, capsys, reuse_pct, rounds
):
    module, run, captured = offline_driver
    requests = []

    def generate(llm, prompts, sampling, request_id):
        requests.append(request_id)
        return 1.0

    monkeypatch.setattr(module, "_generate", generate)
    args = ["--without-tutti", "--rounds", str(rounds)]
    if reuse_pct != 80:
        args += ["--reuse-pct", str(reuse_pct)]
    assert run(args) == 0
    label = f"B-{reuse_pct}pct"
    assert requests[1::2] == [
        label if rounds == 1 else f"{label}|r{idx}" for idx in range(rounds)
    ]
    output = capsys.readouterr().out
    if rounds > 1:
        assert f"[SUMMARY {label}]" in output
    else:
        assert "[SUMMARY" not in output
    if reuse_pct != 80:
        assert "B-80pct" not in output


@pytest.mark.parametrize("reuse_pct", [80, 95])
def test_offline_driver_b_failure_label_follows_reuse_pct(
    offline_driver, monkeypatch, capsys, reuse_pct
):
    module, run, captured = offline_driver

    def generate(llm, prompts, sampling, request_id):
        if request_id.startswith("B-"):
            raise RuntimeError("expected stub failure")
        return 1.0

    monkeypatch.setattr(module, "_generate", generate)
    assert run(["--without-tutti", "--rounds", "2", "--reuse-pct", str(reuse_pct),
                "--expect-b-failure"]) == 0
    output = capsys.readouterr().out
    for idx in range(2):
        assert f"[B-{reuse_pct}pct|r{idx}] expected_request_failure=" in output
    if reuse_pct != 80:
        assert "B-80pct" not in output


def test_offline_driver_validates_phxsafetensors_import(offline_driver, monkeypatch, capsys):
    module, run, captured = offline_driver

    def missing(name):
        assert name == "phxloader"
        raise ImportError("missing optional loader")

    monkeypatch.setattr(module, "importlib", SimpleNamespace(import_module=missing))
    with pytest.raises(SystemExit) as error:
        run(["--without-tutti", "--load-format", "phxsafetensors"])
    assert error.value.code == 2
    assert "configure PYTHONPATH" in capsys.readouterr().err
    assert "llm" not in captured


@pytest.mark.parametrize("args, message", [
    (["--max-model-len", "23"], "must cover"),
    (["--gpu-memory-utilization", "0"], "must be in (0, 1]"),
    (["--gpu-memory-utilization", "1.1"], "must be in (0, 1]"),
])
def test_offline_driver_rejects_invalid_explicit_options(offline_driver, args, message, capsys):
    module, run, captured = offline_driver
    with pytest.raises(SystemExit) as error:
        run(["--without-tutti", *args])
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert "llm" not in captured


@pytest.fixture
def memwatch():
    spec = importlib.util.spec_from_file_location(
        "tutti_memwatch_test", REPO_ROOT / "scripts/vllm/run/memwatch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text, expected", [
    ("", {}),
    ("unrelated resident=99", {}),
    ("[tutti] health resident=3 pending=1 / 20(used)",
     {"resident": 3, "pending": 1, "capacity": 20}),
    ("[tutti] health resident=3\nother line\n[tutti] health resident=7 hit_reqs=4",
     {"resident": 7, "hit_reqs": 4}),
    ("[tutti] health resident=3\n[tutti] health no counters", {}),
])
def test_memwatch_health_uses_last_line(memwatch, text, expected):
    assert memwatch.read_health(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("", {"ckpt_total": 0.0, "ckpt_failed_total": 0.0}),
    ("DIRECT_CHECKPOINT_WRITTEN", {"ckpt_total": 1.0, "ckpt_failed_total": 0.0}),
    ("DIRECT_CHECKPOINT_WRITTEN elapsed=1.25s\nDIRECT_CHECKPOINT_FAILED\n"
     "DIRECT_CHECKPOINT_WRITTEN elapsed=0.50s",
     {"ckpt_total": 2.0, "ckpt_failed_total": 1.0, "ckpt_last_elapsed_s": 0.5}),
])
def test_memwatch_storage_stats_output_unchanged(memwatch, text, expected):
    assert memwatch.read_storage_stats(text) == expected
