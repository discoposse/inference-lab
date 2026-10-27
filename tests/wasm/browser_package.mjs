// Browser-package checks for the wasm-pack `--target web` build.
//
// Run after `wasm-pack build --target web --no-default-features --out-dir pkg`
// and a release CLI build (`cargo build --release`). From the repo root:
//
//   node tests/wasm/browser_package.mjs
//
// The golden run is the README's B200 example:
// configs/qwen3.6-35b-a3b-fp8.toml --hardware b200 --workload
// workloads/chat-closed-256.toml. CLI and WASM TTFT p50 must agree within
// 0.1 percent.

import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import init, {
  catalog_json,
  run_simulation,
  run_simulation_streaming,
} from '../../pkg/inference_lab.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const pkgDir = join(root, 'pkg');

const golden = {
  hardware: 'b200',
  model: 'qwen3.6-35b-a3b-fp8',
  parallel: { tp: 1 },
  scheduler: {
    max_num_batched_tokens: 16384,
    max_num_seqs: 4096,
    policy: 'priority',
    enable_chunked_prefill: true,
    block_size: 64,
    gpu_memory_utilization: 0.8,
  },
  workload: {
    arrival_pattern: 'closed_loop',
    num_concurrent_users: 256,
    closed_loop_jitter_secs: 0.05,
    num_requests: 2000,
    seed: 7,
    input_len_dist: { type: 'lognormal', mean: 7.0, std_dev: 0.5 },
    output_len_dist: { type: 'lognormal', mean: 6.5, std_dev: 0.8 },
  },
};

const tiered = {
  hardware: 'b200',
  model: 'qwen3.5-4b',
  replicas: 2,
  scheduler: {
    max_num_batched_tokens: 8192,
    max_num_seqs: 128,
    policy: 'fcfs',
    enable_chunked_prefill: true,
    block_size: 16,
  },
  memory: { tiers: ['host_dram'] },
  workload: {
    arrival_pattern: 'poisson',
    arrival_rate: 5,
    num_requests: 8,
    seed: 3,
    input_len_dist: { type: 'fixed', value: 128 },
    output_len_dist: { type: 'fixed', value: 16 },
  },
};

const inlineSessions = {
  hardware: 'b200',
  model: 'qwen3.5-4b',
  scheduler: tiered.scheduler,
  workload: {
    sessions: [
      {
        id: 's1',
        steps: [
          { input: 32, new: 32, output: 8, gap: 0 },
          { input: 48, new: 8, output: 8, gap: 0.01 },
        ],
      },
    ],
    num_sessions: 1,
    arrival_pattern: 'poisson',
    arrival_rate: 1,
    seed: 1,
    input_len_dist: { type: 'fixed', value: 1 },
    output_len_dist: { type: 'fixed', value: 1 },
  },
};

function assert(cond, message) {
  if (!cond) {
    throw new Error(message);
  }
}

function assertClose(label, a, b, tol = 0.001) {
  const denom = Math.max(Math.abs(a), Math.abs(b), 1e-9);
  const rel = Math.abs(a - b) / denom;
  assert(rel <= tol, `${label}: ${a} vs ${b} (relative ${rel}, limit ${tol})`);
}

function asArray(value, label) {
  assert(value != null, `${label} missing`);
  return Array.from(value);
}

function cliSummary() {
  const bin = process.env.INFERENCE_LAB_BIN || join(root, 'target', 'release', 'inference-lab');
  const dir = mkdtempSync(join(tmpdir(), 'ilab-'));
  const out = join(dir, 'summary.json');
  const result = spawnSync(
    bin,
    [
      '--config',
      join(root, 'configs', 'qwen3.6-35b-a3b-fp8.toml'),
      '--hardware',
      'b200',
      '--workload',
      join(root, 'workloads', 'chat-closed-256.toml'),
      '--output',
      out,
      '--quiet',
    ],
    { cwd: root, encoding: 'utf8' },
  );
  if (result.status !== 0) {
    throw new Error(
      `CLI exited ${result.status}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`,
    );
  }
  return JSON.parse(readFileSync(out, 'utf8'));
}

function checkCatalog(catalog) {
  assert(Array.isArray(catalog.gpus), 'catalog.gpus');
  assert(Array.isArray(catalog.hardware), 'catalog.hardware');
  assert(Array.isArray(catalog.models), 'catalog.models');
  assert(
    JSON.stringify(catalog.gpus) === JSON.stringify(catalog.hardware),
    'gpus and hardware presets differ',
  );
  const hardwareIds = catalog.hardware.map((entry) => entry.id);
  const modelIds = catalog.models.map((entry) => entry.id);
  for (const id of ['b200', 'h100', 'm5-pro']) {
    assert(hardwareIds.includes(id), `missing hardware preset ${id}`);
  }
  assert(modelIds.includes('qwen3.5-4b'), 'missing model preset qwen3.5-4b');
  assert(modelIds.includes('qwen3.6-35b-a3b-fp8'), 'missing model preset qwen3.6-35b-a3b-fp8');
  const m5 = catalog.hardware.find((entry) => entry.id === 'm5-pro').preset;
  assert(m5.name === 'M5 Pro', `m5 name ${m5.name}`);
  assertClose('m5 bandwidth', m5.memory_bandwidth, 307e9, 1e-12);
  assert(m5.memory_capacity === 24000000000, `m5 capacity ${m5.memory_capacity}`);
  assert(m5.flops_bf16 === undefined, 'm5 preset invented a FLOP rate');
  assert(m5.memory === undefined, 'm5 preset invented a memory graph');
  assert(m5.fabric === undefined, 'm5 preset invented a fabric');
  for (const entry of [...catalog.hardware, ...catalog.models]) {
    assert(typeof entry.id === 'string' && entry.id.length > 0, 'preset id');
    assert(entry.preset && typeof entry.preset === 'object', `${entry.id} preset`);
  }
}

function checkStreaming(config) {
  const parts = [];
  const streamed = run_simulation_streaming(JSON.stringify(config), (progress) => {
    parts.push(progress);
  });
  const once = run_simulation(JSON.stringify(config));
  assertClose('streaming ttft p50', streamed.metrics.ttft_p50, once.metrics.ttft_p50, 0);
  assert(parts.length >= 2, `expected several progress events, got ${parts.length}`);

  const deltaTimes = parts.flatMap((part) => asArray(part.time_series.times, 'delta times'));
  const finalTimes = asArray(streamed.time_series.times, 'final times');
  assert(
    deltaTimes.length === finalTimes.length,
    `time-series deltas (${deltaTimes.length}) do not concatenate to the final series (${finalTimes.length})`,
  );
  for (let i = 0; i < finalTimes.length; i += 1) {
    assert(deltaTimes[i] === finalTimes[i], `time series point ${i} was resent or dropped`);
  }
  const total = streamed.metrics.total_time;
  assert(total > 5, `golden run too short to check the 1s sample interval (${total}s)`);
  assert(
    finalTimes.length <= total + 5,
    `time series has ${finalTimes.length} points over ${total}s; expected about one per second`,
  );
  return streamed;
}

function checkDetail(result) {
  const summary = result.summary;
  assert(summary, 'result.summary');
  assert(summary.router && typeof summary.router.policy === 'string', 'router policy');
  assert(asArray(summary.router.per_replica, 'per_replica').length === 2, 'per-replica counts');
  const pools = asArray(summary.hbm.pools, 'hbm pools');
  assert(pools.length > 0, 'hbm pools');
  const workers = asArray(pools[0].workers, 'hbm workers');
  assert(workers.length > 0, 'per-worker hbm');
  assert(typeof workers[0].resident_prefix_bytes === 'number', 'resident prefix bytes');
  assert(typeof workers[0].running === 'number', 'worker running');
  const memory = summary.memory;
  assert(memory, 'summary.memory');
  const stores = asArray(memory.stores, 'stores');
  const links = asArray(memory.links, 'links');
  assert(stores.some((store) => store.name === 'host_dram'), 'host_dram store');
  assert(links.length > 0, 'links');
  assert(typeof links[0].bytes_moved === 'number', 'link bytes');
  assert(typeof links[0].utilisation === 'number', 'link utilisation');
}

const wasmBytes = readFileSync(join(pkgDir, 'inference_lab_bg.wasm'));
await init({ module_or_path: wasmBytes });

const catalogText = catalog_json();
const catalog = JSON.parse(catalogText);
checkCatalog(catalog);
writeFileSync(join(pkgDir, 'catalog.json'), catalogText);
const written = JSON.parse(readFileSync(join(pkgDir, 'catalog.json'), 'utf8'));
assert(JSON.stringify(written.hardware) === JSON.stringify(catalog.hardware), 'catalog.json hardware');
assert(JSON.stringify(written.models) === JSON.stringify(catalog.models), 'catalog.json models');

const cli = cliSummary();
const wasmGolden = checkStreaming(golden);
assertClose(
  'CLI vs WASM TTFT p50',
  cli.latency_metrics.ttft_ms.p50,
  wasmGolden.metrics.ttft_p50,
);
assertClose(
  'summary vs headline TTFT p50',
  wasmGolden.summary.latency_metrics.ttft_ms.p50,
  wasmGolden.metrics.ttft_p50,
  0,
);
assert(wasmGolden.summary.requests.completed === 2000, 'golden completed requests');
assert(wasmGolden.summary.memory == null, 'untiered golden should not invent memory tiers');
assert(wasmGolden.summary.router.per_replica.length === 1, 'golden router replicas');
assert(wasmGolden.summary.hbm.pools[0].workers.length > 0, 'golden per-worker detail');

checkDetail(run_simulation(JSON.stringify(tiered)));

const inline = run_simulation(JSON.stringify(inlineSessions));
assert(inline.summary.requests.completed === 2, `inline sessions completed ${inline.summary.requests.completed}`);
assert(inline.summary.sessions.started === 1, 'inline session did not start');
assert(inline.metrics.ttft_p50 === inline.summary.latency_metrics.ttft_ms.p50, 'inline ttft');

console.log(
  `browser package ok: TTFT p50 ${wasmGolden.metrics.ttft_p50} ms, ` +
    `${catalog.hardware.length} hardware presets, ${catalog.models.length} model presets`,
);
