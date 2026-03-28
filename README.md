# chitta-research

An autonomous research operating system written in Rust. Three specialized agents run a
closed loop — generate hypotheses, execute experiments, score results — against a
persistent typed belief graph that accumulates structured knowledge across sessions.

Most auto-research systems forget everything between runs and have no notion of
epistemic confidence. chitta-research maintains a graph of what it knows, what it
tested, what failed, and what those failures imply for downstream beliefs.

---

## What makes this different

**Typed belief graph, not a log file.**
The graph stores eight node types (`ResearchProgram`, `Question`, `Hypothesis`,
`ExperimentPlan`, `Run`, `Observation`, `Claim`, `Method`) connected by five
epistemic edge kinds (`Supports`, `Refutes`, `DerivedFrom`, `GeneralizesTo`,
`BlockedBy`). Every hypothesis carries prior and posterior confidence. Every claim
records which observations support it. The graph is serializable to JSON and restored
on the next run — nothing is discarded.

**6-dimensional Pareto fitness, not a scalar score.**
Experiments are scored on novelty, empirical gain, reproducibility, cost efficiency,
transfer potential, and calibration improvement. The Pareto frontier over all scored
runs is maintained continuously. No single weighting scheme is imposed; dominated
results are pruned without losing the trade-off structure.

**Reconsolidation propagates corrections.**
When a high-priority belief is corrected, confidence reductions cascade immediately to
all downstream nodes via `DerivedFrom` traversal. Claims and tier-2+ hypotheses are
reduced by 0.2, and `BlockedBy` warning edges are inserted. This is not a retrospective
audit — it happens in the same cycle.

**chitta-field native.**
Each agent writes memories, triplets, and observations to the chittad daemon over a
Unix socket using the JSON-RPC MCP protocol. The daemon's HNSW + BM25 + SDR cortical
index stores research events durably. Failed experiment outcomes lower domain recall
scores so future hypothesis generation is aware of dead ends.

**No API key required by default.**
The `claude-cli` provider delegates all LLM calls to `claude -p`, reusing the
authenticated Claude Code session. No separate API key, no billing configuration.

---

## Architecture

```
agenda.yaml
    |
    v
cr-agenda ──> BeliefGraph (cr-graph + cr-types)
                    |
    ┌───────────────┼───────────────┐
    v               v               v
 Hotr           Adhvaryu        Udgatr
 (hypotheses)   (executor)      (scorer)
    |               |               |
    +───────────────+───────────────+
                    |
            AgentContext (cr-agents)
            |       |       |       |
          LLM   Chitta  Artifacts Resources
         (cr-llm)(cr-chitta)(cr-artifacts)(cr-resources)
                    |
          chittad daemon (Unix socket)
          HNSW + BM25 + SDR cortical index

 cr-fitness     -- Pareto frontier computation
 cr-reconsolidation -- downstream confidence propagation
```

Three Tokio tasks run concurrently, each polling its own `step()` on a 100 ms loop.
The orchestrator in `cr-daemon/src/main.rs` shuts down when all agents are idle for
`--max-cycles` consecutive cycles, or the budget is exhausted, or SIGINT is received.
The final graph snapshot is written to `graph_state.json`.

---

## Quickstart

### 1. Build

```bash
# Requires Rust 1.92.0 (rust-toolchain.toml is respected automatically)
./build.sh build --release
```

### 2. Run in mock mode (no external calls)

Mock mode uses a deterministic stub LLM that returns realistic structured responses.
The entire pipeline — graph population, agent steps, artifact commits, fitness scoring,
reconsolidation — runs end-to-end without spawning `claude -p` or hitting any API.

```bash
./build.sh run --release --bin chitta-research -- \
    --agenda agenda.example.yaml \
    --mock \
    --max-cycles 5
```

After a few seconds you will see the three agents cycle through hypotheses, runs, and
scoring, then idle out. `graph_state.json` will contain the serialized belief graph and
`artifacts/` will contain git-committed result JSON files.

### 3. Run for real with claude-cli

Make sure `claude` is on your PATH (installed via Claude Code). No API key is needed.

```bash
./build.sh run --release --bin chitta-research -- \
    --agenda agenda.example.yaml \
    --max-cycles 20
```

### 4. Run with Anthropic API

Set the environment variable, then update the agenda:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

In `agenda.yaml`:

```yaml
llm:
  provider: anthropic
  model: claude-opus-4-5
  api_key_env: ANTHROPIC_API_KEY
```

---

## Agenda format

The agenda is the entry point. It declares one or more research programs, the budget,
the LLM backend, and the chitta mind path.

```yaml
programs:
  - title: "ARG prevalence in ancient permafrost"
    domain: metagenomics
    questions:
      - "Do antibiotic resistance genes increase with permafrost depth?"
      - "Are ancient ARGs phylogenetically distinct from modern variants?"
      - "What is the functional potential of recovered ARGs?"
    methods:
      - "BLASTp against CARD"
      - "Short-read assembly with MEGAHIT"
      - "HMM profiling with ResFams"
    priority: 1.0          # higher priority programs are worked first
    max_budget_usd: 5.0

  - title: "Viral diversity in Greenlandic ice cores"
    domain: metagenomics
    questions:
      - "What viral families dominate ancient ice core metagenomes?"
    priority: 0.8
    max_budget_usd: 3.0

budget:
  total_usd: 10.0   # hard ceiling; agents stop when exhausted
  gpu_slots: 0       # semaphore slots for GPU-gated steps
  cpu_workers: 4     # concurrency limit for CPU-bound steps

llm:
  provider: claude-cli       # claude-cli | anthropic | openai | codex
  model: claude-sonnet-4-6

chitta:
  mind_path: ~/.claude/mind  # path passed to chittad for socket resolution
```

Each question becomes a `Question` node. Each method becomes a `Method` node. All are
linked to their `ResearchProgram` with `DerivedFrom` or `Supports` edges at startup.
Hotr then generates `Hypothesis` + `ExperimentPlan` nodes for unanswered questions.

---

## LLM providers

| Provider | Key in YAML | What it does |
|---|---|---|
| `claude-cli` | default | Spawns `claude -p --output-format text`, reads stdout. No API key. |
| `anthropic` | `anthropic` | Direct HTTP to `api.anthropic.com`. Requires `ANTHROPIC_API_KEY`. |
| `openai` | `openai` | OpenAI-compatible HTTP. Requires `OPENAI_API_KEY`. |
| `codex` | `codex` | Spawns `codex exec -`. Model set via `--config model=`. |
| mock | `--mock` flag | Deterministic stub. For testing without any external process. |

All API providers implement exponential back-off with 3 retries on 429/503.

---

## How chitta integration works

The `cr-chitta` crate opens a Unix socket connection to chittad (the cc-soul daemon)
and speaks JSON-RPC using the MCP `tools/call` wire format. Three operations are used:

- **`remember`** — Hotr records hypothesis generation events; Adhvaryu records
  experiment outcomes with tags `[adhvaryu, experiment, succeeded|failed]`.
- **`observe`** — Reconsolidation writes structured observations under the
  `reconsolidation` category so the daemon's subconscious can process them.
- **`connect` (triplet)** — Udgatr writes `hypothesis:ID confirmed_by run:ID` (or
  `refuted_by`) with the posterior confidence as weight.

The daemon is optional. If the socket is unavailable, `ChittaClient::connect()` fails
and a warning is logged. All agent logic continues without it — chitta is an
enhancement, not a hard dependency.

---

## Crate structure

| Crate | Role |
|---|---|
| `cr-types` | Core type definitions: `NodeKind`, `EdgeKind`, `FitnessVector`, `RunStatus`, `ResourceUsage`. All `serde`-serializable. |
| `cr-graph` | `BeliefGraph` — `petgraph`-backed stable directed graph with typed lookup, evidence queries, and JSON snapshot I/O. |
| `cr-agents` | `Agent` trait + `AgentContext` + three implementations: `Hotr`, `Adhvaryu`, `Udgatr`. |
| `cr-fitness` | Pareto frontier computation over `FitnessVector` arrays. |
| `cr-llm` | `LlmClient` trait + four backends: `ClaudeCliClient`, `AnthropicClient`, `OpenAiClient`, `CodexClient`, `MockLlmClient`. |
| `cr-chitta` | Async Unix socket client for the chittad daemon. |
| `cr-reconsolidation` | Downstream confidence propagation on belief correction. |
| `cr-artifacts` | `git2`-backed artifact store — commits experiment result files to a local repo. |
| `cr-resources` | Semaphore-based GPU/CPU slot manager + budget accounting. |
| `cr-agenda` | YAML agenda parser; converts programs + questions into the initial `BeliefGraph`. |
| `cr-daemon` | Binary entry point. Parses CLI, builds context, spawns agent tasks, runs event loop. |

---

## The three agents

### Hotr — hypothesis generation

Reads the belief graph, finds the highest-priority `ResearchProgram`, then finds its
unanswered `Question` nodes (those with no `DerivedFrom` children of type
`Hypothesis`). Sends the question and program context to the LLM with a structured JSON
prompt. Each returned hypothesis is added as a `Hypothesis` node with an
`ExperimentPlan` child. Prior confidence is set by the LLM.

### Adhvaryu — experiment executor

Finds `ExperimentPlan` nodes that have no `Run` child. Sends the plan steps and
hypothesis statement to the LLM. The LLM returns structured outcome, observations, and
metrics. A `Run` node is added with status, timestamps, token/cost accounting, and a
git commit SHA pointing to the committed `results.json` artifact. `Observation` nodes
are added for each observation string.

### Udgatr — analyst and scorer

Finds `Run` nodes with `status = Succeeded` and no fitness score. Sends the hypothesis,
plan, and observations to the LLM for structured scoring on all six fitness dimensions.
Updates the run's `FitnessVector`. Updates the hypothesis posterior confidence. For
results with `empirical_gain > 0.6`, adds `Claim` nodes with `Supports` or `Refutes`
edges back to the hypothesis. Writes a triplet to chitta recording the verdict.

---

## Artifact store

Each experiment run produces a `results.json` file committed to a git repository at
`--artifact-dir` (default: `artifacts/`). Commit SHAs are stored on `Run` nodes so the
full provenance chain is: graph node → git commit → file content. The artifact repo is
initialized on first run if it does not exist.

---

## Reconsolidation

`cr-reconsolidation::trigger_reconsolidation` can be called on any node. It traverses
all `DerivedFrom` descendants, reduces the posterior confidence of tier-2+ hypotheses
and all claims by 0.2 (floored at 0), and adds `BlockedBy` edges from the corrected
node to each affected descendant. An `observe` call records the event in chitta.

This is not called automatically in the current agent loop — it is exposed as a library
function for external callers or future agent integration.

---

## Contributing

The project uses Rust 1.92.0 (specified in `rust-toolchain.toml`). Use `./build.sh`
instead of calling `cargo` directly — it unsets the conda linker override that can
cause link failures on some HPC systems.

```bash
./build.sh test          # run all tests
./build.sh clippy        # lint
./build.sh build         # debug build
./build.sh build --release
```

Tests cover: node uniqueness, Pareto dominance edge cases, graph snapshot roundtrip,
belief graph construction from YAML, LLM mock responses, and agent integration via
`MockLlmClient`.

New LLM backends implement the `LlmClient` async trait in `cr-llm/src/lib.rs`.
New agent roles implement the `Agent` async trait in `cr-agents/src/lib.rs` and are
spawned in `cr-daemon/src/main.rs`.

---

## Design notes

The three agent names — Hotr, Adhvaryu, Udgatr — are the three primary priests of
the Vedic yajna (ritual fire). Hotr recites invocations; Adhvaryu performs the
physical ritual actions; Udgatr chants the responses. The research loop mirrors this
structure: invocation (hypothesis), execution (experiment), evaluation (scoring).

The fitness vector's six dimensions were chosen to avoid the common failure mode of
optimizing for novelty alone. Calibration improvement specifically rewards experiments
that update the system's confidence in proportion to actual evidence — an agent that
consistently over- or under-predicts posterior confidence will score low here.

chitta-research was designed in a three-model room debate (GPT-5.4 + Gemini-2.5-Pro +
Claude Sonnet) to stress-test the architecture against adversarial critique before a
line of code was written.
