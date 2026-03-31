# MedIC Drone Resupply Engine

This repository contains a local-first Python backend for planning medical drone resupply missions for combat medics treating traumatic brain injury patients in contested environments.

The service accepts a simplified medic-system patient payload, normalizes the incoming assessment, evaluates a clinical workbook policy, optionally runs a local LLM recommender through Ollama or Hugging Face, packs the resulting supplies into 10 lb drone manifests, applies redundancy based on drone shootdown risk, and returns a dispatch plan that an operator can review.

## What The Codebase Does

- Accepts a medic-system patient profile over a local HTTP API.
- Preserves the external payload structure while normalizing it into an internal case model.
- Evaluates workbook-authored clinical policy from structured fields such as GCS, suspected ICP, external hemorrhage, and MARCH flags.
- Uses the local supply catalog plus the clinical workbook [data/TBI_Clinical_Rules.xlsx](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/data/TBI_Clinical_Rules.xlsx) as the recommendation source of truth.
- Supports an optional local-only LLM recommender layer through Ollama or Hugging Face, with deterministic validation and fallback if the model is unavailable.
- Builds a required supply list from matching rules.
- Packs supplies into base drone manifests without exceeding the per-drone payload limit.
- Calculates how many replicated drone sets to send based on shootdown rate and target arrival confidence.
- Stores generated plans and operator decisions in a local SQLite database.
- Writes a human-readable operator plan brief to the local `exports/` directory for each created or recalculated plan.

## Main Components

- [resupply_engine/api.py](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/resupply_engine/api.py)
  FastAPI entrypoints for creating, retrieving, recalculating, and approving plans.
- [resupply_engine/ingest.py](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/resupply_engine/ingest.py)
  Transport adapters that make the planner independent of whether input arrives via HTTP or another ingestion path.
- [resupply_engine/models.py](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/resupply_engine/models.py)
  Pydantic models for the medic payload, canonical internal case, supplies, manifests, and plans.
- [resupply_engine/normalize.py](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/resupply_engine/normalize.py)
  Mapping from the raw burst payload into the normalized planner-facing case.
- [resupply_engine/workbook_policy.py](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/resupply_engine/workbook_policy.py)
  Workbook compiler and deterministic policy evaluator for clinical item selection.
- [resupply_engine/llm_recommender.py](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/resupply_engine/llm_recommender.py)
  Optional local-only Ollama and Hugging Face recommender adapters plus structured output parser.
- [resupply_engine/packing.py](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/resupply_engine/packing.py)
  Packing logic that assigns supplies to base drone manifests under the payload limit.
- [resupply_engine/redundancy.py](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/resupply_engine/redundancy.py)
  Redundancy math based on shootdown rate and target arrival probability.
- [resupply_engine/service.py](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/resupply_engine/service.py)
  Main orchestration layer for planning, recalculation, review flags, and persistence.
- [data/supply_catalog.csv](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/data/supply_catalog.csv)
  Sendable items and their weights.
- [data/TBI_Clinical_Rules.xlsx](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/data/TBI_Clinical_Rules.xlsx)
  Clinical workbook used for workbook-driven item policy and ranking.
- [data/example_payload.json](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/data/example_payload.json)
  Example full planning request posted to the API.
- [data/incoming_payload.json](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/data/incoming_payload.json)
  Example inner medic-system payload before the operator adds planning fields.

## Current Input And Output

The primary endpoint is `POST /v1/plans`.

Input:
- `shootdown_rate`
- `target_arrival_probability`
- optional planner-side `burst_id`
- `software_decision_support_payload` containing the medic assessment

Output:
- required supplies to send
- base drone manifests and what each drone carries
- redundancy multiplier
- total drones to dispatch
- achieved arrival confidence
- review flags
- human-readable plan summary
- human-readable `.txt` operator brief written under `exports/<plan_id>/`

## Local Setup

This project is designed to run fully locally.

1. Create a virtual environment if you do not already have one:

```bash
python3 -m venv .venv
```

2. Activate it:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -e ".[dev]"
```

To enable the optional local Hugging Face recommender layer, install the extra too:

```bash
pip install -e ".[dev,llm]"
```

If you plan to use Ollama only, the base project dependencies are enough because the Ollama backend talks to the local Ollama HTTP API directly.

If you prefer not to use editable installs, this also works:

```bash
pip install fastapi uvicorn pydantic pytest httpx
```

## Optional Ollama Setup

The current codebase defaults to `llm_backend="ollama"` when `llm_enabled=True`.

1. Install Ollama on your machine.

On macOS, use the official download:

- Ollama download: https://ollama.com/download/mac

2. Start the local Ollama service.

If the desktop app is installed, launching it usually starts the local service. You can also run:

```bash
ollama serve
```

3. Pull a model locally.

For this project, a good first model is:

```bash
ollama pull qwen2.5:1.5b
```

If your machine can handle a larger model, you can also try:

```bash
ollama pull qwen2.5:3b
```

4. Verify that Ollama is responding:

```bash
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "qwen2.5:1.5b",
  "prompt": "Return only JSON: {\"ok\": true}",
  "stream": false
}'
```

5. Configure the planner to use Ollama.

The relevant settings in [config.py](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/resupply_engine/config.py) are:

- `llm_enabled=True`
- `llm_backend="ollama"`
- `ollama_base_url="http://127.0.0.1:11434/api"`
- `ollama_model="qwen2.5:1.5b"`

With that configuration, the planner sends the `CanonicalCase`, supply catalog, and compiled clinical workbook to the local Ollama server and expects structured JSON supply recommendations back.

## Optional Hugging Face Setup

If you prefer not to run Ollama, the project can also load a local Hugging Face Transformers model directory directly from Python.

1. Install the extra dependencies:

```bash
pip install -e ".[dev,llm]"
```

2. Download a compatible local model directory.

3. Configure:

- `llm_enabled=True`
- `llm_backend="huggingface"`
- `llm_model_path=<path to the downloaded model directory>`

Hugging Face mode loads model files directly into the Python process. Ollama mode talks to a separate local model server.

## Run Locally

Start the API server from the repo root:

```bash
source .venv/bin/activate
uvicorn resupply_engine.api:app --reload
```

Or without activating the virtual environment:

```bash
.venv/bin/uvicorn resupply_engine.api:app --reload
```

Once the server is running:

- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

After a successful `POST /v1/plans` or plan recalculation, the backend writes a readable text brief to:

```text
MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/exports/<plan_id>/mission-<mission_id>__burst-<burst_id>__rev-001.txt
```

Later recalculations create new revisions such as `rev-002.txt` in the same plan folder.

## Run Tests

Run the test suite:

```bash
.venv/bin/pytest -q
```

Run a quick syntax/compile check:

```bash
python3 -m compileall resupply_engine tests backend.py
```

## Example Request Flow

1. Start the FastAPI server.
2. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
3. Expand `POST /v1/plans`.
4. Click `Try it out`.
5. Paste a request body using the shape from [data/example_payload.json](/Users/chelseahuynh/MedIC-DHA-33-Heads-Up-Medics-Drone-Resupply/data/example_payload.json), wrapped inside:

```json
{
  "shootdown_rate": 25,
  "target_arrival_probability": 0.95,
  "burst_id": "BST-20340316-0001",
  "software_decision_support_payload": {
    "...": "..."
  }
}
```

6. Click `Execute`.
7. Review the response fields:

- `required_supplies`
- `base_manifests`
- `redundancy_multiplier`
- `total_drones`
- `per_item_arrival_probability`
- `summary_text`
- `export.export_path`

## Notes

- The current recommendation engine is workbook-driven with an optional local LLM layer.
- Ollama is now the default LLM backend path when local LLMs are enabled.
- The free-text extractor is only a bounded helper for symptom extraction; it does not replace clinician-authored rules.
- Packing and redundancy remain deterministic even when the local LLM recommender is enabled.
- The medic payload is field-only; planning knobs like `shootdown_rate` stay at the top level of the API request.
- The current implementation is backend-first. The FastAPI docs page is a testing interface, not the final operator dashboard.
