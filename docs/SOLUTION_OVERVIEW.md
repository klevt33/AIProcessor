# Solution Overview (Implementation Facts)

This document summarizes how the repository’s invoice-line enrichment pipeline is structured, based on the current implementation.

**Primary references**
- Pipeline stage mapping and ordering: `special_cases.yaml`
- Orchestration logic: `ai_engine.py`, `pipelines.py`, `ai_stages.py`
- Azure AI Search integration: `azure_search_utils.py`, `indexer/semantic_search_indexer.py`
- Stage thresholds and scoring configuration: `thresholds.yaml`, `confidences.yaml`

---

## 1. High-level flow

At a high level, the service:
- Receives invoice processing requests through the API layer (see `app.py`).
- Uses background processing (see `worker.py`, `invoice_extraction.py`) to execute a multi-stage enrichment pipeline per invoice line.
- Persists stage results and final results through the SQL writer (see `sql_writer.py`, `sql_utils.py`).

---

## 2. The default extraction pipeline (CASE_0)

The default stage order is defined in `special_cases.yaml` under `CASES.CASE_0.STAGES`:

1. **CLASSIFICATION**
   - Sub-stages: `DESCRIPTION_CLASSIFIER`, `LOT_CLASSIFIER`
2. **SEMANTIC_SEARCH**
   - Sub-stage: `SEMANTIC_SEARCH`
3. **COMPLETE_MATCH**
   - Sub-stage: `COMPLETE_MATCH`
4. **CONTEXT_VALIDATOR**
   - Sub-stage: `CONTEXT_VALIDATOR`
5. **FINETUNED_LLM**
   - Sub-stage: `FINETUNED_LLM`
6. **EXTRACTION_WITH_LLM_AND_WEBSEARCH**
   - Sub-stage: `AZURE_AI_AGENT_WITH_BING_SEARCH`

The configured output fields for `CASE_0` are:
- `part_number`
- `manufacturer_name`
- `unspsc`

---

## 3. Stage progression and early exit

Stage-to-stage progression is controlled by:
- The set of required output fields on the invoice detail (`ivce_dtl.fields`).
- Per-stage thresholds in `thresholds.yaml`.
- `StageUtils.check_if_next_stage_required(...)` in `ai_stages.py`, which determines whether the next stage must run.

Implementation detail:
- A stage can only stop the pipeline when **all fields** in `ivce_dtl.fields`:
  - exist in the stage output, and
  - have confidence values present, and
  - meet the configured thresholds for that stage/sub-stage.

---

## 4. Where SEMANTIC_SEARCH fits

SEMANTIC_SEARCH runs **after CLASSIFICATION and before COMPLETE_MATCH** (per `special_cases.yaml`).

From the implementation:
- SEMANTIC_SEARCH produces a **summary output** containing manufacturer name and/or UNSPSC with confidence scores.
- SEMANTIC_SEARCH also returns a small list of raw, high-similarity catalog items that are cached for downstream LLM prompt construction.
- SEMANTIC_SEARCH does **not** emit a part number field into its stage output (even though the raw catalog examples can include a manufacturer part number for LLM context).

See: `docs/STAGE_SEMANTIC_SEARCH.md`.

---

## 5. Alternative pipelines (special cases)

`special_cases.yaml` defines additional pipelines by altering stage inclusion and output fields:

- **CASE_1 (Generic pipeline)**
  - Skips `COMPLETE_MATCH`.
  - Output fields: `manufacturer_name`, `unspsc`.

- **CASE_2 (RPA LOT pipeline)**
  - Skips `CONTEXT_VALIDATOR`, `FINETUNED_LLM`, and `EXTRACTION_WITH_LLM_AND_WEBSEARCH`.
  - Output fields: `manufacturer_name`.
