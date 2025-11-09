# InferenceMAX Dashboard

The `dashboard` package bundles all Streamlit applications, reusable view
components, and analytical helpers required to explore InferenceMAX serving
scenarios.  The module hierarchy is intentionally self-contained so the entire
folder can be copied into a fresh repository and operated as a standalone
project.

## Directory layout

- `app.py` and `inferencemax.py` – legacy entry points that assemble the
  composite dashboard experience.
- `components/` – shared Streamlit widgets, headers, and sidebar logic.
- `features/` – domain specific helper classes used across multiple pages.
- `models/` – light wrappers that adapt raw model configuration dictionaries for
  visualisation.
- `services/` – pure-Python analytics such as FLOP estimators, bandwidth models,
  and the chunked prefill planning module.
- `llm_chunked_prefill_decoder_scaleup.py` – the single-page scale-up explorer
  requested in the specification; runnable on its own with `streamlit run`.

## Getting started

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

   ```bash
   pip install streamlit plotly pydantic numpy pandas
   ```

3. Launch one of the dashboards, for example the scale-up explorer:

   ```bash
   streamlit run llm_chunked_prefill_decoder_scaleup.py
   ```

   The Streamlit CLI should be executed from inside the `dashboard` directory so
   that local relative imports resolve correctly.

4. To embed the dashboard in a larger project, add the `dashboard` directory to
   your Python path or package it as a module using your preferred build system.

## Testing

The analytical helpers ship with lightweight unit tests that validate critical
behaviour without needing GPU access.  From the repository root, run:

```bash
pytest dashboard/tests/test_llm_chunked_prefill_decoder_scaleup.py \
    dashboard/tests/test_services_llm_calcs.py
```

These tests rely only on the pure-Python modules under `dashboard/services` and
can be executed in any standard Python 3.10+ environment.
