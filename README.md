# Recruitment Feasibility and Simulation Platform (Prototype)

This repository contains an MVP architecture for simulating study recruitment feasibility from historical institutional data.

## Project structure

```text
src/recruitment_feasibility/
  common/
    schemas.py
  data_ingestion/
    loader.py
  feature_extraction/
    eligibility_parser.py
  model_training/
    trainer.py
  simulation_engine/
    simulator.py
  ui_application/
    streamlit_app.py
```

## Quick start

1. Add CSV files in `data/` following `data/README.md`.
2. Install dependencies:
   - pandas
   - numpy
   - scikit-learn
   - streamlit
3. Run the UI:

```bash
PYTHONPATH=src streamlit run src/recruitment_feasibility/ui_application/streamlit_app.py
```

## Notes

- The MVP uses rule-based eligibility parsing for interpretability.
- Models are designed to tolerate missing values and partial study coverage across source systems.
- Future phases can swap in LLM-based extraction and richer model explainability.
