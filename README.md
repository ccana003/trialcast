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

## Enhancements in this version

- **Multi-model training with model selection**
  - The trainer now evaluates both ElasticNet and GradientBoostingRegressor.
  - Cross-validation MAE is used to select the winning model for each target.
- **Missing-data indicator features**
  - `has_feasibility_data`, `has_recruitment_data`, and `has_protocol_data` are generated during ingestion and included in training.
- **Data-driven risk classification**
  - Risk labels are based on the training target distribution (bottom 30% = High risk, middle 40% = Moderate, top 30% = Low).
- **Monte Carlo recruitment simulation**
  - In addition to deterministic estimates, the simulator now supports month-by-month stochastic simulation to produce median, P80, and P90 duration forecasts and completion probabilities at 12 and 24 months.
- **Feature review workflow in UI**
  - Users can run extraction, inspect and edit extracted protocol features, and then run simulation.
- **Model explanation panel**
  - When ElasticNet is selected, the app displays top positive and negative coefficient drivers for enrollment probability.

## Notes

- The MVP uses rule-based eligibility parsing for interpretability.
- Models are designed to tolerate missing values and partial study coverage across source systems.
- Future phases can swap in LLM-based extraction and richer model explainability.
