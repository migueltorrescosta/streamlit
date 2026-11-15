# Miguel's dashboard

A Streamlit multipage app of small quantum-physics and estimation demos: numerical time evolution of a 1D wave packet, delta estimation and sensitivity heatmaps, Fisher information, partial traces and a comparison of scipy minimizers.

## Run

```bash
poetry install
poetry run streamlit run Home.py
```

## Test

Every page must run without an exception:

```bash
poetry run pip install pytest
PYTHONPATH=. poetry run pytest tests
```
