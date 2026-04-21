# BCI SSVEP (BrainFlow) — prototype scaffold

This folder contains a small, modular Python prototype for a **steady-state visual evoked potential (SSVEP)** brain–computer interface. The user attends one of two flickering on-screen targets (different frequencies, e.g. **10 Hz** and **15 Hz**); EEG is analyzed in the frequency domain and a baseline classifier predicts **LEFT** vs **RIGHT**.

## What is SSVEP?

When a visual stimulus flickers at a fixed rate, cortical activity often shows a matching rhythm in the EEG. Comparing power (or SNR) at the candidate flicker frequencies is a standard way to infer which target the user is focusing on.

## Layout

- `src/acquisition` — BrainFlow board session and labeled recording helpers
- `src/signal_processing` — bandpass filtering and windowing
- `src/features` — Welch PSD and simple band-power features at target frequencies
- `src/models` — sklearn pipeline training / save / load
- `src/realtime` — sliding-window inference and LEFT/RIGHT labels
- `notebooks` — exploration, SSVEP analysis, features, training
- `data/raw` and `data/processed` — recordings and trained artifacts

## Setup

Create a virtual environment, then from this directory (`bci-ssvep/`):

```bash
pip install -r requirements.txt
```

### Python path

Scripts and notebooks expect `bci-ssvep/src` on `PYTHONPATH`, or equivalent `sys.path` insertion (already done in `scripts/run_demo.py` and the notebooks’ first cell).

## BrainFlow synthetic board

The default board is **`BoardIds.SYNTHETIC_BOARD`**: no hardware is required. It is ideal for **testing acquisition, buffering, and the software pipeline**. It does **not** reproduce realistic SSVEP responses; for meaningful class separation, use real EEG with flickering stimuli, playback files, or your own recorded calibration data.

Minimal streaming check (optional):

```python
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

params = BrainFlowInputParams()
board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)
board.prepare_session()
board.start_stream()
# ... get_board_data() / get_current_board_data() ...
board.stop_stream()
board.release_session()
```

The project wrapper is `acquisition.brainflow_stream.BrainFlowStream`.

## Notebooks

Start Jupyter from `bci-ssvep/`:

```bash
jupyter notebook
```

Open, in order:

1. `01_data_exploration.ipynb` — stream or load data, plot time-domain EEG  
2. `02_ssvep_analysis.ipynb` — PSD / FFT-style views around target frequencies  
3. `03_feature_extraction.ipynb` — band-power feature vectors  
4. `04_model_training.ipynb` — train sklearn pipeline, save `data/processed/models/ssvep_pipeline.joblib`

## Real-time demo (CLI)

After a model exists at `data/processed/models/ssvep_pipeline.joblib` (from notebook 04):

```bash
python scripts/run_demo.py
```

Optional:

```bash
python scripts/run_demo.py --model path/to/model.joblib --max-predictions 50
```

The demo prints lines such as `prediction=LEFT (class=0)` for each analysis window.

## Notes

- Sampling rate and EEG channel indices are read from BrainFlow for the active `board_id` — keep `SSVEPConfig` frequencies aligned with your stimulus design.
- DSP is intentionally minimal (Butterworth bandpass + Welch band powers) so you can extend it (CCA, FBCCA, task-related components, etc.) without fighting the scaffold.
