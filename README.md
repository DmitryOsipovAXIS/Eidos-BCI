# Eidos-BCI

A simple, two-button **brain–computer interface (BCI)** for people who are paralyzed
or otherwise unable to use their hands. The user looks at one of two flickering
boxes on the screen; the system reads their brain activity (EEG), figures out
which box they are attending to, and turns that into a **LEFT** or **RIGHT**
command. Those two commands are enough to drive a simple on-screen menu (move
focus / confirm), so the user can call for help, play music, watch a movie, etc.

The goal is deliberately minimal: **two reliable choices** beat a fancy interface
that doesn't work for someone who can only signal "this one" or "that one".

---

## How it works (SSVEP in one minute)

This project uses **SSVEP** — *Steady-State Visual Evoked Potentials*.

When you stare at something flickering at a fixed rate (say **7.5 Hz**), the
visual cortex starts producing electrical activity at that **same** rate. If two
boxes flicker at two different frequencies (e.g. LEFT = 7.5 Hz, RIGHT = 12 Hz),
we can record the EEG and check which frequency is strongest in the brain signal.
Whichever one wins tells us which box the person is looking at.

The signal flow is:

```
 Flickering boxes  ──►  user's eyes/brain  ──►  EEG headset
        │                                            │
        │ (7.5 Hz left, 12 Hz right)                 │ (8 channels @ 125 Hz)
        ▼                                            ▼
   pygame screen  ◄── live LEFT/RIGHT label ──  signal pipeline
                                                 (filter → CCA classify)
                                                     │
                                                     ▼
                                        WebSocket / HTTP events
                                                     │
                                                     ▼
                                           web UI (two buttons)
```

The classifier is **CCA (Canonical Correlation Analysis)**. It needs **no
training data**: it just compares the recorded EEG against ideal sine waves at
each target frequency and picks the best match. This keeps the whole system
"very simple" and robust.

---

## Repository layout

This repo contains **two independent sub-projects** that solve the same problem
("control two buttons hands-free") with two different input methods:

| Folder | Input method | What it is |
|--------|--------------|------------|
| `bci-ssvep/` | **EEG / brain signals** (SSVEP) | The main BCI backend |
| `eye-tracking/` | **Webcam gaze** (look left / right) | A simpler, hardware-free alternative + the shared web frontend |
| `bci-prototype.html` | — | The standalone two-button demo UI |

Both backends speak the same idea to the frontend over a WebSocket, so the UI
doesn't care whether the "LEFT/RIGHT" command came from a brain or from an eye.

### `bci-ssvep/` — the EEG backend

The core of the project. A small, modular Python pipeline that streams EEG,
classifies SSVEP, draws the flickering stimuli, and broadcasts decisions.

```
bci-ssvep/
├── pyproject.toml              # dependencies (brainflow, numpy, scipy, sklearn, pygame, websockets)
├── uv.lock                     # locked dependency versions (uv package manager)
├── README.md                   # backend-specific notes
└── src/
    ├── server.py               # ENTRY POINT: starts WebSocket + HTTP + live SSVEP together
    │
    ├── acquisition/            # talking to the EEG hardware
    │   ├── brainflow_stream.py #   thin wrapper around BrainFlow board (start/stop/read)
    │   └── recording.py        #   board init (Neuropawn Knight @125 Hz) + labeled recording
    │
    ├── pipeline/               # turning raw EEG into a LEFT/RIGHT decision
    │   ├── preprocessing.py    #   detrend + common-average reference (CAR)
    │   ├── signal_filtering.py #   50 Hz notch + 3–30 Hz Butterworth bandpass
    │   ├── cca_classifier.py   #   CCA: match EEG to sine references, no training needed
    │   ├── pipeline.py         #   offline pipeline: preprocess → classify (one epoch / batch)
    │   └── realtime_pipeline.py#   sliding-window live classifier + majority vote + WS broadcast
    │
    ├── features/               # frequency-domain features (Welch PSD, band power) — analysis/debug
    │   └── frequency_features.py
    │
    ├── realtime/
    │   └── pipeline.py         #   "start_live": menu, modes (single/alt/live), wiring it all up
    │
    ├── ui/                     # the on-screen flickering stimuli (pygame)
    │   ├── screens.py          #   start menu + single / alternating / live / one-box screens
    │   └── drawing.py          #   low-level draw helpers (boxes, text, FPS, inference overlay)
    │
    ├── ws/                     # how decisions leave the backend
    │   ├── websockets.py       #   WebSocket server on :8765 (push LEFT/RIGHT to the UI)
    │   └── http.py             #   simple HTTP endpoint on :8080 (manual WIDGET/FUNCTION events)
    │
    └── utils/
        └── config.py           # central config: frequencies, bands, sample rate, paths
```

Other items:
- `bci-ssvep/scripts/` — standalone helper scripts for recording, visualizing data,
  PSD/epoch plots, CCA accuracy, and a PsychoPy live demo. Useful for debugging and
  tuning, not required to run the system.

**How a decision is made (live):** `realtime_pipeline.py` grabs a sliding window
of EEG, runs `pipeline.py` (CAR → notch → bandpass → CCA), takes a **majority vote**
over the last few windows for stability, and only emits a command when it's
**confident** (winner clearly beats runner-up). The decision is sent over the
WebSocket as a JSON message (`LEFT` → `FUNCTION`, `RIGHT` → `WIDGET`).

### `eye-tracking/` — gaze alternative + shared frontend

A camera-only version of the same two-button idea, plus the web UI that both
backends drive. No EEG hardware needed — it tracks the iris with MediaPipe and
reports `LEFT` / `RIGHT` / `CENTER`.

```
eye-tracking/
├── main.py                 # webcam loop: detect gaze direction, send on state change
├── gaze_server.py          # WebSocket broadcaster (ws://127.0.0.1:8765)
├── requirements.txt        # opencv, mediapipe, websockets, numpy
├── bci-prototype.html      # the two-button accessibility UI
├── README.md               # detailed eye-tracking setup & calibration
└── js/
    ├── app.js              # screen state + button actions (splash → menu → category)
    ├── gaze-client.js      # connects to the WebSocket, parses gaze messages
    └── gaze-navigation.js  # focus highlighting + 1-second "dwell to click"
```

The frontend logic: **look left/right** to move focus between the two buttons,
**hold center for 1 second** (dwell) to activate the focused button.

### Root files

- `bci-prototype.html` — the standalone click-only version of the two-button UI
  (good for showing the interface without any backend running).
- `.gitignore` — ignores the local virtualenv.

---

## Running it

### Option A — EEG backend (`bci-ssvep`)

Requires the EEG headset (defaults to a **Neuropawn Knight** board at 125 Hz; the
serial port is set in `server.py`, e.g. `COM4`). Dependencies are managed with
[`uv`](https://github.com/astral-sh/uv).

```bash
cd bci-ssvep
uv sync                 # install dependencies from pyproject.toml / uv.lock
uv run python src/server.py
```

This launches three things at once: the WebSocket server (`:8765`), the HTTP
event endpoint (`:8080`), and the live SSVEP screen + classifier. Stare at a box
and watch the LEFT/RIGHT decisions appear.

> No headset? The BrainFlow `SYNTHETIC_BOARD` lets you test the software path
> without hardware, but it does **not** produce realistic SSVEP, so it won't give
> meaningful LEFT/RIGHT separation — it's only for checking the plumbing.

### Option B — Eye-tracking (no special hardware, just a webcam)

```bash
cd eye-tracking
pip install -r requirements.txt
python main.py                       # terminal 1: gaze tracker + WebSocket
python -m http.server 8080           # terminal 2: serve the UI
```

Then open **http://127.0.0.1:8080/bci-prototype.html** in a browser.

> **Windows note:** MediaPipe cannot load its model from a non-ASCII path
> (e.g. a folder with Cyrillic letters). Put the virtualenv on an ASCII-only path.
> See `eye-tracking/README.md` for the full explanation and calibration tips.

---

## Tuning the important knobs

- **Stimulus frequencies** — `LEFT = 7.5 Hz`, `RIGHT = 12 Hz` by default
  (`realtime/pipeline.py`, also configurable via the start menu / CLI flags).
- **Confidence thresholds** — `left_threshold`, `right_threshold`, `min_score`
  control how sure the classifier must be before sending a command.
- **Window / step** — `window_s` (analysis window length) and `step_s` (how often
  it re-classifies) trade off responsiveness vs. stability.
- **Filtering** — 50 Hz notch (European mains) + 3–30 Hz bandpass in
  `signal_filtering.py`; switch the notch to 60 Hz in North America.
- **Gaze sensitivity** — `left_thr` / `right_thr` in `eye-tracking/main.py`.
