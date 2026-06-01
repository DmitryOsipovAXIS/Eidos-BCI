# EIDOS Eye Tracking + BCI Frontend

Gaze-controlled navigation for the EIDOS accessibility prototype. A Python tracker detects look direction and sends events over WebSocket; the HTML UI highlights buttons and activates them after a 1-second center gaze (dwell).

## Architecture

```
┌─────────────────┐     WebSocket (JSON)      ┌──────────────────────┐
│  main.py        │  {"gaze":"LEFT"|...}     │  bci-prototype.html  │
│  OpenCV +       │ ────────────────────────► │  + js/gaze-client.js │
│  MediaPipe      │      ws://127.0.0.1:8765  │  + gaze-navigation.js│
└─────────────────┘                           └──────────────────────┘
```

- **`gaze_server.py`** — background WebSocket server; broadcasts to all connected browsers.
- **`main.py`** — camera loop, gaze classification, sends JSON **only when gaze state changes**.
- **`js/gaze-client.js`** — connects and parses messages.
- **`js/gaze-navigation.js`** — focus highlighting and 1 s dwell-to-click.
- **`js/app.js`** — screen state and button actions (shared by mouse and gaze).

## Gaze → UI behavior

| Screen | Look left | Look right | Center (1 s) |
|--------|-----------|------------|--------------|
| Splash | Focus ENTER | Focus EXIT | Activate focused button |
| Main menu | Focus NEXT | Focus SELECT | Activate focused button |
| Category | Focus BACK | Focus SELECT/PLAY/CALL | Activate focused button |

WebSocket payloads:

```json
{"gaze": "LEFT"}
{"gaze": "RIGHT"}
{"gaze": "CENTER"}
```

## Setup

### 1. Python environment

> **Windows: the venv path must contain only ASCII characters.**
> MediaPipe's native model loader cannot open files from paths containing
> non-ASCII characters (e.g. Cyrillic like `Проекты`). If the `mediapipe`
> package ends up under such a path, FaceMesh fails with
> `FileNotFoundError: ... face_landmark_front_cpu.binarypb` even though the
> file exists. Put the **virtual environment** on an ASCII-only path; the
> project source itself can stay where it is.

```bash
cd eye-tracking

# Create the venv on an ASCII-only path (Windows example).
# --system-site-packages reuses already-installed global packages.
py -m venv --system-site-packages C:\Users\<you>\eidos-venv
C:\Users\<you>\eidos-venv\Scripts\activate

# macOS / Linux (any path is fine)
# python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

Requires a webcam. MediaPipe FaceMesh with iris refinement is used (`mediapipe==0.10.14`).
`main.py` checks the MediaPipe install path on startup and prints clear guidance if it sits on a non-ASCII path.

### 2. Start the gaze tracker (includes WebSocket server)

```bash
python main.py
```

Options:

| Flag | Description |
|------|-------------|
| `--host` | Bind address (default `127.0.0.1`) |
| `--port` | Port (default `8765`) |
| `--camera` | Camera index (default `0`) |
| `--no-window` | Run without OpenCV preview |
| `--no-server` | Tracker only, no WebSocket |
| `-v` | Debug logging |

Press **Esc** in the preview window to quit.

### 3. Serve the frontend

ES modules require HTTP (not `file://`). From the `eye-tracking` folder:

```bash
python -m http.server 8080
```

Open in a browser:

**http://127.0.0.1:8080/bci-prototype.html**

The status pill shows `GAZE · CONNECTING` until the tracker is running, then returns to the screen label (e.g. `MAIN MENU`).

## Typical workflow

1. Terminal 1: `python main.py` — start tracker + WebSocket.
2. Terminal 2: `python -m http.server 8080` — serve the UI.
3. Browser: open `http://127.0.0.1:8080/bci-prototype.html`.
4. Look **left** / **right** to move focus; hold **center** for 1 second to confirm (spinning ring shows progress).

## Calibration

Iris thresholds are in `main.py` (`left_thr=0.42`, `right_thr=0.58`). Adjust if left/right detection feels inverted or insensitive.

## Files

| File | Role |
|------|------|
| `main.py` | Gaze detection + WebSocket emission |
| `gaze_server.py` | WebSocket broadcaster |
| `bci-prototype.html` | UI markup and styles |
| `js/app.js` | Application state and actions |
| `js/gaze-client.js` | WebSocket client |
| `js/gaze-navigation.js` | Gaze focus and dwell logic |
| `requirements.txt` | Python dependencies |
