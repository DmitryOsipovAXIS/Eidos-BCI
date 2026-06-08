"""Gaze direction tracker with WebSocket output for the EIDOS frontend."""

from __future__ import annotations

import argparse
import logging
import os

import cv2
import numpy as np

try:
    import mediapipe as mp

    mp_face_mesh = mp.solutions.face_mesh
except Exception:
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    except Exception:
        from mediapipe.tasks.python.vision import face_landmarker as mp_face_mesh

from gaze_server import GazeWebSocketServer

LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# Maps internal labels to JSON payloads expected by the frontend.
LABEL_TO_GAZE = {
    "LOOKING LEFT": "LEFT",
    "LOOKING RIGHT": "RIGHT",
    "CENTER": "CENTER",
}


def landmark_px(landmarks, idx, w, h):
    p = landmarks[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def iris_ratio(landmarks, iris_ids, eye_outer, eye_inner, w, h):
    iris_center = np.mean([landmark_px(landmarks, i, w, h) for i in iris_ids], axis=0)
    outer = landmark_px(landmarks, eye_outer, w, h)
    inner = landmark_px(landmarks, eye_inner, w, h)
    eye_vec = inner - outer
    eye_len2 = float(np.dot(eye_vec, eye_vec))
    if eye_len2 < 1e-6:
        return 0.5
    return float(np.dot(iris_center - outer, eye_vec) / eye_len2)


def gaze_label(ratio, left_thr=0.42, right_thr=0.58):
    if ratio < left_thr:
        return "LOOKING LEFT"
    if ratio > right_thr:
        return "LOOKING RIGHT"
    return "CENTER"


def check_mediapipe_path() -> bool:
    """MediaPipe's native loader fails on non-ASCII install paths (Windows).

    Returns True if the install path is safe, otherwise prints guidance and returns False.
    """
    module_file = getattr(mp_face_mesh, "__file__", None)
    if not module_file:
        return True
    mp_dir = os.path.dirname(os.path.abspath(module_file))
    try:
        mp_dir.encode("ascii")
    except UnicodeEncodeError:
        print("=" * 70)
        print("ERROR: MediaPipe is installed under a non-ASCII path:")
        print(f"  {mp_dir}")
        print()
        print("MediaPipe's native model loader cannot open files from paths with")
        print("non-ASCII characters (e.g. Cyrillic), so FaceMesh fails to start.")
        print()
        print("Fix: use a Python/venv located on an ASCII-only path. For example:")
        print('  py -m venv --system-site-packages C:\\Users\\<you>\\eidos-venv')
        print('  C:\\Users\\<you>\\eidos-venv\\Scripts\\activate')
        print("  python main.py")
        print("=" * 70)
        return False
    return True


def run_tracker(server: GazeWebSocketServer | None, camera_index: int = 0, show_window: bool = True) -> None:
    if not hasattr(mp_face_mesh, "FaceMesh"):
        print("Your MediaPipe build does not support FaceMesh. Install: pip install mediapipe==0.10.14")
        return

    if not check_mediapipe_path():
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Camera not found")
        return

    last_sent_gaze: str | None = None

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            label = "NO FACE"
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                left_ratio = iris_ratio(lm, LEFT_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER, w, h)
                right_ratio = iris_ratio(lm, RIGHT_IRIS, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, w, h)
                ratio = (left_ratio + right_ratio) / 2.0
                label = gaze_label(ratio)

                for idx in LEFT_IRIS + RIGHT_IRIS:
                    x, y = landmark_px(lm, idx, w, h).astype(int)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                gaze = LABEL_TO_GAZE.get(label)
                if gaze is not None and gaze != last_sent_gaze:
                    last_sent_gaze = gaze
                    if server is not None:
                        server.send_gaze(gaze)
                        logging.debug("Sent gaze: %s", gaze)
            else:
                last_sent_gaze = None

            if show_window:
                cv2.putText(
                    frame,
                    label,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Gaze Direction", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            elif label == "NO FACE":
                pass

    cap.release()
    if show_window:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="EIDOS gaze tracker with WebSocket output")
    parser.add_argument("--host", default="127.0.0.1", help="WebSocket bind host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket bind port")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--no-window", action="store_true", help="Run without OpenCV preview")
    parser.add_argument("--no-server", action="store_true", help="Run tracker only (no WebSocket)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    server = None
    if not args.no_server:
        server = GazeWebSocketServer(host=args.host, port=args.port)
        server.start()
        print(f"WebSocket server: ws://{args.host}:{args.port}")

    try:
        run_tracker(server, camera_index=args.camera, show_window=not args.no_window)
    finally:
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
