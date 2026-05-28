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


LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]


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


def main():
    if not hasattr(mp_face_mesh, "FaceMesh"):
        print("Your MediaPipe build does not support FaceMesh. Install: pip install mediapipe==0.10.14")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found")
        return

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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
