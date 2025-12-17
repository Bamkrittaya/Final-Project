"""
Precompute tutorial video keypoints + full-body joint angles using YOLOv8 Pose.

Outputs (for each song):
- precomputed/<song>_reference_angles.npy      (T, J)   → angles
- precomputed/<song>_reference_keypoints.npy  (T, 17, 2) → raw keypoints
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------------------
# YOLOv8 Pose Model
# -------------------------------------------------------------
MODEL_PATH = "yolo_weights/yolov8n-pose.pt"
_MODEL = YOLO(MODEL_PATH)


# -------------------------------------------------------------
# 17-JOINT BEGINNER-FRIENDLY ANGLE SET
# -------------------------------------------------------------
# These around 14–16 angles give stable, full-body evaluation.
ANGLE_TRIPLETS = [
    # Arms
    (5, 7, 9),      # left elbow
    (6, 8, 10),     # right elbow
    (7, 5, 6),      # left upper arm vs shoulders
    (8, 6, 5),      # right upper arm vs shoulders

    # Legs
    (11, 13, 15),   # left knee
    (12, 14, 16),   # right knee
    (13, 11, 12),   # left thigh vs hips
    (14, 12, 11),   # right thigh vs hips

    # Torso / Shoulders / Hips
    (5, 6, 12),     # shoulder line angle
    (6, 5, 11),     # opposite shoulder angle
    (11, 5, 6),     # torso twist left
    (12, 6, 5),     # torso twist right
    (5, 11, 12),    # hip angle left
    (6, 12, 11),    # hip angle right

    # Head orientation vs shoulders
    (0, 5, 6),      # head → shoulders line
]


# -------------------------------------------------------------
# NEW — Video Metadata Extractor
# -------------------------------------------------------------
def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("[video-info] Could not open video:", path)
        return None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0

    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
        "duration": duration
    }


# -------------------------------------------------------------
# Angle Computation
# -------------------------------------------------------------
def _joint_angle(a, b, c):
    """
    Compute angle ABC (in degrees) given points a, b, c.
    Safe for missing/zero joints.
    """
    if np.all(a == 0) or np.all(b == 0) or np.all(c == 0):
        return 0.0

    ba = a - b
    bc = c - b

    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-6:
        return 0.0

    cos_val = np.clip(np.dot(ba, bc) / den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def keypoints_to_angles(kpts):
    """
    Convert 17 YOLO keypoints into our angle feature vector.
    Returns shape (J,) where J = len(ANGLE_TRIPLETS)
    """
    J = len(ANGLE_TRIPLETS)
    out = np.zeros(J, dtype=np.float32)

    # Safety: YOLO sometimes returns incomplete array
    if kpts is None or len(kpts) < 17:
        return out

    for idx, (i, j, k) in enumerate(ANGLE_TRIPLETS):
        out[idx] = _joint_angle(kpts[i], kpts[j], kpts[k])

    return out


# -------------------------------------------------------------
# PRECOMPUTE: Keypoints + Angles for Entire Video
# -------------------------------------------------------------
def precompute_video_angles(video_path, out_dir="precomputed"):
    """
    Process full tutorial video → save angles + keypoints.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]

    out_angles = os.path.join(out_dir, f"{base}_reference_angles.npy")
    out_keypoints = os.path.join(out_dir, f"{base}_reference_keypoints.npy")

    # Already computed?
    if os.path.exists(out_angles) and os.path.exists(out_keypoints):
        print(f"[precompute] Loaded cached → {base}")
        return np.load(out_angles), np.load(out_keypoints)

    print(f"[precompute] Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # ---------------------------------------------------------
    # NEW — Print video metadata once at start
    # ---------------------------------------------------------
    info = get_video_info(video_path)
    if info:
        print("\n=== Video Metadata ===")
        print(f"Resolution      : {info['width']} x {info['height']}")
        print(f"Frame rate (FPS): {info['fps']:.2f}")
        print(f"Total frames    : {info['frames']}")
        print(f"Duration        : {info['duration']:.2f} sec")
        print("=========================\n")

    ang_list = []
    kpt_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = _MODEL(frame, conf=0.3, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            kpts = np.zeros((17, 2), dtype=np.float32)
            angs = np.zeros(len(ANGLE_TRIPLETS), dtype=np.float32)
        else:
            xy = results[0].keypoints.xy
            if xy is None or xy.numel() == 0:
                kpts = np.zeros((17, 2), dtype=np.float32)
                angs = np.zeros(len(ANGLE_TRIPLETS), dtype=np.float32)
            else:
                kpts = xy[0].cpu().numpy()
                angs = keypoints_to_angles(kpts)

        kpt_list.append(kpts)
        ang_list.append(angs)

    cap.release()

    # Stack final arrays
    keypoints = np.stack(kpt_list)       # (T,17,2)
    angles = np.stack(ang_list)          # (T,J)

    np.save(out_angles, angles)
    np.save(out_keypoints, keypoints)

    print(f"[precompute] saved angles   → {out_angles}")
    print(f"[precompute] saved keypoints → {out_keypoints}")
    print(f"[precompute] frames: {angles.shape[0]} | joints: {angles.shape[1]}")

    return angles, keypoints


# -------------------------------------------------------------
# Optional: Only Keypoints
# -------------------------------------------------------------
def precompute_video_keypoints(video_path, out_dir="precomputed"):
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{base}_keypoints.npy")

    if os.path.exists(out_path):
        print(f"[precompute] Loaded cached keypoints → {out_path}")
        return np.load(out_path), out_path

    cap = cv2.VideoCapture(video_path)
    all_kpts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = _MODEL(frame, conf=0.3, verbose=False)

        if len(results) == 0 or results[0].keypoints is None:
            k = np.zeros((17, 2), dtype=np.float32)
        else:
            xy = results[0].keypoints.xy
            if xy is None or xy.numel() == 0:
                k = np.zeros((17, 2), dtype=np.float32)
            else:
                k = xy[0].cpu().numpy()

        all_kpts.append(k)

    cap.release()

    arr = np.stack(all_kpts)
    np.save(out_path, arr)
    print(f"[precompute] saved keypoints {arr.shape} → {out_path}")

    return arr, out_path


# -------------------------------------------------------------
# CLI Testing
# -------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()

    precompute_video_angles(args.video)
    print("Done.")
