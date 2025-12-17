import os
import subprocess
import numpy as np
import cv2

from videoPoseDetection import precompute_video_angles, precompute_video_keypoints
from path_helper import get_fast_paths


# ===============================
# CONFIG — EASY SPEED CONTROL
# ===============================

VIDEO_SPEED = 1       # 3× faster video
AUDIO_DELAY_MS = 0       # ms delay


# ===============================
# FFmpeg Helpers
# ===============================

def run_ffmpeg(cmd):
    print("\n[FFmpeg] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ===============================
# VIDEO INFO HELPER
# ===============================

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("[error] Cannot open video:", path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
    }


# ===============================
# MAIN OPTIMIZER
# ===============================

def optimize_song(original_video_path):
    """
    Automatically creates fast video, fast audio, angles, keypoints.
    Prints original + fast video info.
    """

    fast_video, fast_audio, fast_angles, fast_keypoints = get_fast_paths(original_video_path)

    base = os.path.splitext(os.path.basename(original_video_path))[0]

    # --- folders ---
    os.makedirs("videos_fast", exist_ok=True)
    os.makedirs("songs_audio_fast", exist_ok=True)
    os.makedirs("precomputed", exist_ok=True)

    # ===============================
    # SHOW ORIGINAL VIDEO INFO
    # ===============================

    print("\n========== ORIGINAL VIDEO INFO ==========")
    info_orig = get_video_info(original_video_path)
    if info_orig:
        print(f"Resolution      : {info_orig['width']} × {info_orig['height']}")
        print(f"Frame Rate (FPS): {info_orig['fps']:.2f}")
        print(f"Total Frames    : {info_orig['frames']}")
        print(f"Duration        : {info_orig['duration']:.2f} sec")
    print("=========================================")

    # ===============================
    # 1. FAST VIDEO
    # ===============================

    if not os.path.exists(fast_video):
        setpts_value = 1.0 / VIDEO_SPEED
        setpts_str = f"setpts={setpts_value}*PTS"
        filter_v = f"{setpts_str},scale=640:-1"

        cmd_video = [
            "ffmpeg", "-y",
            "-i", original_video_path,
            "-filter:v", filter_v,
            "-preset", "ultrafast",
            fast_video
        ]
        run_ffmpeg(cmd_video)
    else:
        print(f"[skip] Fast video exists → {fast_video}")

    # ===============================
    # 2. FAST AUDIO
    # ===============================

    if not os.path.exists(fast_audio):

        temp_wav = f"songs_audio_fast/{base}_temp.wav"
        run_ffmpeg([
            "ffmpeg", "-y",
            "-i", original_video_path,
            "-vn",
            temp_wav
        ])

        audio_filter = f"adelay={AUDIO_DELAY_MS}|{AUDIO_DELAY_MS}"

        run_ffmpeg([
            "ffmpeg", "-y",
            "-i", temp_wav,
            "-af", audio_filter,
            fast_audio
        ])

        os.remove(temp_wav)

    else:
        print(f"[skip] Fast audio exists → {fast_audio}")

    # ===============================
    # 3. PRECOMPUTE KEYPOINTS
    # ===============================

    if not os.path.exists(fast_keypoints):
        kpts, _ = precompute_video_keypoints(fast_video)
        np.save(fast_keypoints, kpts)
        print(f"[save] {fast_keypoints}")
    else:
        print(f"[skip] Keypoints exist → {fast_keypoints}")

    # ===============================
    # 4. PRECOMPUTE ANGLES
    # ===============================

    if not os.path.exists(fast_angles):
        ang, _ = precompute_video_angles(fast_video)
        np.save(fast_angles, ang)
        print(f"[save] {fast_angles}")
    else:
        print(f"[skip] Angles exist → {fast_angles}")


    # ===============================
    # SHOW FAST VIDEO INFO
    # ===============================

    print("\n========== FAST VIDEO INFO ==========")
    info_fast = get_video_info(fast_video)
    if info_fast:
        print(f"Resolution      : {info_fast['width']} × {info_fast['height']}")
        print(f"Frame Rate (FPS): {info_fast['fps']:.2f}")
        print(f"Total Frames    : {info_fast['frames']}")
        print(f"Duration        : {info_fast['duration']:.2f} sec")
    print("=====================================")

    # ===============================
    # PRINT SHAPES OF PRECOMPUTED FILES
    # ===============================

    if os.path.exists(fast_keypoints):
        k = np.load(fast_keypoints)
        print(f"Keypoints shape : {k.shape}")   # (T,17,2)

    if os.path.exists(fast_angles):
        a = np.load(fast_angles)
        print(f"Angles shape    : {a.shape}")   # (T,J)

    print("\n✅ Optimization complete for:", original_video_path)
