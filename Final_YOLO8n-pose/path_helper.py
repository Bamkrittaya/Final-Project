# path_helper.py
import os

def get_fast_paths(original_video_path: str):
    """
    original_video_path: e.g. 'videos/cheapthrills.mp4'

    Returns:
        fast_video, fast_audio, fast_angles, fast_keypoints
    """
    base = os.path.splitext(os.path.basename(original_video_path))[0]

    fast_video = f"videos_fast/{base}_fast.mp4"
    fast_audio = f"songs_audio_fast/{base}_fast.mp3"
    fast_angles = f"precomputed/{base}_fast_reference_angles.npy"
    fast_keypoints = f"precomputed/{base}_fast_reference_keypoints.npy"

    return fast_video, fast_audio, fast_angles, fast_keypoints
