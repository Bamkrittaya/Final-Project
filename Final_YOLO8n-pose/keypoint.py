"""
Dictionary that maps from joint names to keypoint indices.
"""
import numpy as np


KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDICES_TO_COLOR = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}

def normalize_pose(pose):
    """
    Normalize pose keypoints for scale and position.
    This removes the effects of player distance or frame size.

    Args:
        pose (np.ndarray): Array of shape (17, 3) with (x, y, confidence)

    Returns:
        np.ndarray: Normalized keypoints, same shape as input
    """
    if pose is None or len(pose) == 0:
        return np.zeros((17, 3))

    pose = np.array(pose, dtype=np.float32)
    
    # Use mid-hip as origin (keypoints 11 & 12)
    if pose.shape[0] > 12:
        mid_hip = (pose[11, :2] + pose[12, :2]) / 2.0
    else:
        mid_hip = np.mean(pose[:, :2], axis=0)
    
    # Center the pose
    pose[:, :2] -= mid_hip

    # Compute scale (distance between shoulders)
    if pose.shape[0] > 6:
        left_shoulder, right_shoulder = pose[5, :2], pose[6, :2]
        scale = np.linalg.norm(left_shoulder - right_shoulder)
    else:
        scale = np.linalg.norm(pose[:, :2].ptp(axis=0))

    if scale < 1e-6:
        scale = 1.0

    pose[:, :2] /= scale
    return pose
