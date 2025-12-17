# just_dance_controller_2p.py
"""
Strict 2-player Just Dance controller (v3.2)
- Countdown with SOUND (same as 1P)
- Pygame fullscreen with scaling (Option B)
- Hotkeys: D (webcam skeleton), F (ref skeleton), S (both)
- YOLO tracking ID lock
- Seek-based reference video (perfect sync)
- Pause-aware game clock
- Post-dance scoring (no slow live scoring)
"""

import os
import cv2
import time
import numpy as np
import pygame
from ultralytics import YOLO

from videoPoseDetection import (
    precompute_video_angles,
    keypoints_to_angles,
    ANGLE_TRIPLETS,
)

# =============================
# CONFIG
# =============================
ANGLE_TOLERANCE = 60.0
MODEL_PATH = "yolo_weights/yolov8n-pose.pt"

COUNTDOWN_VIDEO_PATH = "videos/countdown.mp4"
COUNTDOWN_SOUND_PATH = "songs_audio/countdown.mp3"

MAX_MISSING_FRAMES = 15
MAX_SCALE = 0.75        # fullscreen size scaling (0.50–0.90 recommended)

# COCO Skeleton
FULL_BODY_PAIRS = [
    (0, 1), (1, 3),
    (0, 2), (2, 4),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# =============================
# DRAW SKELETON
# =============================
def draw_skeleton(frame, kpts, color):
    if kpts is None:
        return frame

    for x, y in kpts:
        if x == 0 and y == 0:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    for a, b in FULL_BODY_PAIRS:
        x1, y1 = kpts[a]
        x2, y2 = kpts[b]
        if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
            continue
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return frame


# =============================
# MAIN 2P CONTROLLER
# =============================
class JustDanceController2P:
    def __init__(
        self,
        reference_video,
        audio_file,
        reference_angles_path,
        reference_keypoints_path,
        camera_index=0,
        countdown_video=COUNTDOWN_VIDEO_PATH,
    ):

        self.reference_video = reference_video
        self.audio_file = audio_file
        self.reference_angles_path = reference_angles_path
        self.reference_keypoints_path = reference_keypoints_path
        self.camera_index = camera_index
        self.countdown_video = countdown_video

        self.model = YOLO(MODEL_PATH)

        self.ref_angles = None
        self.ref_keypoints = None

        self.angles_p1 = []
        self.angles_p2 = []

        self.p1_id = None
        self.p2_id = None
        self.missing_frames = 0

    # ======================================================
    # PRECOMPUTE
    # ======================================================
    def precompute_reference(self):
        if os.path.exists(self.reference_angles_path) and os.path.exists(self.reference_keypoints_path):
            print("[2P] Loading reference npy…")
            self.ref_angles = np.load(self.reference_angles_path)
            self.ref_keypoints = np.load(self.reference_keypoints_path)
            return

        print("[2P] Precomputing reference…")
        angles, kpts = precompute_video_angles(self.reference_video)
        self.ref_angles = angles
        self.ref_keypoints = kpts
        np.save(self.reference_angles_path, angles)
        np.save(self.reference_keypoints_path, kpts)
        print("[2P] Saved reference files.")

    # ======================================================
    # COUNTDOWN with SOUND
    # ======================================================
    def play_countdown(self):
        played = False

        try:
            pygame.mixer.init()
            if os.path.exists(COUNTDOWN_SOUND_PATH):
                pygame.mixer.music.load(COUNTDOWN_SOUND_PATH)
                pygame.mixer.music.play()
                played = True
        except Exception as e:
            print(f"[countdown] sound error: {e}")

        if not os.path.exists(self.countdown_video):
            return

        cap = cv2.VideoCapture(self.countdown_video)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Get Ready!", frame)
            if cv2.waitKey(33) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyWindow("Get Ready!")

        if played:
            pygame.mixer.music.stop()

    # ======================================================
    # YOLO HELPERS
    # ======================================================
    def _extract_people(self, results):
        if len(results) == 0:
            return None, None
        r = results[0]
        if r.keypoints is None:
            return None, None

        xy = r.keypoints.xy
        if xy is None or xy.numel() == 0:
            return None, None

        kpts = xy.cpu().numpy()
        ids = None
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy()
        return kpts, ids

    def _lock_ids(self, ids, kpts):
        xs = kpts[:, :, 0].mean(axis=1)
        left = int(np.argmin(xs))
        right = int(np.argmax(xs))

        self.p1_id = int(ids[left])
        self.p2_id = int(ids[right])
        self.missing_frames = 0

        print(f"[2P] Locked P1={self.p1_id}, P2={self.p2_id}")

    def _get_p1_p2(self, ids, kpts):
        p1 = p2 = None
        for idx, pid in enumerate(ids):
            if int(pid) == self.p1_id:
                p1 = kpts[idx]
            elif int(pid) == self.p2_id:
                p2 = kpts[idx]
        return p1, p2, (p1 is not None and p2 is not None)

    # ======================================================
    # MAIN GAME
    # ======================================================
    def run_game(self, show_window=True):

        self.precompute_reference()
        self.play_countdown()

        # ---------------- Audio ----------------
        pygame.mixer.init()
        has_audio = False
        if os.path.exists(self.audio_file):
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()
            pygame.mixer.music.pause()
            has_audio = True

        # ---------------- Fullscreen UI ----------------
        pygame.init()
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        sw, sh = screen.get_size()
        font = pygame.font.SysFont(None, 40)

        # ---------------- Video & Camera ----------------
        cam = cv2.VideoCapture(self.camera_index)
        ref = cv2.VideoCapture(self.reference_video)

        fps = ref.get(cv2.CAP_PROP_FPS) or 30.0
        vf = int(ref.get(cv2.CAP_PROP_FRAME_COUNT))
        rf = self.ref_keypoints.shape[0]
        af = self.ref_angles.shape[0]

        num_frames = min(vf, rf, af)

        # ---------------- Timeline ----------------
        game_time = 0.0
        last_time = time.time()
        end_hold = 0.0
        paused = True
        status = "Waiting for 2 players…"

        hide_cam = False
        hide_ref = False

        last_ref_frame = None
        running = True

        # ======================================================
        # GAME LOOP
        # ======================================================
        while running:

            # ---------- Pygame Events ----------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_d:
                        hide_cam = not hide_cam
                    elif event.key == pygame.K_f:
                        hide_ref = not hide_ref
                    elif event.key == pygame.K_s:
                        both = hide_cam and hide_ref
                        hide_cam = not both
                        hide_ref = not both

            # ---------- Δ Time ----------
            now = time.time()
            dt = now - last_time
            last_time = now

            # ---------- Read Webcam ----------
            ok, cam_frame = cam.read()
            if not ok:
                break
            cam_frame = cv2.flip(cam_frame, 1)

            # ---------- YOLO Tracking ----------
            results = self.model.track(cam_frame, conf=0.3, persist=True, verbose=False)
            kpts_all, ids = self._extract_people(results)
            count = 0 if kpts_all is None else kpts_all.shape[0]

            # ---------- Lock IDs ----------
            if self.p1_id is None or self.p2_id is None:
                paused = True
                if has_audio: pygame.mixer.music.pause()

                if count == 2 and ids is not None:
                    self._lock_ids(ids, kpts_all)
                    status = "Locking players…"
                else:
                    status = "Need exactly 2 players"
                both = False
                p1 = p2 = None

            else:
                p1, p2, both = self._get_p1_p2(ids, kpts_all)

                if both and count == 2:
                    self.missing_frames = 0
                    if paused:
                        paused = False
                        if has_audio: pygame.mixer.music.unpause()
                    status = "Dancing!"
                else:
                    paused = True
                    if has_audio: pygame.mixer.music.pause()

                    self.missing_frames += 1
                    if self.missing_frames > MAX_MISSING_FRAMES:
                        print("[2P] Re-locking IDs")
                        self.p1_id = None
                        self.p2_id = None
                        self.missing_frames = 0
                    status = "Players not stable — stand side-by-side"

            # ---------- Advance timeline ----------
            if both and not paused:
                game_time += dt

            dance_idx = int(game_time * fps)
            dance_idx = max(0, min(dance_idx, num_frames - 1))

            # ---------- Check End ----------
            if dance_idx >= num_frames - 1:
                end_hold += dt
                if end_hold > 2:
                    print("[2P] Finished song.")
                    running = False
            else:
                end_hold = 0

            # ---------- Seek reference ----------
            ref.set(cv2.CAP_PROP_POS_FRAMES, dance_idx)
            ret_ref, ref_frame = ref.read()
            if not ret_ref:
                ref_frame = last_ref_frame
            else:
                last_ref_frame = ref_frame

            if not hide_ref:
                ref_frame = draw_skeleton(ref_frame.copy(), self.ref_keypoints[dance_idx], (0, 0, 255))

            # ---------- Webcam skeleton + angles ----------
            cam_draw = cam_frame.copy()
            if both and not paused:
                if not hide_cam:
                    cam_draw = draw_skeleton(cam_draw, p1, (0, 255, 0))
                    cam_draw = draw_skeleton(cam_draw, p2, (0, 255, 255))

                self.angles_p1.append(keypoints_to_angles(p1))
                self.angles_p2.append(keypoints_to_angles(p2))

            # ---------- Compose UI ----------
            h = 480
            r1 = cv2.resize(ref_frame, (int(ref_frame.shape[1] * h / ref_frame.shape[0]), h))
            r2 = cv2.resize(cam_draw, (int(cam_draw.shape[1] * h / cam_draw.shape[0]), h))
            combo = np.hstack((r1, r2))

            ch, cw = combo.shape[:2]
            scale = min(sw / cw, sh / ch)
            scale = min(scale, MAX_SCALE)

            nw, nh = int(cw * scale), int(ch * scale)
            combo_resized = cv2.resize(combo, (nw, nh))

            rgb = cv2.cvtColor(combo_resized, cv2.COLOR_BGR2RGB)
            surf = pygame.image.frombuffer(rgb.tobytes(), (nw, nh), "RGB")

            screen.fill((0, 0, 0))
            screen.blit(surf, ((sw - nw)//2, (sh - nh)//2))

            txt = font.render(status, True, (255, 255, 0))
            screen.blit(txt, (40, 40))
            pygame.display.flip()

        # ======================================================
        # CLEANUP
        # ======================================================
        cam.release()
        ref.release()
        pygame.mixer.music.stop()
        pygame.quit()

        # ======================================================
        # POST-SCORING
        # ======================================================
        if len(self.angles_p1) == 0:
            return 0.0, 0.0

        p1 = np.stack(self.angles_p1)
        p2 = np.stack(self.angles_p2)

        final_p1 = self._score_series(p1)
        final_p2 = self._score_series(p2)

        print("\n====== FINAL SCORES ======")
        print(f"Player 1: {final_p1:.2f}")
        print(f"Player 2: {final_p2:.2f}")

        return final_p1, final_p2

    # ============================
    # SERIES SCORING
    # ============================
    def _score_series(self, player):
        T = min(player.shape[0], self.ref_angles.shape[0])
        ref = self.ref_angles[:T]
        pl = player[:T]
        scores = []

        for j in range(ref.shape[1]):
            diff = np.abs(ref[:, j] - pl[:, j])
            sim = np.clip(1 - diff / ANGLE_TOLERANCE, 0, 1)
            scores.append(sim.mean() * 100)

        return float(np.mean(scores))


# =============================
# CLI ENTRY
# =============================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--audio", default="")
    p.add_argument("--angles", default="")
    p.add_argument("--keypoints", default="")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    ctrl = JustDanceController2P(
        reference_video=args.video,
        audio_file=args.audio,
        reference_angles_path=args.angles,
        reference_keypoints_path=args.keypoints,
        camera_index=args.camera,
    )
    ctrl.run_game()
