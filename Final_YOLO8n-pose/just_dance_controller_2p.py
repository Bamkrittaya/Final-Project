"""
Strict 2-player Just Dance controller (v3.6)

✔ Single master timeline (same as 1P)
✔ Async YOLO tracking (smooth sync feel)
✔ Time-indexed angle capture (no drift)
✔ ID-safe handling (no None crashes)
✔ show_window compatible with 1P
✔ D / F / S skeleton visibility toggles
✔ Skeleton status UX text
✔ Offline post-scoring only
"""

import os
import cv2
import time
import queue
import threading
import numpy as np
import pygame
from ultralytics import YOLO

from videoPoseDetection import (
    precompute_video_angles,
    keypoints_to_angles,
)

# =============================
# CONFIG
# =============================
ANGLE_TOLERANCE = 60.0
MODEL_PATH = "yolo_weights/yolov8n-pose.pt"

COUNTDOWN_VIDEO_PATH = "videos/countdown.mp4"
COUNTDOWN_SOUND_PATH = "songs_audio/countdown.mp3"

MAX_MISSING_FRAMES = 15
MAX_SCALE = 0.75

# COCO skeleton
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
    if frame is None or kpts is None:
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
# ASYNC YOLO THREAD
# =============================
class YOLOThread2P:
    def __init__(self, model, conf=0.3):
        self.model = model
        self.conf = conf
        self.q = queue.Queue(maxsize=1)
        self.running = True

        self.latest_kpts_all = None
        self.latest_ids = None
        self.latest_count = 0

        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    def submit(self, frame):
        if frame is None:
            return
        if self.q.full():
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put(frame)

    def _worker(self):
        while self.running:
            try:
                frame = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                results = self.model.track(
                    frame, conf=self.conf, persist=True, verbose=False
                )
            except Exception:
                continue

            kpts_all, ids, count = None, None, 0
            if results and results[0].keypoints is not None:
                xy = results[0].keypoints.xy
                if xy is not None and xy.numel() > 0:
                    kpts_all = xy.cpu().numpy()
                    count = kpts_all.shape[0]
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy()

            self.latest_kpts_all = kpts_all
            self.latest_ids = ids
            self.latest_count = count

    def stop(self):
        self.running = False

# =============================
# MAIN CONTROLLER
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

        self.angles_p1 = {}
        self.angles_p2 = {}

        self.p1_id = None
        self.p2_id = None
        self.missing_frames = 0

    # ======================================================
    # PRECOMPUTE
    # ======================================================
    def precompute_reference(self):
        if os.path.exists(self.reference_angles_path) and os.path.exists(self.reference_keypoints_path):
            self.ref_angles = np.load(self.reference_angles_path)
            self.ref_keypoints = np.load(self.reference_keypoints_path)
            return

        angles, kpts = precompute_video_angles(self.reference_video)
        self.ref_angles = angles
        self.ref_keypoints = kpts
        np.save(self.reference_angles_path, angles)
        np.save(self.reference_keypoints_path, kpts)

    # ======================================================
    # COUNTDOWN
    # ======================================================
    def play_countdown(self, show_window=True):
        played = False
        try:
            pygame.mixer.init()
            if os.path.exists(COUNTDOWN_SOUND_PATH):
                pygame.mixer.music.load(COUNTDOWN_SOUND_PATH)
                pygame.mixer.music.play()
                played = True
        except Exception:
            pass

        if show_window and os.path.exists(self.countdown_video):
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
    # PLAYER HELPERS
    # ======================================================
    def _lock_ids(self, ids, kpts_all):
        xs = kpts_all[:, :, 0].mean(axis=1)
        self.p1_id = int(ids[np.argmin(xs)])
        self.p2_id = int(ids[np.argmax(xs)])
        self.missing_frames = 0

    def _get_players(self, ids, kpts_all):
        if ids is None or kpts_all is None:
            return None, None, False

        p1 = p2 = None
        for i, pid in enumerate(ids):
            if int(pid) == self.p1_id:
                p1 = kpts_all[i]
            elif int(pid) == self.p2_id:
                p2 = kpts_all[i]
        return p1, p2, (p1 is not None and p2 is not None)

    # ======================================================
    # MAIN GAME LOOP
    # ======================================================
    def run_game(self, show_window=True):
        self.precompute_reference()
        self.play_countdown(show_window)

        pygame.init()
        try:
            pygame.mixer.init()
        except Exception:
            pass

        has_audio = False
        if self.audio_file and os.path.exists(self.audio_file):
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()
            pygame.mixer.music.pause()
            has_audio = True

        screen = None
        if show_window:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            sw, sh = screen.get_size()
            font = pygame.font.SysFont(None, 36)

        cam = cv2.VideoCapture(self.camera_index)
        ref = cv2.VideoCapture(self.reference_video)

        fps = ref.get(cv2.CAP_PROP_FPS) or 30.0
        num_frames = min(
            int(ref.get(cv2.CAP_PROP_FRAME_COUNT)),
            len(self.ref_angles),
            len(self.ref_keypoints),
        )

        yolo = YOLOThread2P(self.model)

        game_time = 0.0
        last_time = time.time()
        paused = True
        status = "Waiting for 2 players…"
        end_hold = 0.0

        hide_cam_skel = True
        hide_ref_skel = True

        last_ref_frame = None
        running = True

        while running:
            # ---------- Events ----------
            if show_window:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                        elif event.key == pygame.K_d:
                            hide_cam_skel = not hide_cam_skel
                        elif event.key == pygame.K_f:
                            hide_ref_skel = not hide_ref_skel
                        elif event.key == pygame.K_s:
                            both = hide_cam_skel and hide_ref_skel
                            hide_cam_skel = not both
                            hide_ref_skel = not both

            now = time.time()
            dt = now - last_time
            last_time = now

            ok, cam_frame = cam.read()
            if not ok:
                break
            cam_frame = cv2.flip(cam_frame, 1)

            yolo.submit(cam_frame)

            kpts_all = yolo.latest_kpts_all
            ids = yolo.latest_ids
            count = yolo.latest_count or 0

            if self.p1_id is None or self.p2_id is None:
                paused = True
                if has_audio:
                    pygame.mixer.music.pause()
                if count == 2 and ids is not None:
                    self._lock_ids(ids, kpts_all)
                    status = "Players locked"
                else:
                    status = "Need 2 players + tracking IDs"
                p1 = p2 = None
                both_players = False
            else:
                p1, p2, both_players = self._get_players(ids, kpts_all)
                if both_players and count == 2:
                    paused = False
                    if has_audio:
                        pygame.mixer.music.unpause()
                    status = "Dancing!"
                    self.missing_frames = 0
                else:
                    paused = True
                    if has_audio:
                        pygame.mixer.music.pause()
                    self.missing_frames += 1
                    status = "Players unstable — stand side-by-side"
                    if self.missing_frames > MAX_MISSING_FRAMES:
                        self.p1_id = self.p2_id = None
                        self.missing_frames = 0

            if not paused:
                game_time += dt

            dance_idx = int(game_time * fps)
            dance_idx = max(0, min(dance_idx, num_frames - 1))

            if dance_idx >= num_frames - 1:
                end_hold += dt
                if end_hold > 2.0:
                    running = False
            else:
                end_hold = 0.0

            ref.set(cv2.CAP_PROP_POS_FRAMES, dance_idx)
            ok_ref, ref_frame = ref.read()
            if ok_ref:
                last_ref_frame = ref_frame
            else:
                ref_frame = last_ref_frame
                if ref_frame is None:
                    break

            if both_players and not paused:
                self.angles_p1[dance_idx] = keypoints_to_angles(p1)
                self.angles_p2[dance_idx] = keypoints_to_angles(p2)

            if show_window:
                # reference draw
                if hide_ref_skel:
                    ref_draw = ref_frame.copy()
                else:
                    ref_draw = draw_skeleton(
                        ref_frame.copy(),
                        self.ref_keypoints[dance_idx],
                        (0, 0, 255),
                    )

                cam_draw = cam_frame.copy()
                if both_players and not hide_cam_skel:
                    cam_draw = draw_skeleton(cam_draw, p1, (0, 255, 0))
                    cam_draw = draw_skeleton(cam_draw, p2, (0, 255, 255))

                h = 480
                r1 = cv2.resize(ref_draw, (int(ref_draw.shape[1] * h / ref_draw.shape[0]), h))
                r2 = cv2.resize(cam_draw, (int(cam_draw.shape[1] * h / cam_draw.shape[0]), h))
                combo = np.hstack((r1, r2))

                ch, cw = combo.shape[:2]
                scale = min(sw / cw, sh / ch, MAX_SCALE)
                combo = cv2.resize(combo, (int(cw * scale), int(ch * scale)))

                rgb = cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)
                surf = pygame.image.frombuffer(rgb.tobytes(), combo.shape[1::-1], "RGB")

                screen.fill((0, 0, 0))
                screen.blit(surf, ((sw - combo.shape[1]) // 2, (sh - combo.shape[0]) // 2))

                icon = "🟢 Skeletons ON"
                if hide_cam_skel and hide_ref_skel:
                    icon = "🔴 ALL OFF"
                elif hide_cam_skel:
                    icon = "🟡 Webcam OFF"
                elif hide_ref_skel:
                    icon = "🔵 Reference OFF"

                screen.blit(font.render(status, True, (255, 255, 0)), (40, 40))
                screen.blit(font.render(icon, True, (255, 255, 255)), (40, 80))
                pygame.display.flip()

        yolo.stop()
        cam.release()
        ref.release()
        if has_audio:
            pygame.mixer.music.stop()
        pygame.quit()

        return self._final_scores()

    # ======================================================
    # SCORING
    # ======================================================
    def _score_from_dict(self, angle_dict):
        if not angle_dict:
            return 0.0
        idxs = np.array(sorted(angle_dict.keys()), dtype=np.int32)
        player = np.stack([angle_dict[i] for i in idxs], axis=0)
        ref = self.ref_angles[idxs]
        diff = np.abs(ref - player)
        sim = np.clip(1.0 - diff / ANGLE_TOLERANCE, 0.0, 1.0)
        return float(sim.mean() * 100.0)

    def _final_scores(self):
        p1 = self._score_from_dict(self.angles_p1)
        p2 = self._score_from_dict(self.angles_p2)
        print("\n====== FINAL SCORES ======")
        print(f"Player 1: {p1:.2f}")
        print(f"Player 2: {p2:.2f}")
        return p1, p2
