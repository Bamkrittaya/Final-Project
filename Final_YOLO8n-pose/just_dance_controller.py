# just_dance_controller.py
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
    ANGLE_TRIPLETS,
)

# ============================================================
# CONFIG
# ============================================================
ANGLE_TOLERANCE = 60
MODEL_PATH = "yolo_weights/yolov8n-pose.pt"

COUNTDOWN_VIDEO_PATH = "videos/countdown.mp4"
COUNTDOWN_SOUND_PATH = "songs_audio/countdown.mp3"

# Skeleton pairs (COCO 17)
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

# ============================================================
# ASYNC YOLO DETECTOR THREAD
# ============================================================
class YOLOThread:
    """
    Runs YOLO pose detection in a separate thread.
    Main loop just submits frames and reads latest keypoints.
    """
    def __init__(self, model, conf=0.3):
        self.model = model
        self.conf = conf
        self.queue = queue.Queue(maxsize=1)
        self.running = True

        # latest results (read from main thread)
        self.latest_keypoints = None   # np.ndarray shape (17, 2) or None
        self.latest_num_people = 0

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, frame_bgr):
        """
        Submit a frame for detection (non-blocking).
        Oldest unprocessed frame is dropped → we always use freshest.
        """
        if frame_bgr is None:
            return
        if self.queue.full():
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put(frame_bgr)

    def _worker(self):
        while self.running:
            try:
                frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                results = self.model(frame, conf=self.conf, verbose=False)
            except Exception as e:
                print(f"[YOLOThread] Inference error: {e}")
                continue

            kpts = None
            num_people = 0
            if results and results[0].keypoints is not None:
                xy = results[0].keypoints.xy
                if xy is not None and xy.numel() > 0:
                    num_people = xy.shape[0]
                    # first person only
                    kpts = xy[0].cpu().numpy()

            self.latest_keypoints = kpts
            self.latest_num_people = num_people

    def stop(self):
        self.running = False


# ============================================================
# DRAW SKELETON
# ============================================================
def draw_skeleton(frame, keypoints, color=(0, 255, 0)):
    if keypoints is None or frame is None:
        return frame

    for x, y in keypoints:
        if x == 0 and y == 0:
            continue
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    for a, b in FULL_BODY_PAIRS:
        x1, y1 = keypoints[a]
        x2, y2 = keypoints[b]
        if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
            continue
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return frame


# ============================================================
# MAIN CONTROLLER
# ============================================================
class JustDanceController:
    def __init__(
        self,
        reference_video: str,
        audio_file: str,
        reference_angles_path: str,
        reference_keypoints_path: str,
        camera_index: int = 0,
        countdown_video: str = COUNTDOWN_VIDEO_PATH,
        countdown_sound: str = COUNTDOWN_SOUND_PATH,
    ):
        self.reference_video = reference_video
        self.audio_file = audio_file
        self.reference_angles_path = reference_angles_path
        self.reference_keypoints_path = reference_keypoints_path
        self.camera_index = camera_index
        self.countdown_video = countdown_video
        self.countdown_sound = countdown_sound

        # YOLO pose model (shared with YOLOThread)
        self.model = YOLO(MODEL_PATH)

        self.ref_angles = None
        self.ref_keypoints = None
        self.player_angles = []   # we record angles, score later

    # ---------------------------------------------------------
    # PRECOMPUTE
    # ---------------------------------------------------------
    def precompute_reference(self):
        if os.path.exists(self.reference_angles_path) and os.path.exists(self.reference_keypoints_path):
            print("[precompute] Loading reference data…")
            self.ref_angles = np.load(self.reference_angles_path)
            self.ref_keypoints = np.load(self.reference_keypoints_path)
            return

        print("[precompute] Computing reference from video…")
        angles, kpts = precompute_video_angles(self.reference_video)
        self.ref_angles = angles
        self.ref_keypoints = kpts

        np.save(self.reference_angles_path, angles)
        np.save(self.reference_keypoints_path, kpts)
        print("[precompute] Saved reference npy files.")

    # ---------------------------------------------------------
    # COUNTDOWN
    # ---------------------------------------------------------
    def play_countdown(self, show_window=True):
        try:
            pygame.mixer.init()
        except pygame.error:
            pass

        played = False

        if os.path.exists(self.countdown_sound):
            try:
                pygame.mixer.music.load(self.countdown_sound)
                pygame.mixer.music.play()
                played = True
            except Exception as e:
                print(f"[countdown] Failed to play sound: {e}")

        if not show_window:
            return

        if not os.path.exists(self.countdown_video):
            print("[countdown] Missing video.")
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

    # ---------------------------------------------------------
    # SCORING HELPERS
    # ---------------------------------------------------------
    @staticmethod
    def score_joint(ref, player):
        """
        ref, player: 1D arrays over time for ONE joint angle.
        """
        T = min(len(ref), len(player))
        ref, player = ref[:T], player[:T]

        mask = ~((ref == 0) | (player == 0))
        if mask.sum() == 0:
            return 0.0

        diff = np.abs(ref[mask] - player[mask])
        sim = np.clip(1.0 - diff / ANGLE_TOLERANCE, 0, 1)
        return float(sim.mean() * 100)

    def score_total_series(self, ref_series, player_series):
        """
        ref_series, player_series: (T, J) arrays (time, joint_angles)
        Returns a single final score.
        """
        T = min(ref_series.shape[0], player_series.shape[0])
        ref = ref_series[:T]
        player = player_series[:T]

        scores = []
        for j in range(ref.shape[1]):
            scores.append(self.score_joint(ref[:, j], player[:, j]))
        return float(np.mean(scores))

    # ---------------------------------------------------------
    # GAME LOOP
    # ---------------------------------------------------------
    def run_game(self, show_window=True):
        # Load / compute reference
        self.precompute_reference()

        if self.ref_keypoints is None or len(self.ref_keypoints) == 0:
            print("[error] No reference keypoints loaded.")
            return 0.0

        if self.ref_angles is None or len(self.ref_angles) == 0:
            print("[error] No reference angles loaded.")
            return 0.0

        self.play_countdown(show_window)

        pygame.init()
        try:
            pygame.mixer.init()
        except pygame.error:
            pass

        # Load audio
        has_audio = False
        if self.audio_file and os.path.exists(self.audio_file):
            try:
                pygame.mixer.music.load(self.audio_file)
                pygame.mixer.music.play()
                pygame.mixer.music.pause()
                has_audio = True
            except Exception as e:
                print(f"[audio] Failed to load: {e}")

        # Display
        if show_window:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            screen_w, screen_h = screen.get_size()
            font = pygame.font.SysFont(None, 40)
        else:
            screen = None
            screen_w = screen_h = 0
            font = None

        # Webcam
        cap_cam = cv2.VideoCapture(self.camera_index)

        # Reference video (seek-based, synced with keypoints)
        ref_cap = cv2.VideoCapture(self.reference_video)
        if not ref_cap.isOpened():
            print("[error] Cannot open reference video:", self.reference_video)
            return 0.0

        video_fps = ref_cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0  # fallback
        print(f"[info] Reference video FPS: {video_fps:.2f}")

        # frame count should roughly match npy length
        video_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_ref_frames = min(video_frames, len(self.ref_keypoints), len(self.ref_angles))
        print(f"[info] Ref frames (video vs npy): video={video_frames}, npy={len(self.ref_keypoints)} → using {num_ref_frames}")

        # Async YOLO thread
        yolo_thread = YOLOThread(self.model, conf=0.3)

        # PAUSE-AWARE GAME CLOCK
        game_time = 0.0          # only advances when 1 person detected
        last_loop_time = time.time()
        end_hold_time = 0.0      # time spent on final frame

        paused = True            # logical pause (no dancer)
        status_text = "Waiting…"

        hide_webcam_skel = True
        hide_ref_skel = True

        last_ref_frame = None
        running = True

        while running:
            # --------------- EVENTS -----------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_d:
                        hide_webcam_skel = not hide_webcam_skel
                    elif event.key == pygame.K_f:
                        hide_ref_skel = not hide_ref_skel
                    elif event.key == pygame.K_s:
                        both_off = hide_webcam_skel and hide_ref_skel
                        hide_webcam_skel = not both_off
                        hide_ref_skel = not both_off

            # --------------- TIME STEP (dt) -----------------
            now = time.time()
            dt = now - last_loop_time
            last_loop_time = now

            # --------------- WEBCAM -----------------
            ret_cam, frame_cam = cap_cam.read()
            if not ret_cam:
                print("[camera] Failed to read frame.")
                break

            frame_cam = cv2.flip(frame_cam, 1)

            # Submit frame to YOLO thread (non-blocking)
            yolo_thread.submit(frame_cam)

            # Read latest YOLO result (may be one or two frames old, that's fine)
            kpts_player = yolo_thread.latest_keypoints
            num_people = yolo_thread.latest_num_people or 0

            # --------------- PAUSE LOGIC (AUDIO ONLY) -----------------
            if num_people == 1:
                if paused and has_audio:
                    pygame.mixer.music.unpause()
                paused = False
                status_text = "Dancing!"
            else:
                if not paused and has_audio:
                    pygame.mixer.music.pause()
                paused = True
                status_text = "Waiting…"

            # --------------- GAME CLOCK (ONLY ADVANCES WHEN DANCING) ----
            if num_people == 1:
                game_time += dt

            # Convert game_time → reference frame index (MASTER TIMELINE)
            if num_ref_frames > 0:
                dance_idx = int(game_time * video_fps)
                dance_idx = max(0, min(dance_idx, num_ref_frames - 1))
            else:
                dance_idx = 0

            # End detection: if at last frame for > 2 sec, we stop.
            if dance_idx >= num_ref_frames - 1:
                end_hold_time += dt
            else:
                end_hold_time = 0.0

            if end_hold_time > 2.0:
                running = False

            # --------------- REFERENCE VIDEO (SEEK-BASED) ---------------
            # Hard lock: frame index = dance_idx
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, dance_idx)
            ret_ref, frame_ref = ref_cap.read()
            if not ret_ref:
                # if seek fails, reuse last frame
                if last_ref_frame is None:
                    print("[video] Failed to read reference frame.")
                    break
                frame_ref = last_ref_frame
            else:
                last_ref_frame = frame_ref

            # Draw reference skeleton using SAME index
            if 0 <= dance_idx < num_ref_frames:
                kref = self.ref_keypoints[dance_idx]
                if hide_ref_skel:
                    frame_ref_draw = frame_ref.copy()
                else:
                    frame_ref_draw = draw_skeleton(frame_ref.copy(), kref, (0, 0, 255))
            else:
                frame_ref_draw = frame_ref.copy()

            # --------------- PLAYER SKELETON (NO LIVE SCORING) --------
            if kpts_player is not None and num_people == 1 and not paused:
                if hide_webcam_skel:
                    frame_cam_draw = frame_cam.copy()
                else:
                    frame_cam_draw = draw_skeleton(frame_cam.copy(), kpts_player, (0, 255, 0))

                # Record angles only; scoring happens AFTER the song
                ang = keypoints_to_angles(kpts_player)
                self.player_angles.append(ang)
            else:
                frame_cam_draw = frame_cam.copy()

            # --------------- RENDER -----------------
            if show_window and screen is not None:
                target_h = 480

                if frame_ref_draw is None or frame_cam_draw is None:
                    continue

                r1 = cv2.resize(
                    frame_ref_draw,
                    (int(frame_ref_draw.shape[1] * target_h / frame_ref_draw.shape[0]), target_h),
                )
                r2 = cv2.resize(
                    frame_cam_draw,
                    (int(frame_cam_draw.shape[1] * target_h / frame_cam_draw.shape[0]), target_h),
                )

                combined = np.hstack((r1, r2))

                # Status text only (no live score)
                cv2.putText(
                    combined,
                    status_text,
                    (20, combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                h, w = combined.shape[:2]
                scale = min(screen_w / w, screen_h / h)
                nw, nh = int(w * scale), int(h * scale)

                final = cv2.resize(combined, (nw, nh))
                rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

                surf = pygame.image.frombuffer(rgb.tobytes(), (nw, nh), "RGB")
                screen.fill((0, 0, 0))
                screen.blit(surf, ((screen_w - nw) // 2, (screen_h - nh) // 2))

                # Emoji skeleton status
                icon = "🟢 Skeletons ON"
                if hide_webcam_skel and not hide_ref_skel:
                    icon = "🟡 Webcam OFF"
                elif not hide_webcam_skel and hide_ref_skel:
                    icon = "🔵 Ref OFF"
                elif hide_webcam_skel and hide_ref_skel:
                    icon = "🔴 ALL OFF"

                txt = font.render(icon, True, (255, 255, 255))
                screen.blit(txt, (20, 20))

                pygame.display.flip()

        # Cleanup
        yolo_thread.stop()
        cap_cam.release()
        ref_cap.release()
        cv2.destroyAllWindows()

        if has_audio:
            pygame.mixer.music.stop()

        pygame.quit()

        # ======================================================
        # POST-DANCE SCORING (OFFLINE, SMOOTH)
        # ======================================================
        if not self.player_angles:
            print("\n[final] No player data recorded — score = 0")
            return 0.0

        print("\n[scoring] Computing final score...")
        player_arr = np.stack(self.player_angles, axis=0)   # (T_player, J)
        final_score = self.score_total_series(self.ref_angles, player_arr)

        print(f"[final] Score = {final_score:.2f}")
        return final_score


# ============================================================
# CLI MODE
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", default="")
    parser.add_argument("--angles", default="")
    parser.add_argument("--keypoints", default="")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    ctrl = JustDanceController(
        reference_video=args.video,
        audio_file=args.audio,
        reference_angles_path=args.angles,
        reference_keypoints_path=args.keypoints,
        camera_index=args.camera,
    )
    ctrl.run_game(show_window=True)