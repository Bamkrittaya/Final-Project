# just_dance_controller.py
import os
import cv2
import time
import json
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

# Logging / metrics config
LOG_DIR = "logs"
CONF_THRESHOLD = 0.30          # for "valid joint" (completeness)
ZERO_KPT_IS_INVALID = True     # treat (0,0) as invalid
MAX_LOG_FRAMES = None          # set int to cap logs, or None = no cap

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
    Main loop just submits frames and reads latest keypoints/confidence.
    """
    def __init__(self, model, conf=0.3):
        self.model = model
        self.conf = conf
        self.queue = queue.Queue(maxsize=1)
        self.running = True

        # latest results (read from main thread)
        self.latest_keypoints = None     # (17,2) float
        self.latest_conf = None          # (17,) float
        self.latest_num_people = 0
        self.latest_latency = None       # ms

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def submit(self, frame_bgr):
        """Submit a frame for detection (non-blocking). Drop old frame if needed."""
        if frame_bgr is None:
            return
        if self.queue.full():
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def _worker(self):
        while self.running:
            try:
                frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                t0 = time.perf_counter()
                results = self.model(frame, conf=self.conf, verbose=False)
                t1 = time.perf_counter()
                self.latest_latency = (t1 - t0) * 1000.0
            except Exception as e:
                print(f"[YOLOThread] Inference error: {e}")
                continue

            kpts = None
            kconf = None
            num_people = 0

            if results and results[0].keypoints is not None:
                kp = results[0].keypoints
                xy = kp.xy
                cf = kp.conf  # (N,17) if available
                if xy is not None and xy.numel() > 0:
                    num_people = int(xy.shape[0])
                    # first person only
                    kpts = xy[0].detach().cpu().numpy()
                    if cf is not None and cf.numel() > 0:
                        kconf = cf[0].detach().cpu().numpy()

            self.latest_keypoints = kpts
            self.latest_conf = kconf
            self.latest_num_people = num_people

    def stop(self):
        self.running = False


# ============================================================
# METRICS / LOGGING (lightweight during runtime)
# ============================================================
class MetricsLogger:
    """
    Collects per-frame metrics with minimal overhead.
    Saves to CSV/JSON at end.
    """
    def __init__(self, log_dir=LOG_DIR):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Per-frame logs (append-only)
        self.lat_total = []
        self.lat_yolo = []
        self.lat_post = []

        self.playback = []   # [t, dt_ms, fps, game_time, dance_idx, num_people, paused]
        self.conf = []       # [t, dance_idx, conf0..conf16]
        self.validity = []   # [t, dance_idx, valid_count, valid_ratio]
        self.jitter = []     # [t, dance_idx, jit0..jit16] (pixels)

        # state for jitter
        self._prev_kpts = None

    @staticmethod
    def _valid_mask(kpts, conf, conf_thr=CONF_THRESHOLD):
        """
        Returns boolean mask (17,) where joint is considered valid.
        Valid if:
          - conf >= threshold (when conf exists)
          - and not (0,0) when ZERO_KPT_IS_INVALID
        """
        if kpts is None:
            return None

        mask = np.ones((kpts.shape[0],), dtype=bool)

        if conf is not None:
            mask &= (conf >= conf_thr)

        if ZERO_KPT_IS_INVALID:
            mask &= ~((kpts[:, 0] == 0) & (kpts[:, 1] == 0))

        return mask

    def log_frame(
        self,
        t_rel,
        dt_ms,
        game_time,
        dance_idx,
        num_people,
        paused,
        kpts,
        conf,
        lat_total_ms,
        lat_yolo_ms,
        lat_post_ms,
    ):
        # Optional cap
        if MAX_LOG_FRAMES is not None:
            if len(self.playback) >= MAX_LOG_FRAMES:
                return

        fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0

        # playback/fps
        self.playback.append([
            float(t_rel), float(dt_ms), float(fps),
            float(game_time), int(dance_idx), int(num_people), int(paused)
        ])

        # latency
        self.lat_total.append(float(lat_total_ms))
        if lat_yolo_ms is not None:
            self.lat_yolo.append(float(lat_yolo_ms))
        self.lat_post.append(float(lat_post_ms))

        # confidence + completeness + jitter only when we have a player and kpts
        if kpts is None:
            self._prev_kpts = None
            return

        # confidence csv (store zeros if missing conf)
        if conf is None:
            conf_row = np.zeros((17,), dtype=float)
        else:
            conf_row = conf.astype(float)

        self.conf.append([float(t_rel), int(dance_idx), *conf_row.tolist()])

        # completeness
        vmask = self._valid_mask(kpts, conf_row, CONF_THRESHOLD)
        if vmask is None:
            valid_count = 0
            valid_ratio = 0.0
        else:
            valid_count = int(vmask.sum())
            valid_ratio = float(valid_count / 17.0)

        self.validity.append([float(t_rel), int(dance_idx), valid_count, valid_ratio])

        # jitter (pixels): per joint displacement since last frame
        if self._prev_kpts is None or self._prev_kpts.shape != kpts.shape:
            jit = np.zeros((17,), dtype=float)
        else:
            diff = kpts.astype(float) - self._prev_kpts.astype(float)
            jit = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)

            # If joint invalid now, mark jitter 0 to avoid garbage spikes
            if vmask is not None:
                jit = np.where(vmask, jit, 0.0)

        self.jitter.append([float(t_rel), int(dance_idx), *jit.tolist()])
        self._prev_kpts = kpts.copy()

    @staticmethod
    def _stats(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return {"count": 0}
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def save_all(self):
        # latency
        np.savetxt(os.path.join(self.log_dir, "latency_total.csv"),
                   np.asarray(self.lat_total), delimiter=",")

        if len(self.lat_yolo) > 0:
            np.savetxt(os.path.join(self.log_dir, "latency_yolo.csv"),
                       np.asarray(self.lat_yolo), delimiter=",")

        np.savetxt(os.path.join(self.log_dir, "latency_post.csv"),
                   np.asarray(self.lat_post), delimiter=",")

        # playback / fps
        playback_arr = np.asarray(self.playback, dtype=float)
        # columns: t_rel, dt_ms, fps, game_time, dance_idx, num_people, paused
        np.savetxt(os.path.join(self.log_dir, "playback_fps.csv"),
                   playback_arr, delimiter=",")

        # confidence
        if len(self.conf) > 0:
            conf_arr = np.asarray(self.conf, dtype=float)
            np.savetxt(os.path.join(self.log_dir, "keypoint_conf.csv"),
                       conf_arr, delimiter=",")

        # completeness
        if len(self.validity) > 0:
            val_arr = np.asarray(self.validity, dtype=float)
            np.savetxt(os.path.join(self.log_dir, "skeleton_validity.csv"),
                       val_arr, delimiter=",")

        # jitter per joint
        if len(self.jitter) > 0:
            jit_arr = np.asarray(self.jitter, dtype=float)
            np.savetxt(os.path.join(self.log_dir, "jitter_per_joint.csv"),
                       jit_arr, delimiter=",")

        # summary.json (nice for your report)
        summary = {}

        summary["latency_total_ms"] = self._stats(self.lat_total)
        summary["latency_yolo_ms"] = self._stats(self.lat_yolo) if len(self.lat_yolo) else {"count": 0}
        summary["latency_post_ms"] = self._stats(self.lat_post)

        # fps stats from playback
        if playback_arr.size > 0:
            fps_col = playback_arr[:, 2]
            dt_col = playback_arr[:, 1]
            summary["fps"] = self._stats(fps_col)
            summary["frame_time_ms"] = self._stats(dt_col)
        else:
            summary["fps"] = {"count": 0}
            summary["frame_time_ms"] = {"count": 0}

        # completeness stats
        if len(self.validity) > 0:
            val_arr = np.asarray(self.validity, dtype=float)
            valid_ratio = val_arr[:, 3]
            summary["skeleton_valid_ratio"] = self._stats(valid_ratio)
        else:
            summary["skeleton_valid_ratio"] = {"count": 0}

        # jitter joint-wise mean/std (for your evaluation section)
        if len(self.jitter) > 0:
            jit_arr = np.asarray(self.jitter, dtype=float)
            jit_only = jit_arr[:, 2:]  # 17 joints
            summary["jitter_per_joint_mean_px"] = [float(x) for x in np.mean(jit_only, axis=0)]
            summary["jitter_per_joint_std_px"] = [float(x) for x in np.std(jit_only, axis=0)]
            summary["jitter_overall_mean_px"] = float(np.mean(jit_only))
        else:
            summary["jitter_per_joint_mean_px"] = []
            summary["jitter_per_joint_std_px"] = []
            summary["jitter_overall_mean_px"] = 0.0

        with open(os.path.join(self.log_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[logs] Saved to: {self.log_dir}/")
        print(" - latency_total.csv, latency_yolo.csv, latency_post.csv")
        print(" - keypoint_conf.csv")
        print(" - skeleton_validity.csv")
        print(" - jitter_per_joint.csv")
        print(" - playback_fps.csv")
        print(" - summary.json\n")


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

        self.model = YOLO(MODEL_PATH)

        self.ref_angles = None
        self.ref_keypoints = None
        self.player_angles = []

        # metrics logger (NEW)
        self.metrics = MetricsLogger(LOG_DIR)

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
        T = min(len(ref), len(player))
        ref, player = ref[:T], player[:T]

        mask = ~((ref == 0) | (player == 0))
        if mask.sum() == 0:
            return 0.0

        diff = np.abs(ref[mask] - player[mask])
        sim = np.clip(1.0 - diff / ANGLE_TOLERANCE, 0, 1)
        return float(sim.mean() * 100)

    def score_total_series(self, ref_series, player_series):
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
        # Load reference
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

        # Audio
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

        # Reference video
        ref_cap = cv2.VideoCapture(self.reference_video)
        if not ref_cap.isOpened():
            print("[error] Cannot open reference video:", self.reference_video)
            return 0.0

        video_fps = ref_cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        print(f"[info] Reference video FPS: {video_fps:.2f}")

        video_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_ref_frames = min(video_frames, len(self.ref_keypoints), len(self.ref_angles))
        print(f"[info] Ref frames (video vs npy): {video_frames} vs {len(self.ref_keypoints)} (using {num_ref_frames})")

        # Async YOLO
        yolo_thread = YOLOThread(self.model, conf=0.3)

        # Game time logic
        game_time = 0.0
        last_loop_time = time.perf_counter()
        end_hold_time = 0.0

        paused = True
        status_text = "Waiting…"

        hide_webcam_skel = True
        hide_ref_skel = True

        last_ref_frame = None
        running = True

        # relative clock for logs
        start_perf = time.perf_counter()

        # =============================
        # MAIN LOOP
        # =============================
        while running:
            # ---------- EVENTS ----------
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

            # ---------- TIME STEP ----------
            now = time.perf_counter()
            dt = now - last_loop_time
            last_loop_time = now
            dt_ms = dt * 1000.0

            # ---------- CAPTURE WEBCAM ----------
            ret_cam, frame_cam = cap_cam.read()
            if not ret_cam:
                print("[camera] Failed frame.")
                break

            t0 = time.perf_counter()  # total begin (post-capture)
            frame_cam = cv2.flip(frame_cam, 1)

            # Submit to YOLO thread
            yolo_thread.submit(frame_cam)

            # Get latest results
            kpts_player = yolo_thread.latest_keypoints
            conf_player = yolo_thread.latest_conf
            num_people = yolo_thread.latest_num_people or 0

            # ---------- PAUSE LOGIC ----------
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

            # ---------- GAME CLOCK ----------
            if num_people == 1:
                game_time += dt

            dance_idx = int(game_time * video_fps)
            dance_idx = max(0, min(dance_idx, num_ref_frames - 1))

            # End detection
            if dance_idx >= num_ref_frames - 1:
                end_hold_time += dt
            else:
                end_hold_time = 0.0

            if end_hold_time > 2.0:
                running = False

            # ---------- REFERENCE VIDEO ----------
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, dance_idx)
            ret_ref, frame_ref = ref_cap.read()

            if not ret_ref:
                if last_ref_frame is None:
                    print("[video] Failed ref frame.")
                    break
                frame_ref = last_ref_frame
            else:
                last_ref_frame = frame_ref

            # Draw ref
            if 0 <= dance_idx < num_ref_frames:
                kref = self.ref_keypoints[dance_idx]
                if hide_ref_skel:
                    frame_ref_draw = frame_ref.copy()
                else:
                    frame_ref_draw = draw_skeleton(frame_ref.copy(), kref, (0, 0, 255))
            else:
                frame_ref_draw = frame_ref.copy()

            # ---------- PLAYER ----------
            t2 = time.perf_counter()  # post-processing begin

            if kpts_player is not None and num_people == 1 and not paused:
                if hide_webcam_skel:
                    frame_cam_draw = frame_cam.copy()
                else:
                    frame_cam_draw = draw_skeleton(frame_cam.copy(), kpts_player, (0, 255, 0))

                ang = keypoints_to_angles(kpts_player)
                self.player_angles.append(ang)
            else:
                frame_cam_draw = frame_cam.copy()

            t3 = time.perf_counter()  # post-processing end

            # ---------- RECORD METRICS (very lightweight) ----------
            lat_yolo_ms = getattr(yolo_thread, "latest_latency", None)
            lat_post_ms = (t3 - t2) * 1000.0
            lat_total_ms = (t3 - t0) * 1000.0
            t_rel = time.perf_counter() - start_perf

            self.metrics.log_frame(
                t_rel=t_rel,
                dt_ms=dt_ms,
                game_time=game_time,
                dance_idx=dance_idx,
                num_people=num_people,
                paused=paused,
                kpts=kpts_player if (num_people == 1 and not paused) else None,
                conf=conf_player if (num_people == 1 and not paused) else None,
                lat_total_ms=lat_total_ms,
                lat_yolo_ms=lat_yolo_ms,
                lat_post_ms=lat_post_ms,
            )

            # ---------- RENDER ----------
            if show_window and screen is not None:
                target_h = 480

                r1 = cv2.resize(frame_ref_draw, (int(frame_ref_draw.shape[1] * target_h / frame_ref_draw.shape[0]), target_h))
                r2 = cv2.resize(frame_cam_draw, (int(frame_cam_draw.shape[1] * target_h / frame_cam_draw.shape[0]), target_h))

                combined = np.hstack((r1, r2))

                cv2.putText(combined, status_text, (20, combined.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                h, w = combined.shape[:2]
                scale = min(screen_w / w, screen_h / h)
                nw, nh = int(w * scale), int(h * scale)

                final = cv2.resize(combined, (nw, nh))
                rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

                surf = pygame.image.frombuffer(rgb.tobytes(), (nw, nh), "RGB")
                screen.fill((0, 0, 0))
                screen.blit(surf, ((screen_w - nw) // 2, (screen_h - nh) // 2))

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
        # POST-DANCE SCORING
        # ======================================================
        if not self.player_angles:
            print("\n[final] No player data recorded — score = 0")
            # still save logs
            self.metrics.save_all()
            return 0.0

        print("\n[scoring] Computing final score...")
        player_arr = np.stack(self.player_angles, axis=0)
        final_score = self.score_total_series(self.ref_angles, player_arr)
        print(f"[final] Score = {final_score:.2f}")

        # ======================================================
        # SAVE LOGS (end-only)
        # ======================================================
        self.metrics.save_all()

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
