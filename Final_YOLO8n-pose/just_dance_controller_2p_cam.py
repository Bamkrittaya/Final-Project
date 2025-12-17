"""
Just Dance Controller – 2 Players – Dual Camera Edition (v4.0)
---------------------------------------------------------------

FEATURES
========
✔ Dual cameras (cam0 = P1, cam1 = P2)
✔ Supports fallback: 1 camera detecting 2 persons (auto)
✔ Reference video (top 50% height)
✔ P1 + P2 webcam feeds (bottom 50%, side-by-side)
✔ Perfect timeline sync (seek-based)
✔ Pause when players disappear
✔ Post-dance scoring (smooth)
✔ YOLO tracking IDs for stable identity
✔ Fullscreen Pygame rendering
✔ Hotkeys:
    D = toggle P1 skeleton
    F = toggle P2 skeleton
    R = toggle reference skeleton
    S = toggle ALL skeletons
    ESC/Q = quit game
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

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MODEL_PATH = "yolo_weights/yolov8n-pose.pt"
ANGLE_TOLERANCE = 60
MAX_MISSING_FRAMES = 12

FULL_BODY_PAIRS = [
    (0,1),(1,3),(0,2),(2,4),
    (5,7),(7,9),(6,8),(8,10),
    (5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

# ------------------------------------------------------------
# DRAW SKELETON
# ------------------------------------------------------------
def draw_skeleton(frame, kpts, color=(0,255,0)):
    if kpts is None: 
        return frame

    for x,y in kpts:
        if x==0 and y==0: continue
        cv2.circle(frame, (int(x),int(y)), 4, color, -1)

    for a,b in FULL_BODY_PAIRS:
        x1,y1 = kpts[a]; x2,y2 = kpts[b]
        if x1==0 or y1==0 or x2==0 or y2==0: continue
        cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)

    return frame


# ------------------------------------------------------------
# 2-PLAYER CONTROLLER – DUAL CAMERA
# ------------------------------------------------------------
class JustDanceController2P_Cam:
    def __init__(
        self,
        reference_video,
        audio_file,
        reference_angles_path,
        reference_keypoints_path,
        cam1_index=0,
        cam2_index=1,
        dual_mode=True,  # True = use two cameras, False = fallback to 1 camera
    ):
        self.ref_video_path = reference_video
        self.audio_file = audio_file
        self.angles_path = reference_angles_path
        self.kpts_path = reference_keypoints_path

        self.cam1_index = cam1_index
        self.cam2_index = cam2_index
        self.dual_mode = dual_mode

        self.model = YOLO(MODEL_PATH)

        self.ref_angles = None
        self.ref_kpts = None

        self.p1_angles = []
        self.p2_angles = []

        self.p1_id = None
        self.p2_id = None
        self.missing = 0

    # --------------------------------------------------------
    def precompute_reference(self):
        if os.path.exists(self.angles_path) and os.path.exists(self.kpts_path):
            self.ref_angles = np.load(self.angles_path)
            self.ref_kpts = np.load(self.kpts_path)
            return

        angles, kpts = precompute_video_angles(self.ref_video_path)
        self.ref_angles = angles
        self.ref_kpts = kpts
        np.save(self.angles_path, angles)
        np.save(self.kpts_path, kpts)

    # --------------------------------------------------------
    @staticmethod
    def extract_people_with_ids(result):
        if len(result)==0 or result[0].keypoints is None:
            return None, None
        xy = result[0].keypoints.xy
        if xy is None or xy.numel()==0:
            return None, None
        kpts = xy.cpu().numpy()
        ids = None
        if result[0].boxes.id is not None:
            ids = result[0].boxes.id.cpu().numpy()
        return kpts, ids

    # --------------------------------------------------------
    def lock_ids(self, ids, kpts):
        if ids is None or len(ids)<2: return
        xcent = kpts[:,:,0].mean(axis=1)
        left = int(np.argmin(xcent))
        right = int(np.argmax(xcent))
        self.p1_id = int(ids[left])
        self.p2_id = int(ids[right])
        self.missing = 0
        print(f"[2P] Locked IDs: P1={self.p1_id}, P2={self.p2_id}")

    # --------------------------------------------------------
    def get_by_id(self, ids, kpts):
        p1 = p2 = None
        if ids is None: return None, None, False
        for i,pid in enumerate(ids):
            if self.p1_id==pid: p1 = kpts[i]
            if self.p2_id==pid: p2 = kpts[i]
        return p1, p2, (p1 is not None and p2 is not None)

    # --------------------------------------------------------
    def score_joint(self, ref, pl):
        T = min(len(ref), len(pl))
        ref = ref[:T]; pl = pl[:T]
        mask = ~((ref==0)|(pl==0))
        if mask.sum()==0: return 0
        diff = np.abs(ref[mask]-pl[mask])
        sim = np.clip(1 - diff/ANGLE_TOLERANCE, 0,1)
        return float(sim.mean()*100)

    def final_score(self, ref_series, pl_series):
        if len(pl_series)==0: return 0
        T = min(ref_series.shape[0], pl_series.shape[0])
        ref = ref_series[:T]
        pl = pl_series[:T]
        scores = [self.score_joint(ref[:,j], pl[:,j]) for j in range(ref.shape[1])]
        return float(np.mean(scores))

    # --------------------------------------------------------
    def run_game(self, show_window=True):

        # load reference
        self.precompute_reference()

        pygame.init()
        pygame.display.set_caption("Just Dance 2-Player Dual Camera")

        screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
        sw, sh = screen.get_size()
        font = pygame.font.SysFont(None, 45)

        # open cameras
        cam1 = cv2.VideoCapture(self.cam1_index)
        cam2 = cv2.VideoCapture(self.cam2_index) if self.dual_mode else None

        # reference video
        ref_cap = cv2.VideoCapture(self.ref_video_path)
        fps = ref_cap.get(cv2.CAP_PROP_FPS) or 30
        vframes = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ref_frames = min(vframes, len(self.ref_kpts), len(self.ref_angles))

        # audio
        have_audio = False
        try:
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()
            pygame.mixer.music.pause()
            have_audio = True
        except:
            pass

        # skeleton toggles
        show_ref = True
        show_p1 = True
        show_p2 = True

        # timeline
        game_t = 0
        last_t = time.time()
        end_hold = 0
        running = True
        paused = True

        last_ref_frame = None

        while running:

            # ---------- EVENTS ----------
            for e in pygame.event.get():
                if e.type==pygame.QUIT: running=False
                if e.type==pygame.KEYDOWN:
                    if e.key in (pygame.K_ESCAPE, pygame.K_q):
                        running=False
                    elif e.key==pygame.K_d:
                        show_p1 = not show_p1
                    elif e.key==pygame.K_f:
                        show_p2 = not show_p2
                    elif e.key==pygame.K_r:
                        show_ref = not show_ref
                    elif e.key==pygame.K_s:
                        all_off = (not show_ref and not show_p1 and not show_p2)
                        show_ref = show_p1 = show_p2 = not all_off

            # ---------- TIME ----------
            now = time.time()
            dt = now - last_t
            last_t = now

            # ---------- WEBCAM READS ----------
            ret1, f1 = cam1.read()
            if not ret1: break
            f1 = cv2.flip(f1,1)

            if self.dual_mode:
                ret2, f2 = cam2.read()
                if not ret2: break
                f2 = cv2.flip(f2,1)
            else:
                f2 = None

            # ---------- YOLO -------------
            if self.dual_mode:
                r1 = self.model.track(f1, conf=0.3, persist=True, verbose=False)
                r2 = self.model.track(f2, conf=0.3, persist=True, verbose=False)
                k1, id1 = self.extract_people_with_ids(r1)
                k2, id2 = self.extract_people_with_ids(r2)

                # treat cam1 = P1, cam2 = P2
                p1_kpts = k1[0] if k1 is not None and len(k1)>0 else None
                p2_kpts = k2[0] if k2 is not None and len(k2)>0 else None
                visible = (p1_kpts is not None and p2_kpts is not None)
            else:
                r = self.model.track(f1, conf=0.3, persist=True, verbose=False)
                k, ids = self.extract_people_with_ids(r)
                if self.p1_id is None or self.p2_id is None:
                    if k is not None and ids is not None and len(k)>=2:
                        self.lock_ids(ids,k)
                    visible=False
                    p1_kpts=p2_kpts=None
                else:
                    p1_kpts,p2_kpts,visible = self.get_by_id(ids,k)

            # ---------- PAUSE LOGIC ----------
            if visible:
                if paused and have_audio:
                    pygame.mixer.music.unpause()
                paused=False
            else:
                if not paused and have_audio:
                    pygame.mixer.music.pause()
                paused=True

            # ---------- TIMELINE ----------
            if visible:
                game_t += dt

            idx = int(game_t * fps)
            idx = max(0, min(idx, ref_frames-1))

            if idx>=ref_frames-1:
                end_hold += dt
            else:
                end_hold = 0

            if end_hold>2: break

            # ---------- REFERENCE SEEK ----------
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, rf = ref_cap.read()
            if ok:
                last_ref_frame = rf
            else:
                rf = last_ref_frame

            rk = self.ref_kpts[idx]
            if show_ref:
                rf_draw = draw_skeleton(rf.copy(), rk, (0,0,255))
            else:
                rf_draw = rf.copy()

            # ---------- PLAYER SKELETON ----------
            if p1_kpts is not None and p2_kpts is not None and visible:
                if show_p1:
                    f1_draw = draw_skeleton(f1.copy(), p1_kpts, (0,255,0))
                else:
                    f1_draw = f1.copy()

                if show_p2:
                    f2_draw = draw_skeleton(f2.copy(), p2_kpts, (0,255,255))
                else:
                    f2_draw = f2.copy()

                self.p1_angles.append(keypoints_to_angles(p1_kpts))
                self.p2_angles.append(keypoints_to_angles(p2_kpts))
            else:
                f1_draw = f1.copy()
                f2_draw = f2.copy() if self.dual_mode else f1.copy()

            # ---------- UI LAYOUT ----------
            # top = reference video (50% height)
            # bottom = P1 + P2 side by side

            top_h = sh // 2
            bot_h = sh - top_h

            # resize top
            t_h, t_w = rf_draw.shape[:2]
            top_w = int(t_w * (top_h / t_h))
            rf_resized = cv2.resize(rf_draw, (top_w, top_h))

            # resize bottom cams to equal size
            cam_h = bot_h
            cam_w = sw // 2

            f1_res = cv2.resize(f1_draw, (cam_w, cam_h))
            f2_res = cv2.resize(f2_draw, (cam_w, cam_h))

            # compose bottom
            bottom = np.hstack((f1_res, f2_res))

            # pad top horizontally centered
            big = np.zeros((sh, sw, 3), dtype=np.uint8)

            start_x = (sw - top_w)//2
            big[0:top_h, start_x:start_x+top_w] = rf_resized

            big[top_h:sh, 0:sw] = bottom

            # ---------- DRAW TO SCREEN ----------
            rgb = cv2.cvtColor(big, cv2.COLOR_BGR2RGB)
            surf = pygame.image.frombuffer(rgb.tobytes(), (sw, sh), "RGB")
            screen.blit(surf, (0,0))

            status = "Dancing!" if visible else "Waiting..."
            txt = font.render(status, True, (255,255,0))
            screen.blit(txt, (30,30))

            pygame.display.flip()

        # END LOOP
        cam1.release()
        if self.dual_mode and cam2 is not None:
            cam2.release()
        ref_cap.release()
        pygame.quit()

        # ---------- FINAL SCORE ----------
        if len(self.p1_angles)==0: 
            return 0.0, 0.0

        p1 = np.stack(self.p1_angles,axis=0)
        p2 = np.stack(self.p2_angles,axis=0)

        score1 = self.final_score(self.ref_angles, p1)
        score2 = self.final_score(self.ref_angles, p2)

        print(f"\nFINAL SCORES:\n P1 = {score1:.2f}\n P2 = {score2:.2f}\n")
        return score1, score2
