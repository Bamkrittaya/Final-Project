# interpretation.py
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
RUN_DIR = "logs/run2"   # <<< CHANGE THIS PER EXPERIMENT
SAVE_FIGS = False      # True = save figures instead of show()

def load_csv(name):
    path = os.path.join(RUN_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return np.loadtxt(path, delimiter=",", skiprows=1)

def maybe_show(title):
    if SAVE_FIGS:
        os.makedirs(os.path.join(RUN_DIR, "figures"), exist_ok=True)
        plt.savefig(os.path.join(RUN_DIR, "figures", title + ".png"), dpi=150)
        plt.close()
    else:
        plt.show()

print(f"\n📂 Analysing run: {RUN_DIR}")

# ============================================================
# 1. LATENCY ANALYSIS
# ============================================================
lat = load_csv("latency_total.csv")

plt.figure(figsize=(16, 5))
plt.plot(lat, linewidth=0.7)
plt.title("Total End-to-End Latency per Frame")
plt.xlabel("Frame Index")
plt.ylabel("Latency (ms)")
plt.grid(alpha=0.3)
plt.tight_layout()
maybe_show("latency_full")

plt.figure(figsize=(16, 5))
plt.plot(lat[:1000], linewidth=0.7)
plt.title("Latency (First 1000 Frames — Stability)")
plt.xlabel("Frame Index")
plt.ylabel("Latency (ms)")
plt.grid(alpha=0.3)
plt.tight_layout()
maybe_show("latency_zoom")

print("\n===== LATENCY =====")
print(f"Frames        : {len(lat)}")
print(f"Mean          : {lat.mean():.2f} ms")
print(f"Median        : {np.median(lat):.2f} ms")
print(f"Std (jitter)  : {lat.std():.2f} ms")
print(f"Min / Max     : {lat.min():.2f} / {lat.max():.2f} ms")
print(f">50ms frames  : {(lat > 50).sum()} ({(lat > 50).mean()*100:.2f}%)")
print(f"Effective FPS : {1000/lat.mean():.2f}")

# ============================================================
# 2. FPS / PLAYBACK STABILITY
# ============================================================
fps_data = load_csv("fps_playback.csv")
t = fps_data[:, 0]
dt = fps_data[:, 1]
fps = fps_data[:, 2]

plt.figure(figsize=(16, 5))
plt.plot(fps, linewidth=0.7)
plt.title("Instantaneous FPS Over Time")
plt.xlabel("Frame Index")
plt.ylabel("FPS")
plt.grid(alpha=0.3)
plt.tight_layout()
maybe_show("fps_over_time")

print("\n===== PLAYBACK FPS =====")
print(f"Mean FPS   : {np.mean(fps):.2f}")
print(f"Median FPS : {np.median(fps):.2f}")
print(f"Min / Max  : {fps.min():.2f} / {fps.max():.2f}")

# ============================================================
# 3. SKELETON COMPLETENESS
# ============================================================
valid = load_csv("skeleton_validity.csv")

num_people = valid[:, 0]
valid_ratio = valid[:, 1]

plt.figure(figsize=(16, 5))
plt.plot(valid_ratio, linewidth=0.7)
plt.title("Skeleton Completeness Ratio Over Time")
plt.xlabel("Frame Index")
plt.ylabel("Valid Joint Ratio")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.tight_layout()
maybe_show("skeleton_validity")

print("\n===== SKELETON COMPLETENESS =====")
print(f"Mean validity   : {valid_ratio.mean():.3f}")
print(f"Median validity : {np.median(valid_ratio):.3f}")
print(f"Min / Max       : {valid_ratio.min():.3f} / {valid_ratio.max():.3f}")

# ============================================================
# 4. JITTER ANALYSIS (NaN-safe)
# ============================================================
jitter = load_csv("jitter_per_joint.csv")  # shape (T, 17)

jitter_mean_per_frame = np.nanmean(jitter, axis=1)
jitter_mean_per_joint = np.nanmean(jitter, axis=0)

plt.figure(figsize=(16, 5))
plt.plot(jitter_mean_per_frame, linewidth=0.7)
plt.title("Mean Skeleton Jitter per Frame")
plt.xlabel("Frame Index")
plt.ylabel("Jitter (pixels)")
plt.grid(alpha=0.3)
plt.tight_layout()
maybe_show("jitter_per_frame")

plt.figure(figsize=(12, 5))
plt.bar(range(17), jitter_mean_per_joint)
plt.title("Mean Jitter per Joint")
plt.xlabel("Joint Index (COCO 17)")
plt.ylabel("Jitter (pixels)")
plt.tight_layout()
maybe_show("jitter_per_joint")

print("\n===== JITTER =====")
print(f"Mean jitter (px)   : {np.nanmean(jitter):.2f}")
print(f"Median jitter (px) : {np.nanmedian(jitter):.2f}")

# ============================================================
# 5. KEYPOINT CONFIDENCE
# ============================================================
conf = load_csv("keypoint_conf.csv")  # shape (T, 17)

conf_mean_per_joint = conf.mean(axis=0)

plt.figure(figsize=(12, 5))
plt.bar(range(17), conf_mean_per_joint)
plt.title("Mean Keypoint Confidence per Joint")
plt.xlabel("Joint Index (COCO 17)")
plt.ylabel("Confidence")
plt.ylim(0, 1)
plt.tight_layout()
maybe_show("keypoint_confidence")

print("\n===== KEYPOINT CONFIDENCE =====")
print(f"Mean confidence : {conf.mean():.3f}")
print(f"Min / Max       : {conf.min():.3f} / {conf.max():.3f}")

# ============================================================
# 6. FINAL SUMMARY (REPORT-READY)
# ============================================================
print("\n================ FINAL SUMMARY ================")
print(f"Run folder           : {RUN_DIR}")
print(f"Frames analysed      : {len(lat)}")
print(f"Avg latency          : {lat.mean():.2f} ms")
print(f"Avg FPS              : {np.mean(fps):.2f}")
print(f"Avg skeleton validity: {valid_ratio.mean():.3f}")
print(f"Avg jitter           : {np.nanmean(jitter):.2f} px")
print(f"Avg confidence       : {conf.mean():.3f}")
print("==============================================\n")
