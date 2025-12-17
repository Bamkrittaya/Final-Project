# 🎮 Just Dance – YOLO Pose Edition  
A real-time dance game powered by **YOLOv8-Pose**, **OpenCV**, and **Python**.

> Detect your full-body pose, compare it to pre-computed dance moves, score your performance, and play in **1-player** or **strict 2-player** mode.

---

## 🔥 Features

### 🎥 Real-Time Pose Detection  
- Uses **YOLOv8 Pose (17 keypoints)**  
- Stable 30+ FPS (depending on hardware)

### 👯 1P & 2P Game Modes  
- **1-Player Mode** — same behavior as original Just Dance  
- **Strict 2-Player Mode**  
  - Requires **exactly 2 dancers**  
  - Auto-assigns **left = Player 1**, **right = Player 2**  
  - Computes **two separate scores**  
  - Declares a **winner**

### 🎵 Music + Video Sync  
- Auto-optimized fast tutorial videos  
- Audio delayed 10 seconds (or configurable)  
- Pauses music automatically if players disappear

### 🕺 Beginner-Friendly Scoring  
- Full-body 17-joint scoring  
- **Large angle tolerance (60°)** → easy, fun gameplay  
- Running live score, final average score

### ⚡ Auto Optimization (First-Time Only)  
Automatically generates:
- fast video (`videos_fast/`)
- delayed audio (`songs_audio_fast/`)
- reference keypoints + angles (`precomputed/`)

---

# 📁 Project Structure

```
multi_person_dance_game/
│
├── just_dance_gui.py
├── just_dance_controller.py
├── just_dance_controller_2p.py
├── videoPoseDetection.py
├── auto_optimize_videos.py
├── path_helper.py
│
├── videos/                ← original slow tutorial videos
├── videos_fast/           ← auto generated fast videos
├── songs_audio_fast/      ← auto generated delayed audio
├── precomputed/           ← auto generated pose data
│
└── yolo_weights/
      └── yolov8s-pose.pt
```

---

# 🛠 Installation Guide

## 1️⃣ Install Python 3.10 or 3.11  
Recommended:  
https://www.python.org/downloads/

---

## 2️⃣ Install FFmpeg  
Required for video/audio processing.

### macOS
```
brew install ffmpeg
```

### Windows  
Download: https://ffmpeg.org/download.html  
Add to PATH.

---

## 3️⃣ Create Virtual Environment (Optional but recommended)

```
python3 -m venv multi_dance_env
source multi_dance_env/bin/activate  # macOS
multi_dance_env\Scripts\activate     # Windows
```

---

## 4️⃣ Install Python Requirements

Your environment already includes all major packages:

- opencv-python  
- torch + torchvision  
- ultralytics  
- pygame  
- playsound  
- numpy  
- scipy  

If missing:
```
pip install ultralytics pygame opencv-python playsound
```

Place the YOLO model in:

```
yolo_weights/yolov8s-pose.pt
```

---

# 🚀 Running the Game

```
python3 just_dance_gui.py
```

---

# 🖥 What Happens When You Run the Game

### 1️⃣ Welcome Screen  
“Let’s begin! → Next”

### 2️⃣ Choose Number of Players  
- 1 Player  
- 2 Players (Strict Mode)

### 3️⃣ Enter Name(s)

### 4️⃣ Select Song

### 5️⃣ Auto Optimization (only first time)

### 6️⃣ Gameplay  
- live pose detection  
- live scoring  
- video + music sync  
- pause when no players detected  

### 7️⃣ End Screen  
- Show score  
- In 2P mode → show winner  
- Option to restart

---

# 🕹 Gameplay Rules

## 🎤 1-Player Mode  
- 0 people → pause  
- 1 person → play  
- 2+ people → pause  
- Full-body skeleton  
- Score updated every frame  

## 👯 2-Player Strict Mode  
- Requires **exactly 2** players  
- Left = P1, Right = P2  
- Two independent scores  

---

# 🎯 Scoring System (Beginner-Friendly)

### Using 17 joint angles (full-body)

Angle difference → Similarity score  
```
0°–10°   → 100%
10°–20°  → ~78%
20°–30°  → ~55%
30°–45°  → ~11%
≥60°     → 0%
```

---

# 🏆 Leaderboard  
After every game:
```
player_name, song_name, score
```
Viewable via GUI.

---

# 🔧 Running on GPU Server

```
ssh USERNAME@SERVER_IP
python3 just_dance_gui.py
```

Must have:
- webcam  
- GUI/X11 enabled  

---

# ❤️ Credits

- Ultralytics YOLOv8  
- OpenCV  
- Pygame  
- Python 3.10–3.11  

---

Enjoy dancing! 💃🕺
