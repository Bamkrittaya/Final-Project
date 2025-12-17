# just_dance_gui.py

import tkinter as tk
from tkinter import messagebox
import os

from just_dance_controller import JustDanceController
from just_dance_controller_2p import JustDanceController2P

from just_dance_score import save_score
from just_dance_gui_score import Score
from path_helper import get_fast_paths
from auto_optimize_videos import optimize_song


# List of songs (original slow video paths)
SONGS = [
    ("Call Me Maybe", "videos/callmemaybe.mp4"),
    ("Cheap Thrills", "videos/cheapthrills.mp4"),
    # ("Don't Start Now", "videos/dontstartnow.mp4"),
    # ("Lay it down (No speed up)", "videos/layitdown.mp4"),
    # ("Let me Love you (15sec)", "videos/letmeloveyou.mp4"),
    # ("Wet the bed (Tiktok)", "videos/wetthebed.mp4"),
    # ("Miniskirt (Tiktok)", "videos/miniskirt.mp4"),
    ("Swimming Pool (15sec)", "videos/swimmingpool.mp4"),
    ("Under the influence (Hardest Boy Moves)", "videos/undertheinfluence.mp4"),
    ("One of the Girls (Hardest Girl Moves)", "videos/oneofthegirls.mp4"),
]


class JustDanceGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Just Dance – YOLO Edition")
        self.geometry("400x380")

        # shared state
        self.num_players = 1
        self.player1_name = ""
        self.player2_name = ""
        self.selected_song_path = None
        self.last_p1_score = None
        self.last_p2_score = None

        # container for pages
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (WelcomePage, PlayerCountPage, PlayerNamesPage,
                  SongSelectPage, ResultPage):
            frame = F(parent=container, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("WelcomePage")

    def show_frame(self, name: str):
        """Raise a frame by name and call its on_show() if it has one."""
        frame = self.frames[name]
        if hasattr(frame, "on_show"):
            frame.on_show()
        frame.tkraise()

    # ----------------------------------------------------------
    # Core game-running logic, called from SongSelectPage
    # ----------------------------------------------------------
    def run_dance_for_current_selection(self):
        """Run either 1P or 2P controller based on current state."""
        original_video_path = self.selected_song_path
        if not original_video_path or not os.path.exists(original_video_path):
            messagebox.showerror("Error", f"Video not found:\n{original_video_path}")
            return False

        # Fast paths
        fast_video, fast_audio, fast_angles, fast_keypoints = get_fast_paths(original_video_path)

        # Auto-generate fast versions once
        if not all(os.path.exists(p) for p in [fast_video, fast_audio, fast_angles, fast_keypoints]):
            messagebox.showinfo(
                "Preparing...",
                "Optimizing video/audio & generating precompute files.\n"
                "This happens only the first time."
            )
            optimize_song(original_video_path)
            fast_video, fast_audio, fast_angles, fast_keypoints = get_fast_paths(original_video_path)

        try:
            if self.num_players == 1:
                controller = JustDanceController(
                    reference_video=fast_video,
                    audio_file=fast_audio,
                    reference_angles_path=fast_angles,
                    reference_keypoints_path=fast_keypoints,
                    camera_index=0,
                )
                score = controller.run_game(show_window=True)
                self.last_p1_score = score
                self.last_p2_score = None

                # save to leaderboard
                song_display = os.path.basename(original_video_path)
                save_score(self.player1_name, song_display, score)

            else:
                controller = JustDanceController2P(
                    reference_video=fast_video,
                    audio_file=fast_audio,
                    reference_angles_path=fast_angles,
                    reference_keypoints_path=fast_keypoints,
                    camera_index=0,
                )
                p1, p2 = controller.run_game(show_window=True)
                self.last_p1_score = p1
                self.last_p2_score = p2

                song_display = os.path.basename(original_video_path)
                save_score(f"{self.player1_name} (P1)", song_display, p1)
                save_score(f"{self.player2_name} (P2)", song_display, p2)

        except Exception as e:
            messagebox.showerror("Error", f"Something went wrong while dancing:\n{e}")
            return False

        # After running, show results page
        self.show_frame("ResultPage")
        return True

    def open_leaderboard(self):
        Score(self)


# ============================================================
# Page 0 — Welcome
# ============================================================
class WelcomePage(tk.Frame):
    def __init__(self, parent, controller: JustDanceGUI):
        super().__init__(parent)
        self.controller = controller

        title = tk.Label(self, text="Just Dance – YOLO Edition",
                         font=("Helvetica", 22, "bold"))
        title.pack(pady=30)

        subtitle = tk.Label(
            self,
            text="Move your body, let YOLO judge you (nicely).",
            font=("Helvetica", 12),
        )
        subtitle.pack(pady=10)

        info = tk.Label(
            self,
            text="This game uses pose estimation to score your dance.\n"
                 "You can play solo or challenge a friend in 2-player mode.",
            font=("Helvetica", 11),
            justify="center",
        )
        info.pack(pady=20)

        lets_begin = tk.Label(
            self,
            text="Ready?\nLet's begin~",
            font=("Helvetica", 14, "italic"),
            justify="center",
        )
        lets_begin.pack(pady=10)

        nav = tk.Frame(self)
        nav.pack(side="bottom", pady=20, fill="x")

        back_btn = tk.Button(
            nav,
            text="← Back",
            state="disabled",   # no previous from welcome
            width=10,
        )
        back_btn.pack(side="left", padx=20)

        next_btn = tk.Button(
            nav,
            text="Next →",
            width=10,
            command=lambda: controller.show_frame("PlayerCountPage"),
        )
        next_btn.pack(side="right", padx=20)


# ============================================================
# Page 1 — How many players?
# ============================================================
class PlayerCountPage(tk.Frame):
    def __init__(self, parent, controller: JustDanceGUI):
        super().__init__(parent)
        self.controller = controller
        self.num_players_var = tk.IntVar(value=1)

        label = tk.Label(self, text="How many players?",
                         font=("Helvetica", 20, "bold"))
        label.pack(pady=30)

        tk.Radiobutton(
            self,
            text="1 Player",
            variable=self.num_players_var,
            value=1,
            font=("Helvetica", 14),
        ).pack(pady=10)

        tk.Radiobutton(
            self,
            text="2 Players",
            variable=self.num_players_var,
            value=2,
            font=("Helvetica", 14),
        ).pack(pady=10)

        nav = tk.Frame(self)
        nav.pack(side="bottom", pady=20, fill="x")

        tk.Button(
            nav,
            text="← Back",
            width=10,
            command=lambda: controller.show_frame("WelcomePage"),
        ).pack(side="left", padx=20)

        tk.Button(
            nav,
            text="Next →",
            width=10,
            command=self.go_next,
        ).pack(side="right", padx=20)

    def on_show(self):
        # sync with controller state (if user went back)
        self.num_players_var.set(self.controller.num_players)

    def go_next(self):
        self.controller.num_players = self.num_players_var.get()
        self.controller.show_frame("PlayerNamesPage")


# ============================================================
# Page 2 — Enter name(s)
# ============================================================
class PlayerNamesPage(tk.Frame):
    def __init__(self, parent, controller: JustDanceGUI):
        super().__init__(parent)
        self.controller = controller

        self.label_title = tk.Label(self, text="Enter Player Name(s)",
                                    font=("Helvetica", 20, "bold"))
        self.label_title.pack(pady=20)

        # Player 1
        frame1 = tk.Frame(self)
        frame1.pack(pady=10)
        tk.Label(frame1, text="Player 1:", font=("Helvetica", 14)).pack(side="left", padx=5)
        self.entry_p1 = tk.Entry(frame1, width=25)
        self.entry_p1.pack(side="left", padx=5)

        # Player 2 (shown only when needed)
        self.frame2 = tk.Frame(self)
        tk.Label(self.frame2, text="Player 2:", font=("Helvetica", 14)).pack(side="left", padx=5)
        self.entry_p2 = tk.Entry(self.frame2, width=25)
        self.entry_p2.pack(side="left", padx=5)

        nav = tk.Frame(self)
        nav.pack(side="bottom", pady=20, fill="x")

        tk.Button(
            nav,
            text="← Back",
            width=10,
            command=lambda: controller.show_frame("PlayerCountPage"),
        ).pack(side="left", padx=20)

        tk.Button(
            nav,
            text="Next →",
            width=10,
            command=self.go_next,
        ).pack(side="right", padx=20)

    def on_show(self):
        """Adjust fields based on 1P/2P and prefill from controller."""
        # Pre-fill entries
        if self.controller.player1_name:
            self.entry_p1.delete(0, tk.END)
            self.entry_p1.insert(0, self.controller.player1_name)
        else:
            self.entry_p1.delete(0, tk.END)

        if self.controller.player2_name:
            self.entry_p2.delete(0, tk.END)
            self.entry_p2.insert(0, self.controller.player2_name)
        else:
            self.entry_p2.delete(0, tk.END)

        # Show/hide Player 2 row
        if self.controller.num_players == 2:
            self.frame2.pack(pady=10)
        else:
            self.frame2.pack_forget()

    def go_next(self):
        self.controller.player1_name = self.entry_p1.get().strip() or "Player 1"
        if self.controller.num_players == 2:
            self.controller.player2_name = self.entry_p2.get().strip() or "Player 2"
        else:
            self.controller.player2_name = ""

        self.controller.show_frame("SongSelectPage")


# ============================================================
# Page 3 — Song selection
# ============================================================
class SongSelectPage(tk.Frame):
    def __init__(self, parent, controller: JustDanceGUI):
        super().__init__(parent)
        self.controller = controller
        self.song_var = tk.StringVar(value=SONGS[0][1])

        tk.Label(self, text="Choose a song",
                 font=("Helvetica", 20, "bold")).pack(pady=20)

        self.song_buttons_frame = tk.Frame(self)
        self.song_buttons_frame.pack(pady=10, fill="x")

        for display, path in SONGS:
            tk.Radiobutton(
                self.song_buttons_frame,
                text=display,
                value=path,
                variable=self.song_var,
                font=("Helvetica", 14),
            ).pack(anchor="w", padx=40, pady=3)

        nav = tk.Frame(self)
        nav.pack(side="bottom", pady=20, fill="x")

        tk.Button(
            nav,
            text="← Back",
            width=10,
            command=lambda: controller.show_frame("PlayerNamesPage"),
        ).pack(side="left", padx=20)

        tk.Button(
            nav,
            text="Start Dance →",
            width=14,
            command=self.start_dance,
        ).pack(side="right", padx=20)

    def on_show(self):
        # If a song was chosen earlier, restore it
        if self.controller.selected_song_path:
            self.song_var.set(self.controller.selected_song_path)

    def start_dance(self):
        self.controller.selected_song_path = self.song_var.get()
        self.controller.run_dance_for_current_selection()


# ============================================================
# Page 4 — Results
# ============================================================
class ResultPage(tk.Frame):
    def __init__(self, parent, controller: JustDanceGUI):
        super().__init__(parent)
        self.controller = controller

        self.label_title = tk.Label(self, text="Results",
                                    font=("Helvetica", 22, "bold"))
        self.label_title.pack(pady=20)

        self.label_body = tk.Label(
            self,
            text="",
            font=("Helvetica", 14),
            justify="left",
        )
        self.label_body.pack(pady=10)

        self.label_winner = tk.Label(
            self,
            text="",
            font=("Helvetica", 16, "bold"),
            fg="darkgreen",
        )
        self.label_winner.pack(pady=10)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=25)

        tk.Button(
            btn_frame,
            text="Play Again",
            width=12,
            command=lambda: controller.show_frame("WelcomePage"),
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame,
            text="Leaderboard",
            width=12,
            command=controller.open_leaderboard,
        ).pack(side="left", padx=10)

        tk.Button(
            btn_frame,
            text="Quit",
            width=10,
            command=controller.destroy,
        ).pack(side="left", padx=10)

    def on_show(self):
        """Fill in text based on last scores."""
        num_players = self.controller.num_players
        p1 = self.controller.last_p1_score
        p2 = self.controller.last_p2_score
        name1 = self.controller.player1_name or "Player 1"
        name2 = self.controller.player2_name or "Player 2"

        if num_players == 1:
            self.label_title.config(text="Your Score")
            if p1 is None:
                self.label_body.config(text="No score recorded.")
            else:
                self.label_body.config(
                    text=f"{name1}: {p1:.1f}"
                )
            self.label_winner.config(text="")
        else:
            self.label_title.config(text="2-Player Results")

            if p1 is None or p2 is None:
                self.label_body.config(text="No scores recorded.")
                self.label_winner.config(text="")
                return

            self.label_body.config(
                text=f"{name1} (P1): {p1:.1f}\n"
                     f"{name2} (P2): {p2:.1f}"
            )

            if p1 > p2:
                win_text = f"🏆 {name1} wins!"
            elif p2 > p1:
                win_text = f"🏆 {name2} wins!"
            else:
                win_text = "🤝 It's a tie!"
            self.label_winner.config(text=win_text)


def main():
    app = JustDanceGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
