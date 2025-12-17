# just_dance_gui_score.py

import tkinter as tk
import tkinter.font as tk_font
from just_dance_score import get_leaderboard_scores, get_latest_score
import os

# Optional avatars – you can change these emojis
AVATARS = ["🟦", "🟩", "🟧", "🟪", "🟥", "🟫", "🟩", "🟦", "🟨", "🟩"]


class Score(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)

        self.title("Leaderboard & Score")
        self.geometry("580x440")

        self.title_font = tk_font.Font(
            family="Helvetica", size=18, weight="bold"
        )
        self.text_font = tk_font.Font(family="Helvetica", size=12)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (ScorePage, LeaderboardPage):
            page = F(parent=container, controller=self)
            self.frames[F.__name__] = page
            page.grid(row=0, column=0, sticky="nsew")

        self.show_frame("ScorePage")

    def show_frame(self, name):
        self.frames[name].tkraise()


# -----------------------------------------
# PAGE 1 — Latest Score
# -----------------------------------------
class ScorePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        # --- Center wrapper ---
        wrapper = tk.Frame(self)
        wrapper.pack(expand=True)   # expands to center vertically

        tk.Label(wrapper, text="Your Latest Score", font=controller.title_font).pack(pady=15)

        latest = get_latest_score()

        if latest:
            clean_song = os.path.splitext(latest["song"])[0]

            tk.Label(wrapper, text=f"Player: {latest['player']}", font=("Helvetica", 14)).pack(pady=5)
            tk.Label(wrapper, text=f"Song: {clean_song}", font=("Helvetica", 14)).pack(pady=5)
            tk.Label(wrapper, text=f"Score: {latest['score']:.1f}", font=("Helvetica", 16, "bold")).pack(pady=10)

        else:
            tk.Label(wrapper, text="No scores recorded yet.", font=("Helvetica", 14)).pack(pady=5)

        tk.Button(
            wrapper,
            text="View Leaderboard",
            command=lambda: controller.show_frame("LeaderboardPage"),
            width=18
        ).pack(pady=15)



# -----------------------------------------
# PAGE 2 — Leaderboard (with scroll)
# -----------------------------------------
class LeaderboardPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        tk.Label(self, text="Top Scores", font=controller.title_font).pack(pady=10)

        # --- Load only Top 10 scores ---
        scores = get_leaderboard_scores(10)

        list_frame = tk.Frame(self)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)

        if not scores:
            tk.Label(list_frame, text="No scores yet.", font=("Helvetica", 14)).pack(pady=10)
        else:
            for idx, s in enumerate(scores, start=1):
                clean_song = os.path.splitext(s["song"])[0]
                clean_song = clean_song.replace("_fast", "")

                avatar = AVATARS[(idx - 1) % len(AVATARS)]

                entry = tk.Frame(list_frame)
                entry.pack(fill="x", pady=4)

                # Rank #
                tk.Label(entry, text=f"{idx}.", width=4, font=("Helvetica", 14, "bold")).pack(side="left")

                # Avatar
                tk.Label(entry, text=avatar, width=3, font=("Helvetica", 14)).pack(side="left")

                # Player + Song
                tk.Label(
                    entry,
                    text=f"{s['player']} — {clean_song}",
                    font=("Helvetica", 13),
                    anchor="w"
                ).pack(side="left", padx=8)

                # Score on right (blue)
                tk.Label(
                    entry,
                    text=f"{s['score']:.1f}",
                    font=("Helvetica", 14, "bold"),
                    fg="#0077cc"
                ).pack(side="right", padx=10)

        # --- Bottom centered button ---
        tk.Button(
            self,
            text="Back to Score",
            command=lambda: controller.show_frame("ScorePage"),
            width=15
        ).pack(side="bottom", pady=15)

