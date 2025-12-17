# just_dance_score.py

import json
import os
import time

SCORE_FILE = "leaderboard.json"


def _load_scores():
    if not os.path.exists(SCORE_FILE):
        return []
    with open(SCORE_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_scores(scores):
    with open(SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)


def save_score(player_name, song_name, score):
    scores = _load_scores()
    scores.append(
        {
            "player": player_name or "Player",
            "song": song_name,
            "score": float(score),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    # sort high to low
    scores.sort(key=lambda s: s["score"], reverse=True)
    _save_scores(scores)


def get_leaderboard_scores(limit=10):
    scores = _load_scores()
    return scores[:limit]


def get_latest_score():
    scores = _load_scores()
    return scores[0] if scores else None

def get_current_score():
    return get_latest_score()

