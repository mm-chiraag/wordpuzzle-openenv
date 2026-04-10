"""
app.py — FastAPI server for WordPuzzle OpenEnv
Endpoints: /reset  /step  /state  /grader  /tasks  /health
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from run_standalone import (
    WordPuzzleEnvironment,
    WordPuzzleAction,
    LEVEL_CONFIG,
    WORDS,
)

app = FastAPI(title="WordPuzzle OpenEnv")

# ─── Session storage ──────────────────────────────────────────────────────────
env_sessions: dict = {}

TASK_LEVEL_MAP = {
    "wordpuzzle-easy":   1,
    "wordpuzzle-medium": 2,
    "wordpuzzle-hard":   3,
}


# ─── Score must be STRICTLY between 0 and 1 (not 0.0, not 1.0) ───────────────
def clamp_score(score: float) -> float:
    return round(max(0.01, min(0.99, float(score))), 4)


# ─── Helper ───────────────────────────────────────────────────────────────────
def _obs_dict(obs, reward=0.01, done=False):
    return {
        "observation": {
            "feedback":      obs.feedback,
            "attempts_used": obs.attempts_used,
            "max_attempts":  obs.max_attempts,
            "task_level":    obs.task_level,
            "message":       obs.message,
            "solved":        obs.solved,
            "revealed_word": obs.revealed_word,
        },
        "reward": clamp_score(reward),
        "done":   done,
    }


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><head><meta charset="utf-8"><title>WordPuzzle OpenEnv</title>
    <style>
      body{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:40px;max-width:800px;margin:0 auto;}
      h1{color:#58a6ff;}
      .badge{background:#238636;color:#fff;padding:4px 12px;border-radius:20px;margin:4px;display:inline-block;}
      pre{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;}
    </style></head>
    <body>
    <h1>WordPuzzle &mdash; OpenEnv RL Environment</h1>
    <span class="badge">Runtime Correctness</span>
    <span class="badge">Interface Compliance</span>
    <span class="badge">Task Design</span>
    <span class="badge">Grading Logic</span>
    <pre>Tasks: wordpuzzle-easy | wordpuzzle-medium | wordpuzzle-hard
API  : /reset  /step  /state  /grader  /tasks  /health
Score: strictly between 0.01 and 0.99</pre>
    <p>See <a href="/docs" style="color:#58a6ff;">/docs</a></p>
    </body></html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    tasks = []
    for task_id, level in TASK_LEVEL_MAP.items():
        cfg = LEVEL_CONFIG[level]
        tasks.append({
            "id":           task_id,
            "name":         f"WordPuzzle {cfg['name']}",
            "description":  f"Guess a {cfg['word_length']}-letter word in {cfg['max_attempts']} attempts.",
            "word_length":  cfg["word_length"],
            "max_attempts": cfg["max_attempts"],
            "grader": {"endpoint": "/grader", "field": "score", "range": [0.01, 0.99]},
        })
    return {"tasks": tasks}


@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body is None:
        body = {}

    task_id    = body.get("task_id", "wordpuzzle-easy") or "wordpuzzle-easy"
    session_id = body.get("session_id", "default") or "default"
    level      = TASK_LEVEL_MAP.get(task_id, 1)

    env = WordPuzzleEnvironment(task_level=level)
    obs = env.reset()
    env_sessions[session_id] = env

    result = _obs_dict(obs)
    result["task_id"] = task_id
    return result


@app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body is None:
        body = {}

    session_id = body.get("session_id", "default") or "default"
    action_str = body.get("action", "") or ""

    if session_id not in env_sessions:
        env = WordPuzzleEnvironment(task_level=1)
        env.reset()
        env_sessions[session_id] = env

    env    = env_sessions[session_id]
    action = WordPuzzleAction(guess=action_str)

    try:
        obs, reward, done = env.step(action)
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "error": str(e), "done": True, "reward": 0.05
        })

    if done:
        env_sessions.pop(session_id, None)

    return _obs_dict(obs, reward, done)


@app.get("/state")
def state(session_id: str = "default"):
    if session_id not in env_sessions:
        return {"error": "No active session"}
    env = env_sessions[session_id]
    s   = env.state()
    return {
        "state": {
            "target_word":  s.target_word,
            "guesses":      s.guesses,
            "task_level":   s.task_level,
            "total_reward": s.total_reward,
            "done":         s.done,
        }
    }


@app.post("/grader")
async def grader(request: Request):
    """
    Grade a guess. Score is STRICTLY between 0.01 and 0.99 (never 0.0 or 1.0).
    """
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if body is None:
            body = {}

        task_id    = body.get("task_id", "wordpuzzle-easy") or "wordpuzzle-easy"
        guess      = (body.get("guess", "") or "").lower().strip()
        target     = (body.get("target", "") or "").lower().strip()
        session_id = body.get("session_id", None)

        level    = TASK_LEVEL_MAP.get(task_id, 1)
        cfg      = LEVEL_CONFIG[level]
        word_len = cfg["word_length"]

        # If checker provides a completed session, score from its final state
        if session_id and session_id in env_sessions:
            env = env_sessions[session_id]
            s   = env.state()
            total = max(float(s.total_reward), 0.0)
            raw = total / max(total + 1.0, 10.0)
            return {"score": clamp_score(raw), "task_id": task_id, "valid": True}

        # Grade a single guess using a fresh env
        env = WordPuzzleEnvironment(task_level=level)
        env.reset()

        # Set target word
        if target and len(target) == word_len and target.isalpha():
            env._target_word = target
        else:
            env._target_word = WORDS[level][0]

        # Validate guess
        if not guess or not guess.isalpha() or len(guess) != word_len:
            return JSONResponse(status_code=200, content={
                "score":   0.05,
                "valid":   False,
                "reason":  f"Guess must be {word_len} alpha letters (got '{guess}').",
                "task_id": task_id,
            })

        # Use the env's own methods
        feedback = env._compute_feedback(guess, env._target_word)
        raw      = env._compute_reward(guess, env._target_word, attempt_num=1)
        solved   = (guess == env._target_word)

        # CRITICAL: clamp to strictly (0, 1) — never 0.0 or 1.0
        score = clamp_score(raw)

        return {
            "score":    score,
            "valid":    True,
            "solved":   solved,
            "feedback": feedback,
            "task_id":  task_id,
            "guess":    guess,
            "target":   env._target_word,
        }

    except Exception as e:
        return JSONResponse(status_code=200, content={
            "score":   0.05,
            "valid":   False,
            "error":   str(e),
            "task_id": "wordpuzzle-easy",
        })


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
