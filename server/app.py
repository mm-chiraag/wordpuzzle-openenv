import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request

from run_standalone import (
    WordPuzzleEnvironment,
    WordPuzzleAction,
    LEVEL_CONFIG,
    WORDS,
)

app = FastAPI(title="WordPuzzle OpenEnv")

# ✅ ROOT ENDPOINT FIX
@app.get("/")
def root():
    return {"message": "WordPuzzle API is running"}

env_sessions = {}

TASK_LEVEL_MAP = {
    "wordpuzzle-easy":   1,
    "wordpuzzle-medium": 2,
    "wordpuzzle-hard":   3,
}

def clamp_score(score: float) -> float:
    try:
        score = float(score)
    except:
        return 0.01
    if score <= 0:
        return 0.01
    if score >= 1:
        return 0.99
    score = min(score, 0.9999)
    score = max(score, 0.0001)
    return round(score, 4)

def _obs_dict(obs, reward=0.01, done=False):
    return {
        "observation": {
            "feedback": obs.feedback,
            "attempts_used": obs.attempts_used,
            "max_attempts": obs.max_attempts,
            "task_level": obs.task_level,
            "message": obs.message,
            "solved": obs.solved,
            "revealed_word": obs.revealed_word,
        },
        "reward": clamp_score(reward),
        "done": done,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def list_tasks():
    tasks = []
    for task_id, level in TASK_LEVEL_MAP.items():
        cfg = LEVEL_CONFIG[level]
        tasks.append({
            "id": task_id,
            "name": f"WordPuzzle {cfg['name']}",
            "grader": {"endpoint": "/grader", "field": "score", "range": [0.01, 0.99]},
        })
    return {"tasks": tasks}

@app.post("/reset")
async def reset(request: Request):
    body = await request.json()
    task_id = body.get("task_id", "wordpuzzle-easy")
    session_id = body.get("session_id", "default")

    level = TASK_LEVEL_MAP.get(task_id, 1)

    env = WordPuzzleEnvironment(task_level=level)
    obs = env.reset()
    env_sessions[session_id] = env

    result = _obs_dict(obs)
    result["task_id"] = task_id
    return result

@app.post("/step")
async def step(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "default")
    action_str = body.get("action", "")

    if session_id not in env_sessions:
        env = WordPuzzleEnvironment(task_level=1)
        env.reset()
        env_sessions[session_id] = env

    env = env_sessions[session_id]
    action = WordPuzzleAction(guess=action_str)

    try:
        obs, reward, done = env.step(action)
    except Exception:
        return {"reward": 0.05, "done": True}

    if done:
        env_sessions.pop(session_id, None)

    return _obs_dict(obs, reward, done)

@app.post("/grader")
async def grader(request: Request):
    try:
        body = await request.json()
    except:
        body = {}

    task_id = body.get("task_id", "wordpuzzle-easy")
    guess = (body.get("guess", "") or "").lower().strip()
    target = (body.get("target", "") or "").lower().strip()

    level = TASK_LEVEL_MAP.get(task_id, 1)
    cfg = LEVEL_CONFIG[level]
    word_len = cfg["word_length"]

    env = WordPuzzleEnvironment(task_level=level)
    env.reset()

    if target and len(target) == word_len:
        env._target_word = target
    else:
        env._target_word = WORDS[level][0]

    if not guess or len(guess) != word_len:
        return {"score": 0.05, "valid": False}

    raw = env._compute_reward(guess, env._target_word, 1)
    score = clamp_score(raw)

    return {"score": score, "valid": True, "task_id": task_id}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
