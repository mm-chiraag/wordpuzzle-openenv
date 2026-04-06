# WordPuzzle — OpenEnv RL Environment

A Wordle-style mini-game reinforcement learning environment built with the OpenEnv framework.

## Quick Start

### Option 1: Run directly (no install needed)
```bash
python run_standalone.py
```

### Option 2: Run with Docker
```bash
# Build the image
docker build -t wordpuzzle-env .

# Run it
docker run wordpuzzle-env
```

### Option 3: Hugging Face Spaces
Push this repo to a Hugging Face Space (Docker SDK).

---

## Environment Details

| Property | Value |
|---|---|
| Framework | OpenEnv (Gymnasium-style) |
| Interface | `reset()` / `step()` / `state()` |
| Action | `WordPuzzleAction(guess="word")` |
| Observation | `WordPuzzleObservation(feedback, attempts_used, solved, ...)` |

## Tasks (3 difficulty levels)

| Level | Word Length | Max Attempts | Label |
|---|---|---|---|
| 1 | 4 letters | 6 | Easy |
| 2 | 5 letters | 5 | Medium |
| 3 | 6 letters | 4 | Hard |

## Reward Function

| Event | Reward |
|---|---|
| Letter in correct position | +1.0 |
| Letter present but wrong position | +0.3 |
| Solve bonus (scales with efficiency) | +1.0 to +3.0 |
| Invalid guess | −0.5 |

## How to Use the Environment in Your Own Agent

```python
from run_standalone import WordPuzzleEnvironment, WordPuzzleAction

env = WordPuzzleEnvironment(task_level=1)
obs = env.reset()

while True:
    action = WordPuzzleAction(guess="care")   # your agent's guess
    obs, reward, done = env.step(action)
    if done:
        break
```

## Evaluation Output

Running the file produces an automated grader report:
- ✅ Runtime correctness
- ✅ Interface compliance
- ✅ Task design (3 levels, increasing difficulty)
- ✅ Grading logic (per-step reward + efficiency bonus)
