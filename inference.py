inference.py — WordPuzzle OpenEnv RL Environment
Hackathon submission inference script.
"""

import os
import textwrap
from typing import List, Optional
from openai import OpenAI

from run_standalone import (
    WordPuzzleEnvironment,
    WordPuzzleAction,
    LEVEL_CONFIG,
)

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = "wordpuzzle"
BENCHMARK = "wordpuzzle-openenv"

SYSTEM_PROMPT = textwrap.dedent("""
    You are playing a Wordle-style word guessing game.
    You will be told the word length and given feedback on your previous guesses.
    Feedback format per letter:
      correct = right letter, right position
      present = right letter, wrong position
      absent  = letter not in word
    Reply with exactly one word of the correct length.
    No explanation. No punctuation. Just the word in UPPERCASE.
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def format_feedback(feedback: list) -> str:
    """Convert feedback list to readable string for LLM."""
    parts = []
    for f in feedback:
        symbol = {"correct": "[+]", "present": "[?]", "absent": "[-]"}[f["status"]]
        parts.append(f"{f['letter'].upper()}{symbol}")
    return " ".join(parts)


def get_model_guess(
    client: OpenAI,
    word_length: int,
    feedback_history: List[str],
    step: int,
) -> str:
    history_block = "\n".join(feedback_history[-6:]) if feedback_history else "None"
    user_prompt = textwrap.dedent(f"""
        Word length: {word_length} letters
        Step: {step}
        Previous guesses and feedback:
        {history_block}
        Your next guess (exactly {word_length} letters, uppercase):
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=20,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip().upper()
        # Extract valid word of correct length
        for word in text.split():
            if len(word) == word_length and word.isalpha():
                return word
        # Fallback: trim or pad
        return text[:word_length] if text.isalpha() else ("CRANE"[:word_length])
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        fallbacks = {4: "CARE", 5: "CRANE", 6: "BRIDGE"}
        return fallbacks.get(word_length, "CRANE"[:word_length])


def run_level(client: OpenAI, level: int) -> dict:
    cfg = LEVEL_CONFIG[level]
    word_length = cfg["word_len"]
    max_attempts = cfg["max_attempts"]

    env = WordPuzzleEnvironment(task_level=level)
    obs = env.reset()

    feedback_history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    done = False

    for step in range(1, max_attempts + 1):
        if done:
            break

        guess = get_model_guess(client, word_length, feedback_history, step)
        action = WordPuzzleAction(guess=guess)

        try:
            obs, reward, done = env.step(action)
            error = None
        except Exception as e:
            error = str(e)
            reward = 0.0
            done = True

        rewards.append(reward)
        steps_taken = step

        # Build feedback string for LLM history
        if obs.feedback:
            fb_str = format_feedback(obs.feedback)
            feedback_history.append(f"Step {step}: {guess} → {fb_str} (reward {reward:+.2f})")

        log_step(step=step, action=guess, reward=reward, done=done, error=error)

        if done:
            break

    return {
        "solved": obs.solved,
        "steps": steps_taken,
        "rewards": rewards,
        "max_attempts": max_attempts,
    }


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    all_rewards: List[float] = []
    all_steps = 0
    level_scores: List[float] = []

    for level in [1, 2, 3]:
        result = run_level(client, level)

        all_rewards.extend(result["rewards"])
        all_steps += result["steps"]

        level_score = sum(result["rewards"]) / result["max_attempts"]
        level_score = min(max(level_score, 0.0), 1.0)
        level_scores.append(level_score)

    final_score = sum(level_scores) / len(level_scores)
    final_score = min(max(final_score, 0.0), 1.0)
    success = final_score >= 0.1

    log_end(
        success=success,
        steps=all_steps,
        score=final_score,
        rewards=all_rewards,
    )


if __name__ == "__main__":
    main()