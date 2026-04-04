"""03 — Let an LLM play the maze (no training).

Loads SmolLM2-135M-Instruct and has it navigate mazes using its
pretrained knowledge. Spoiler: it's pretty bad out of the box.

Run:
    python 03_llm_play.py
"""

import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from maze_env.models import MazeAction
from env_helpers import start_server, create_client, print_grid

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

SYSTEM_PROMPT = (
    "You are navigating an 8x8 maze.\n"
    "The grid uses: # = wall, . = open path, A = you (agent), E = exit\n"
    "You MUST respond with exactly one word: UP, DOWN, LEFT, or RIGHT.\n"
    "Strategy: Find the shortest path from A to E while avoiding walls (#)."
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def format_observation(grid, agent_pos, exit_pos, steps_taken):
    """Render the maze grid as text for the LLM."""
    rows = [" ".join(row) for row in grid]
    grid_text = "\n".join(rows)
    return (
        f"Maze:\n{grid_text}\n\n"
        f"Your position: row={agent_pos[0]}, col={agent_pos[1]}\n"
        f"Exit position: row={exit_pos[0]}, col={exit_pos[1]}\n"
        f"Steps taken: {steps_taken}\n"
        "Which direction? Reply UP, DOWN, LEFT, or RIGHT."
    )


def parse_direction(text):
    """Extract direction from model output."""
    upper = text.strip().upper()
    for d in DIRECTIONS:
        if d in upper:
            return d
    return random.choice(DIRECTIONS)


def play_episode(client, model, tokenizer, device, verbose=False):
    """Play one maze episode with the LLM. Returns (solved, steps, total_reward)."""
    result = client.reset()
    total_reward = 0.0
    steps = 0

    if verbose:
        print("\nStarting maze:")
        print_grid(result.observation.grid)

    while not result.done and steps < 100:
        obs = result.observation
        user_prompt = format_observation(obs.grid, obs.agent_pos, obs.exit_pos, obs.steps_taken)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=True,
                temperature=0.7,
            )

        gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        direction = parse_direction(gen_text)

        result = client.step(MazeAction(direction=direction))
        reward = result.reward or 0.0
        total_reward += reward
        steps += 1

        if verbose:
            print(f"  Step {steps}: model said '{gen_text.strip()}' -> {direction}  reward={reward:.2f}")

    solved = result.done and (result.reward or 0) >= 10.0

    if verbose:
        print(f"\n{'SOLVED!' if solved else 'Failed.'} Steps: {steps}  Total reward: {total_reward:.2f}")
        print_grid(result.observation.grid)

    return solved, steps, total_reward


def main():
    server, thread = start_server()
    client = create_client()

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Loading model: {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    model.eval()
    print("Model loaded.\n")

    # Play one verbose episode so we can see what's happening
    print("=== Verbose Episode ===")
    play_episode(client, model, tokenizer, device, verbose=True)

    # Evaluate over multiple episodes
    num_episodes = 20
    print(f"\n=== Evaluating over {num_episodes} episodes ===")
    solve_count = 0
    total_steps = []
    total_rewards = []

    for i in range(num_episodes):
        solved, steps, reward = play_episode(client, model, tokenizer, device)
        total_rewards.append(reward)
        total_steps.append(steps)
        if solved:
            solve_count += 1
        print(f"  Episode {i+1}: {'SOLVED' if solved else 'failed':7s}  steps={steps:3d}  reward={reward:.2f}")

    solve_rate = solve_count / num_episodes
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nResults: solve_rate={solve_rate:.2f}  avg_reward={avg_reward:.2f}")
    print(f"\nThe untrained LLM is basically random — GRPO training should improve this!")

    client.close()


if __name__ == "__main__":
    main()