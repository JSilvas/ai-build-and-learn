"""04 — Train an LLM to navigate mazes with GRPO.

Group Relative Policy Optimization: play episodes in groups, compute
advantages relative to the group mean, and update the policy with
a policy gradient.

Run:
    python 04_grpo_train.py

Options (edit constants below or extend with argparse):
    TRAINING_STEPS      Number of GRPO iterations
    ROLLOUTS_PER_STEP   Episodes per training step
    GROUP_SIZE          Episodes per advantage group
    EVAL_EPISODES       Episodes for evaluation
"""

import json
import os
import random
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from maze_env.models import MazeAction
from env_helpers import start_server, create_client, print_grid

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
TRAINING_STEPS = 3
ROLLOUTS_PER_STEP = 8
GROUP_SIZE = 4
EVAL_EPISODES = 20
LR = 1e-5
USE_BFLOAT16 = True
GRADIENT_CHECKPOINTING = True

DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

SYSTEM_PROMPT = (
    "You are navigating an 8x8 maze.\n"
    "The grid uses: # = wall, . = open path, A = you (agent), E = exit\n"
    "You MUST respond with exactly one word: UP, DOWN, LEFT, or RIGHT.\n"
    "Strategy: Find the shortest path from A to E while avoiding walls (#)."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def format_observation(grid, agent_pos, exit_pos, steps_taken):
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
    upper = text.strip().upper()
    for d in DIRECTIONS:
        if d in upper:
            return d
    return random.choice(DIRECTIONS)


# ---------------------------------------------------------------------------
# Episode playing (with trajectory for training)
# ---------------------------------------------------------------------------


def play_episode_train(client, model, tokenizer, device, temperature=0.7):
    """Play one episode, collecting trajectory data for GRPO.

    Returns: (trajectory, total_reward, solved)
    """
    result = client.reset()
    trajectory = []
    total_reward = 0.0
    steps = 0

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
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                return_dict_in_generate=True,
                output_scores=True,
            )

        gen_ids = outputs.sequences[0, prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        log_probs = []
        for i, score in enumerate(outputs.scores):
            if i < len(gen_ids):
                lp = torch.log_softmax(score[0], dim=-1)
                log_probs.append(lp[gen_ids[i]].item())

        direction = parse_direction(gen_text)
        result = client.step(MazeAction(direction=direction))

        steps += 1
        reward = result.reward or 0.0
        total_reward += reward

        trajectory.append({
            "prompt_ids": inputs["input_ids"][0].tolist(),
            "completion_ids": gen_ids.tolist(),
            "log_probs": log_probs,
            "action": gen_text.strip(),
        })

    solved = result.done and (result.reward or 0) >= 10.0
    return trajectory, total_reward, solved


def play_episode_eval(client, model, tokenizer, device):
    """Play one episode greedily (temperature=0) for evaluation."""
    result = client.reset()
    total_reward = 0.0
    steps = 0

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
                do_sample=False,
            )

        gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        direction = parse_direction(gen_text)

        result = client.step(MazeAction(direction=direction))
        total_reward += result.reward or 0.0
        steps += 1

    solved = result.done and (result.reward or 0) >= 10.0
    return solved, steps, total_reward


# ---------------------------------------------------------------------------
# GRPO training step
# ---------------------------------------------------------------------------


def grpo_step(model, tokenizer, client, device, optimizer, step_idx):
    """One GRPO iteration: rollouts -> group advantages -> policy gradient."""
    model.train()
    all_rewards = []
    all_solved = []
    total_loss = 0.0
    num_groups = max(ROLLOUTS_PER_STEP // GROUP_SIZE, 1)

    for g in range(num_groups):
        group_trajs = []
        group_rewards = []

        for _ in range(GROUP_SIZE):
            traj, reward, solved = play_episode_train(client, model, tokenizer, device)
            group_trajs.append(traj)
            group_rewards.append(reward)
            all_solved.append(solved)

        all_rewards.extend(group_rewards)

        # Group-relative advantages
        mean_r = sum(group_rewards) / len(group_rewards)
        std_r = (sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
        std_r = max(std_r, 1e-8)
        advantages = [(r - mean_r) / std_r for r in group_rewards]

        # Policy gradient update — accumulate grads one step at a time to save memory
        optimizer.zero_grad()
        group_loss = 0.0

        for traj, adv in zip(group_trajs, advantages):
            for step_data in traj:
                if not step_data["completion_ids"]:
                    continue
                prompt_t = torch.tensor([step_data["prompt_ids"]], device=device)
                comp_t = torch.tensor([step_data["completion_ids"]], device=device)
                full_ids = torch.cat([prompt_t, comp_t], dim=1)

                out = model(full_ids)
                logits = out.logits[0, prompt_t.shape[1] - 1 : -1]
                lp = torch.log_softmax(logits, dim=-1)
                token_lp = lp.gather(1, comp_t[0].unsqueeze(1)).squeeze(1)
                step_loss = -token_lp.sum() * adv
                step_loss.backward()
                group_loss += step_loss.item()

                del out, logits, lp, token_lp, step_loss, prompt_t, comp_t, full_ids

        optimizer.step()
        total_loss += group_loss
        cleanup_memory()

    solve_rate = sum(all_solved) / len(all_solved) if all_solved else 0
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    avg_loss = total_loss / num_groups

    print(f"  Step {step_idx}:  solve_rate={solve_rate:.2f}  avg_reward={avg_reward:.2f}  loss={avg_loss:.2f}")
    return solve_rate, avg_reward


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(model, tokenizer, client, device, num_episodes=EVAL_EPISODES, label=""):
    """Evaluate the model over N episodes."""
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    solve_count = 0
    rewards = []
    solve_steps = []

    for _ in range(num_episodes):
        solved, steps, reward = play_episode_eval(client, model, tokenizer, device)
        rewards.append(reward)
        if solved:
            solve_count += 1
            solve_steps.append(steps)

    solve_rate = solve_count / num_episodes
    avg_reward = sum(rewards) / len(rewards)
    avg_steps = sum(solve_steps) / len(solve_steps) if solve_steps else 0

    print(f"  {label:20s}  solve_rate={solve_rate:.2f}  avg_steps={avg_steps:.1f}  avg_reward={avg_reward:.2f}")

    # Re-enable gradient checkpointing for training
    if GRADIENT_CHECKPOINTING and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    return {"solve_rate": solve_rate, "avg_steps": avg_steps, "avg_reward": avg_reward}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    server, thread = start_server()
    client = create_client()

    device = get_device()
    dtype = torch.bfloat16 if USE_BFLOAT16 and device.type in ("cuda", "mps") else torch.float32
    print(f"\nDevice: {device}  dtype: {dtype}")
    print(f"Model: {MODEL_ID}")
    print(f"Training: {TRAINING_STEPS} steps, {ROLLOUTS_PER_STEP} rollouts/step, group_size={GROUP_SIZE}\n")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Evaluate untrained
    print("\n=== Untrained Model ===")
    eval_results = [evaluate(model, tokenizer, client, device, label="Untrained")]

    # Training loop
    for step in range(1, TRAINING_STEPS + 1):
        print(f"\n=== Training Step {step}/{TRAINING_STEPS} ===")
        grpo_step(model, tokenizer, client, device, optimizer, step)

        print(f"\n  Evaluating...")
        result = evaluate(model, tokenizer, client, device, label=f"After step {step}")
        eval_results.append(result)

    # Save final checkpoint
    save_dir = "checkpoint_final"
    print(f"\nSaving checkpoint to {save_dir}/")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    labels = ["Untrained"] + [f"Step {i}" for i in range(1, TRAINING_STEPS + 1)]
    for label, result in zip(labels, eval_results):
        print(f"  {label:15s}  solve_rate={result['solve_rate']:.2f}  avg_reward={result['avg_reward']:.2f}")

    improvement = eval_results[-1]["solve_rate"] - eval_results[0]["solve_rate"]
    print(f"\nSolve rate change: {improvement:+.2f}")

    client.close()
    cleanup_memory()
    print("\nDone!")


if __name__ == "__main__":
    main()