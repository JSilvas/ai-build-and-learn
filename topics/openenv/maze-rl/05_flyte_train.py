"""05 — Train maze RL on a Flyte cluster.

Same GRPO training as 04, but packaged as a Flyte task so it can run
on a cluster with GPU. The env server is co-located in the same container
(no separate deployment, no network overhead).

Run locally:
    flyte run --local 05_flyte_train.py pipeline --training_steps 2 --eval_episodes 5

Run on cluster:
    flyte run 05_flyte_train.py pipeline --training_steps 10 --rollouts_per_step 16
"""

import json
import os
import random
import shutil
import sys
import threading
import time
from pathlib import Path

import flyte
import flyte.report
from flyte.io import File

# ---------------------------------------------------------------------------
# Flyte task environment — defines the container image + resources
# ---------------------------------------------------------------------------

env = flyte.TaskEnvironment(
    name="maze_rl",
    image=flyte.Image.from_debian_base()
    .with_pip_packages(
        "torch",
        "transformers",
        "openenv-core",
        "matplotlib",
        "uvicorn",
    )
    .with_source_folder(Path(__file__).parent / "maze_env"),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

SYSTEM_PROMPT = (
    "You are navigating an 8x8 maze.\n"
    "The grid uses: # = wall, . = open path, A = you (agent), E = exit\n"
    "You MUST respond with exactly one word: UP, DOWN, LEFT, or RIGHT.\n"
    "Strategy: Find the shortest path from A to E while avoiding walls (#)."
)


# ---------------------------------------------------------------------------
# Env server + client helpers (co-located in same container)
# ---------------------------------------------------------------------------


def start_local_env_server(port: int = 8000):
    """Start the maze OpenEnv server in a background thread."""
    import uvicorn

    # Ensure maze_env is importable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from openenv.core.env_server import create_app
    from maze_env.models import MazeAction, MazeObservation
    from maze_env.server.environment import MazeEnvironment

    app = create_app(
        MazeEnvironment, MazeAction, MazeObservation,
        env_name="maze", max_concurrent_envs=8,
    )
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(2)
    print(f"  Env server running on localhost:{port}")
    return server


def create_client(port: int = 8000):
    """Create a sync maze env client connected to localhost."""
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
    from maze_env.models import MazeAction, MazeObservation, MazeState

    class MazeEnv(EnvClient[MazeAction, MazeObservation, MazeState]):
        def _step_payload(self, action):
            return {"direction": action.direction}

        def _parse_result(self, payload):
            obs_data = payload.get("observation", payload)
            observation = MazeObservation(
                grid=obs_data.get("grid", []),
                agent_pos=tuple(obs_data.get("agent_pos", (1, 1))),
                exit_pos=tuple(obs_data.get("exit_pos", (6, 6))),
                steps_taken=obs_data.get("steps_taken", 0),
                done=payload.get("done", False),
                reward=payload.get("reward"),
            )
            return StepResult(
                observation=observation,
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload):
            return MazeState(
                episode_id=payload.get("episode_id"),
                step_count=payload.get("step_count", 0),
                maze_seed=payload.get("maze_seed", 0),
                optimal_path_length=payload.get("optimal_path_length", 0),
            )

    async_client = MazeEnv(
        base_url=f"http://localhost:{port}",
        connect_timeout_s=30.0,
        message_timeout_s=300.0,
    )
    client = async_client.sync()
    client.connect()
    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup_memory():
    import gc
    import torch
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
# Episode playing
# ---------------------------------------------------------------------------


def play_episode_train(client, model, tokenizer, device, temperature=0.7):
    """Play one episode collecting trajectory data for GRPO."""
    import torch
    from maze_env.models import MazeAction

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
    """Play one episode greedily for evaluation."""
    import torch
    from maze_env.models import MazeAction

    result = client.reset()
    total_reward = 0.0
    steps = 0
    direction_counts = {d: 0 for d in DIRECTIONS}

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
            outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)

        gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        direction = parse_direction(gen_text)
        direction_counts[direction] += 1

        result = client.step(MazeAction(direction=direction))
        total_reward += result.reward or 0.0
        steps += 1

    solved = result.done and (result.reward or 0) >= 10.0
    return solved, steps, total_reward, direction_counts


def play_episode_baseline(client, policy="random"):
    """Play one episode with a simple baseline policy."""
    from maze_env.models import MazeAction

    result = client.reset()
    total_reward = 0.0
    steps = 0
    visited = set()

    while not result.done and steps < 100:
        obs = result.observation
        agent_r, agent_c = obs.agent_pos
        exit_r, exit_c = obs.exit_pos
        grid = obs.grid

        if policy == "wall_follower":
            visited.add((agent_r, agent_c))
            best_dir, best_dist = None, float("inf")
            for d in DIRECTIONS:
                dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[d]
                nr, nc = agent_r + dr, agent_c + dc
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "#":
                    dist = abs(nr - exit_r) + abs(nc - exit_c)
                    if (nr, nc) in visited:
                        dist += 5
                    if dist < best_dist:
                        best_dist = dist
                        best_dir = d
            direction = best_dir or random.choice(DIRECTIONS)
        else:
            direction = random.choice(DIRECTIONS)

        result = client.step(MazeAction(direction=direction))
        total_reward += result.reward or 0.0
        steps += 1

    solved = result.done and (result.reward or 0) >= 10.0
    return solved, steps, total_reward


# ---------------------------------------------------------------------------
# Pipeline (single Flyte task with report)
# ---------------------------------------------------------------------------


@env.task(report=True)
async def pipeline(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    training_steps: int = 5,
    rollouts_per_step: int = 8,
    group_size: int = 4,
    eval_episodes: int = 20,
    lr: float = 1e-5,
    use_bfloat16: bool = True,
    gradient_checkpointing: bool = True,
) -> tuple[str, File]:
    """Full Maze RL pipeline: baselines -> GRPO training -> eval -> report.

    The env server runs co-located in the same container — no separate
    deployment needed, zero network overhead.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Start co-located env server ---
    start_local_env_server()
    client = create_client()

    device = get_device()
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Model:  {model_id}")
    print(f"Config: {training_steps} steps, {rollouts_per_step} rollouts/step, group_size={group_size}\n")

    # --- Baselines ---
    print("=== Baselines ===")
    baselines = {}
    for policy in ["random", "wall_follower"]:
        solve_count, rewards, solve_steps_list = 0, [], []
        for _ in range(50):
            solved, steps, reward = play_episode_baseline(client, policy)
            rewards.append(reward)
            if solved:
                solve_count += 1
                solve_steps_list.append(steps)
        baselines[policy] = {
            "solve_rate": solve_count / 50,
            "avg_steps": sum(solve_steps_list) / len(solve_steps_list) if solve_steps_list else 0,
            "avg_reward": sum(rewards) / len(rewards),
        }
        print(f"  {policy:15s}  solve_rate={baselines[policy]['solve_rate']:.2f}")

    # --- Load model ---
    print(f"\nLoading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # --- Evaluate + Train loop ---
    def run_eval(label, step_idx):
        model.eval()
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()

        solve_count, rewards, solve_steps_list = 0, [], []
        all_dir_counts = {d: 0 for d in DIRECTIONS}
        for _ in range(eval_episodes):
            solved, steps, reward, dir_counts = play_episode_eval(client, model, tokenizer, device)
            rewards.append(reward)
            if solved:
                solve_count += 1
                solve_steps_list.append(steps)
            for d, c in dir_counts.items():
                all_dir_counts[d] += c

        if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        result = {
            "step": step_idx,
            "solve_rate": solve_count / eval_episodes,
            "avg_steps": sum(solve_steps_list) / len(solve_steps_list) if solve_steps_list else 0,
            "avg_reward": sum(rewards) / len(rewards),
            "direction_distribution": all_dir_counts,
        }
        print(f"  {label:20s}  solve_rate={result['solve_rate']:.2f}  avg_reward={result['avg_reward']:.2f}")
        return result

    print("\n=== Untrained ===")
    eval_results = [run_eval("Untrained", 0)]

    for step in range(1, training_steps + 1):
        print(f"\n=== Training Step {step}/{training_steps} ===")
        model.train()
        all_rewards, all_solved = [], []
        total_loss = 0.0
        num_groups = max(rollouts_per_step // group_size, 1)

        for g in range(num_groups):
            group_trajs, group_rewards = [], []
            for _ in range(group_size):
                traj, reward, solved = play_episode_train(client, model, tokenizer, device)
                group_trajs.append(traj)
                group_rewards.append(reward)
                all_solved.append(solved)
            all_rewards.extend(group_rewards)

            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = max((sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5, 1e-8)
            advantages = [(r - mean_r) / std_r for r in group_rewards]

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

        train_solve = sum(all_solved) / len(all_solved) if all_solved else 0
        train_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        print(f"  Train: solve_rate={train_solve:.2f}  avg_reward={train_reward:.2f}  loss={total_loss / num_groups:.2f}")

        print(f"  Evaluating...")
        eval_results.append(run_eval(f"After step {step}", step))

    # --- Save checkpoint ---
    save_dir = "checkpoint_final"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    tar_path = f"{save_dir}.tar.gz"
    shutil.make_archive(save_dir, "gztar", ".", save_dir)
    checkpoint_file = await File.from_local(tar_path)

    # --- Generate report ---
    import base64
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_list = [e["step"] for e in eval_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Solve rate
    ax = axes[0]
    ax.plot(steps_list, [e["solve_rate"] for e in eval_results], "b-o", linewidth=2, label="GRPO Agent")
    ax.axhline(baselines["random"]["solve_rate"], color="r", linestyle="--",
               label=f"Random ({baselines['random']['solve_rate']:.2f})")
    ax.axhline(baselines["wall_follower"]["solve_rate"], color="g", linestyle="--",
               label=f"Wall-follower ({baselines['wall_follower']['solve_rate']:.2f})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Solve Rate")
    ax.set_title("Solve Rate Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Avg steps
    ax = axes[1]
    ax.plot(steps_list, [e["avg_steps"] for e in eval_results], "m-o", linewidth=2)
    if baselines["wall_follower"]["avg_steps"] > 0:
        ax.axhline(baselines["wall_follower"]["avg_steps"], color="g", linestyle="--",
                    label=f"Wall-follower ({baselines['wall_follower']['avg_steps']:.0f})")
        ax.legend()
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Avg Steps to Solve")
    ax.set_title("Efficiency (lower = better)")
    ax.grid(True, alpha=0.3)

    # Direction distribution
    ax = axes[2]
    for d in DIRECTIONS:
        fracs = []
        for e in eval_results:
            dist = e.get("direction_distribution", {})
            total = sum(dist.values())
            fracs.append(dist.get(d, 0) / max(total, 1))
        ax.plot(steps_list, fracs, "-o", markersize=4, label=d)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Fraction")
    ax.set_title("Direction Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    charts_b64 = base64.b64encode(buf.getvalue()).decode()
    charts_html = f'<img src="data:image/png;base64,{charts_b64}" />'
    plt.close(fig)

    final = eval_results[-1]
    await flyte.report.replace.aio(
        f"<h2>Maze RL Training Report</h2>"
        f"<h3>Results</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><th>Policy</th><th>Solve Rate</th><th>Avg Steps</th><th>Avg Reward</th></tr>"
        f"<tr><td>Random</td><td>{baselines['random']['solve_rate']:.2f}</td>"
        f"<td>{baselines['random']['avg_steps']:.0f}</td>"
        f"<td>{baselines['random']['avg_reward']:.2f}</td></tr>"
        f"<tr><td>Wall-follower</td><td>{baselines['wall_follower']['solve_rate']:.2f}</td>"
        f"<td>{baselines['wall_follower']['avg_steps']:.0f}</td>"
        f"<td>{baselines['wall_follower']['avg_reward']:.2f}</td></tr>"
        f"<tr><td><b>GRPO (untrained)</b></td><td>{eval_results[0]['solve_rate']:.2f}</td>"
        f"<td>{eval_results[0]['avg_steps']:.0f}</td>"
        f"<td>{eval_results[0]['avg_reward']:.2f}</td></tr>"
        f"<tr><td><b>GRPO (final)</b></td><td><b>{final['solve_rate']:.2f}</b></td>"
        f"<td><b>{final['avg_steps']:.0f}</b></td>"
        f"<td><b>{final['avg_reward']:.2f}</b></td></tr>"
        f"</table>"
        f"<h3>Training Progress</h3>{charts_html}"
        f"<h3>Config</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><td>Model</td><td>{model_id}</td></tr>"
        f"<tr><td>Training Steps</td><td>{training_steps}</td></tr>"
        f"<tr><td>Rollouts/Step</td><td>{rollouts_per_step}</td></tr>"
        f"<tr><td>Group Size</td><td>{group_size}</td></tr>"
        f"<tr><td>Learning Rate</td><td>{lr}</td></tr>"
        f"<tr><td>Device</td><td>{device}</td></tr>"
        f"</table>"
    )
    await flyte.report.flush.aio()

    summary = (
        f"Final solve_rate: {final['solve_rate']:.2f} "
        f"(random: {baselines['random']['solve_rate']:.2f}, "
        f"wall-follower: {baselines['wall_follower']['solve_rate']:.2f})"
    )
    print(f"\n{summary}")

    client.close()
    del model, optimizer
    cleanup_memory()

    return summary, checkpoint_file
