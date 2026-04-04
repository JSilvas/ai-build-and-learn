"""02 — Baseline policies for maze navigation.

Runs random and wall-follower (greedy manhattan) policies to establish
performance baselines before training an LLM.

Run:
    python 02_baselines.py
"""

import random

from maze_env.models import MazeAction
from env_helpers import start_server, create_client

DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


def play_random(client, max_steps=100):
    """Random policy: pick a random direction each step."""
    result = client.reset()
    total_reward = 0.0
    steps = 0
    while not result.done and steps < max_steps:
        direction = random.choice(DIRECTIONS)
        result = client.step(MazeAction(direction=direction))
        total_reward += result.reward or 0.0
        steps += 1
    solved = result.done and (result.reward or 0) >= 10.0
    return solved, steps, total_reward


def play_wall_follower(client, max_steps=100):
    """Greedy manhattan policy with visited-cell avoidance."""
    result = client.reset()
    total_reward = 0.0
    steps = 0
    visited = set()

    while not result.done and steps < max_steps:
        obs = result.observation
        agent_r, agent_c = obs.agent_pos
        exit_r, exit_c = obs.exit_pos
        grid = obs.grid
        visited.add((agent_r, agent_c))

        best_dir = None
        best_dist = float("inf")
        for d in DIRECTIONS:
            dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[d]
            nr, nc = agent_r + dr, agent_c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "#":
                dist = abs(nr - exit_r) + abs(nc - exit_c)
                if (nr, nc) in visited:
                    dist += 5  # Penalize revisits
                if dist < best_dist:
                    best_dist = dist
                    best_dir = d

        direction = best_dir or random.choice(DIRECTIONS)
        result = client.step(MazeAction(direction=direction))
        total_reward += result.reward or 0.0
        steps += 1

    solved = result.done and (result.reward or 0) >= 10.0
    return solved, steps, total_reward


def evaluate_policy(name, play_fn, client, num_episodes=50):
    """Run a policy for N episodes and report stats."""
    solve_count = 0
    solve_steps = []
    rewards = []

    for _ in range(num_episodes):
        solved, steps, reward = play_fn(client)
        rewards.append(reward)
        if solved:
            solve_count += 1
            solve_steps.append(steps)

    solve_rate = solve_count / num_episodes
    avg_steps = sum(solve_steps) / len(solve_steps) if solve_steps else 0
    avg_reward = sum(rewards) / len(rewards)

    print(f"  {name:15s}  solve_rate={solve_rate:.2f}  avg_steps={avg_steps:.1f}  avg_reward={avg_reward:.2f}")
    return {"solve_rate": solve_rate, "avg_steps": avg_steps, "avg_reward": avg_reward}


def main():
    server, thread = start_server()
    client = create_client()

    num_episodes = 50
    print(f"\nRunning baselines ({num_episodes} episodes each):\n")

    random_stats = evaluate_policy("Random", play_random, client, num_episodes)
    follower_stats = evaluate_policy("Wall-follower", play_wall_follower, client, num_episodes)

    print(f"\n--- Summary ---")
    print(f"Random solves {random_stats['solve_rate']:.0%} of mazes")
    print(f"Wall-follower solves {follower_stats['solve_rate']:.0%} of mazes")
    print(f"\nThese are the baselines an LLM agent needs to beat!")

    client.close()


if __name__ == "__main__":
    main()