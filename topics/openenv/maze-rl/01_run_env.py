"""01 — Explore the Maze OpenEnv environment.

Starts the env server locally, connects a client, and steps through
a maze to see how the environment works.

Run:
    python 01_run_env.py
"""

from maze_env.models import MazeAction
from env_helpers import start_server, create_client, print_grid


def main():
    # Start the OpenEnv server in a background thread
    server, thread = start_server()

    # Connect a client
    client = create_client()

    # Reset creates a new random maze
    print("\n=== New Maze ===")
    result = client.reset()
    obs = result.observation
    print_grid(obs.grid)
    print(f"Agent: {obs.agent_pos}  Exit: {obs.exit_pos}")
    print(f"Done: {obs.done}  Reward: {result.reward}")

    # Take a few steps
    moves = ["RIGHT", "RIGHT", "DOWN", "DOWN", "RIGHT"]
    for direction in moves:
        print(f"\n--- Move: {direction} ---")
        result = client.step(MazeAction(direction=direction))
        obs = result.observation
        print_grid(obs.grid)
        print(f"Agent: {obs.agent_pos}  Reward: {result.reward:.2f}  Done: {obs.done}")

        if result.done:
            print("Episode finished!")
            break

    # Check the env state (metadata)
    state = client.state()
    print(f"\nState: seed={state.maze_seed}, optimal_path={state.optimal_path_length}, steps={state.step_count}")

    # Play a full episode with random moves
    print("\n\n=== Random Episode ===")
    import random
    result = client.reset()
    total_reward = 0.0
    steps = 0
    while not result.done and steps < 100:
        direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        result = client.step(MazeAction(direction=direction))
        total_reward += result.reward or 0.0
        steps += 1

    print_grid(result.observation.grid)
    print(f"Steps: {steps}  Total reward: {total_reward:.2f}  Solved: {result.observation.done and (result.reward or 0) >= 10.0}")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()