"""Shared helpers for starting the env server and creating clients."""

import threading
import time

import uvicorn
from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from maze_env.models import MazeAction, MazeObservation, MazeState


ENV_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MazeEnv(EnvClient[MazeAction, MazeObservation, MazeState]):
    """Client that connects to a Maze OpenEnv server."""

    def _step_payload(self, action: MazeAction) -> dict:
        return {"direction": action.direction}

    def _parse_result(self, payload: dict) -> StepResult[MazeObservation]:
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

    def _parse_state(self, payload: dict) -> MazeState:
        return MazeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            maze_seed=payload.get("maze_seed", 0),
            optimal_path_length=payload.get("optimal_path_length", 0),
        )


def create_client(env_url: str = ENV_URL):
    """Create and connect a sync maze env client."""
    async_client = MazeEnv(
        base_url=env_url,
        connect_timeout_s=30.0,
        message_timeout_s=300.0,
    )
    client = async_client.sync()
    client.connect()
    return client


# ---------------------------------------------------------------------------
# Server (background thread)
# ---------------------------------------------------------------------------


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the maze env server in a background thread. Returns the thread."""
    from openenv.core.env_server import create_app
    from maze_env.models import MazeAction, MazeObservation
    from maze_env.server.environment import MazeEnvironment

    app = create_app(
        MazeEnvironment,
        MazeAction,
        MazeObservation,
        env_name="maze",
        max_concurrent_envs=8,
    )

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(2)  # Wait for server to be ready
    print(f"Maze env server running at http://{host}:{port}")
    return server, thread


def print_grid(grid: list[list[str]]):
    """Pretty-print a maze grid."""
    for row in grid:
        print(" ".join(row))