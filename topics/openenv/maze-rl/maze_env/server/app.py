"""FastAPI app serving the Maze environment via OpenEnv protocol.

Run directly:
    python -m maze_env.server.app
"""

import uvicorn
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
