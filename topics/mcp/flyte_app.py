# /// script
# requires-python = "==3.12"
# dependencies = ["flyte==2.0.0b56"]
# ///

"""Flyte App deployment for the Demo MCP Server."""

import os
import logging
import pathlib

import flyte
from flyte.app import AppEnvironment, Domain, Scaling, Link


APP_NAME = os.getenv("APP_NAME", "demo-mcp-server")
APP_SUBDOMAIN = os.getenv("APP_SUBDOMAIN")
APP_PORT = int(os.getenv("APP_PORT", 8000))

# -----------------
# App Environment
# -----------------

image = (
    flyte.Image.from_debian_base(name="demo-mcp-server")
    .with_requirements("requirements.txt")
)

app_env = AppEnvironment(
    name=APP_NAME,
    port=APP_PORT,
    domain=Domain(subdomain=APP_SUBDOMAIN) if APP_SUBDOMAIN else None,
    include=["./server.py"],
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    scaling=Scaling(replicas=(0, 1)),
    requires_auth=False,
    env_vars={"APP_NAME": APP_NAME},
    links=[
        Link(
            path="/mcp",
            title="Streamable HTTP transport endpoint",
            is_relative=True,
        ),
        Link(
            path="/health",
            title="Health check endpoint",
            is_relative=True,
        ),
    ],
)


@app_env.server
async def server():
    # Imports deferred so flyte deploy doesn't need fastmcp/starlette installed
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.responses import PlainTextResponse

    from server import mcp

    async def health(request):
        return PlainTextResponse("OK")

    mcp_app = mcp.http_app()

    starlette_app = Starlette(
        lifespan=mcp_app.lifespan,
        routes=[
            Route("/health", health),
            Mount("/", app=mcp_app),
        ],
    )

    uvicorn_server = uvicorn.Server(uvicorn.Config(starlette_app, port=APP_PORT))
    await uvicorn_server.serve()


if __name__ == "__main__":
    flyte.init_from_config(
        root_dir=pathlib.Path(__file__).parent,
        log_level=logging.INFO,
    )

    served_app = flyte.serve(app_env)
    print(f"Served app: {served_app.url}")
