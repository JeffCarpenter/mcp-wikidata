"""Client for interacting with the Wikidata MCP server.

This module defines a pydantic ``Config`` object that encapsulates both
environment-driven settings and server configuration. It connects to the local
MCP server defined in ``server.py`` and queries it using LangChain's React
agent.
"""

from __future__ import annotations

from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseSettings, Field


class Config(BaseSettings):
    """Settings for the MCP client and server.

    Attributes:
        openai_api_key: API key for OpenAI. Loaded from ``OPENAI_API_KEY``.
        model: Name of the OpenAI chat model to use. Can be overridden with
            ``MCP_MODEL``.
        prompt: Default system prompt for the React agent. Can be overridden
            with ``MCP_PROMPT``.
        server_cmd: Executable used to launch the MCP server.
        server_script: Path to the MCP server script.
    """

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model: str = Field("gpt-4o", env="MCP_MODEL")
    prompt: str = Field(
        "You are a helpful assistant. Answer the user's questions based on Wikidata.",
        env="MCP_PROMPT",
    )
    server_cmd: str = "python"
    server_script: Path = Path(__file__).resolve().parent / "server.py"

    def stdio_params(self) -> StdioServerParameters:
        """Return parameters for launching the MCP server via stdio."""

        return StdioServerParameters(
            command=self.server_cmd, args=[str(self.server_script)]
        )


async def main() -> None:
    cfg = Config()

    model = ChatOpenAI(model=cfg.model, api_key=cfg.openai_api_key)
    server_params = cfg.stdio_params()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            agent = create_react_agent(model, tools, prompt=cfg.prompt)

            agent_response = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Can you recommend me a movie directed by Bong Joonho?",
                        }
                    ]
                }
            )
            print(agent_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
