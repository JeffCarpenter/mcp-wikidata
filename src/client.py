"""Client for interacting with the Wikidata MCP server.

This module defines pydantic models for configuration and requests,
which provides validation and type safety for the client. It connects to the
local MCP server defined in ``server.py`` and queries it using LangChain's
React agent.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, BaseSettings, Field


class ClientSettings(BaseSettings):
    """Configuration for the MCP client.

    Attributes:
        openai_api_key: API key for OpenAI. Loaded from the ``OPENAI_API_KEY``
            environment variable.
        model: Name of the OpenAI chat model to use.
        prompt: Default system prompt for the React agent.
    """

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    model: str = "gpt-4o"
    prompt: str = (
        "You are a helpful assistant. Answer the user's questions based on Wikidata."
    )


class ServerConfig(BaseModel):
    """Configuration for starting the MCP server."""

    command: str = "python"
    script: Path = Path(__file__).resolve().parent / "server.py"

    def to_stdio_params(self) -> StdioServerParameters:
        """Convert the server configuration to ``StdioServerParameters``."""

        return StdioServerParameters(command=self.command, args=[str(self.script)])


class AgentRequest(BaseModel):
    """Schema for requests sent to the agent."""

    messages: str


async def main() -> None:
    settings = ClientSettings()
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    model = ChatOpenAI(model=settings.model)
    server_params = ServerConfig().to_stdio_params()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            agent = create_react_agent(model, tools, prompt=settings.prompt)

            request = AgentRequest(
                messages="Can you recommend me a movie directed by Bong Joonho?"
            )
            agent_response = await agent.ainvoke(request.model_dump())
            print(agent_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
