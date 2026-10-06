"""Opt-in web search through Parallel's anonymous Streamable HTTP MCP endpoint.

    uv sync --locked --extra mcp
    OPENAI_API_KEY=... uv run --locked --extra mcp python examples/17_parallel_search_mcp.py

Parallel needs no key for light use; the LLM still needs its provider credentials.
Approve remote calls at the prompt. Tools and the agent run share one connection.
"""
import argparse
import asyncio

from agentmaker import Agent, LLMClient, MCPClient, __version__, cli_confirm


async def search(question, llm, *, confirm=cli_confirm):
    """Run an agent with only Parallel's search and fetch tools, while connected."""
    async with MCPClient(
        url="https://search.parallel.ai/mcp",
        namespace="parallel",
        headers={"User-Agent": f"agentmaker/{__version__}"},
        timeout=60.0,
    ) as client:
        tools = await client.load_tools()
        tools = [tool for tool in tools if tool.name in {
            "parallel_web_search", "parallel_web_fetch",
        }]
        agent = Agent(
            "web_researcher", llm, tools=tools, confirm=confirm,
            system_prompt="Answer using web search when needed. Cite the source URLs.",
        )
        return await agent.arun(question)


async def main():
    parser = argparse.ArgumentParser(description="Research with Parallel Search MCP.")
    parser.add_argument("question", nargs="?", default="What is the Model Context Protocol? Cite sources.")
    parser.add_argument("--provider", default="openai", help="Configured LLM provider (default: openai).")
    args = parser.parse_args()
    result = await search(args.question, LLMClient(args.provider))
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
