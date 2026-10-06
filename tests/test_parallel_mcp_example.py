"""Exercise the runnable example through the real MCP HTTP loader and agent loop."""
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from agentmaker import __version__
from agentmaker.core.llm_response import LLMResponse
from agentmaker.testing import ScriptedLLM

pytest.importorskip("mcp")
import httpx

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/17_parallel_search_mcp.py"
spec = importlib.util.spec_from_file_location("parallel_example", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class ToolResultLLM(ScriptedLLM):
    """Request a tool, then return the actual tool message without synthesis."""

    async def chat(self, messages, *, tools=None, **kwargs):
        if self.calls == 0:
            self.schema = tools
            return await super().chat(messages, tools=tools, **kwargs)
        return LLMResponse(content=messages[-1]["content"], model=self.model)


@pytest.fixture
def remote(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        rpc = json.loads(request.content)
        if "id" not in rpc:
            return httpx.Response(202)
        method = rpc["method"]
        if method == "initialize":
            result = {"protocolVersion": rpc["params"]["protocolVersion"],
                      "capabilities": {"tools": {}}, "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [
                {"name": "web_search", "description": "Search public web pages.",
                 "inputSchema": {"type": "object", "properties": {"objective": {"type": "string"},
                                 "search_queries": {"type": "array", "items": {"type": "string"}}},
                                 "required": ["objective", "search_queries"]}},
                {"name": "web_fetch", "description": "Read a public web page.",
                 "inputSchema": {"type": "object", "properties": {"urls": {"type": "array", "items": {"type": "string"}}},
                                 "required": ["urls"]}},
                {"name": "unrelated", "description": "Other tool", "inputSchema": {"type": "object"}},
            ]}
        else:
            assert method == "tools/call"
            result = {"content": [{"type": "text", "text": "MCP connects AI applications to external tools. https://modelcontextprotocol.io/"}],
                      "isError": False}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": rpc["id"], "result": result})

    original = httpx.AsyncClient.__init__

    def init(self, *args, **kwargs):
        assert kwargs.get("timeout").connect == 60.0
        kwargs["transport"] = httpx.MockTransport(respond)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    return requests


@pytest.mark.parametrize("name,arguments", [
    ("parallel_web_search", {"objective": "Explain MCP", "search_queries": ["Model Context Protocol"]}),
    ("parallel_web_fetch", {"urls": ["https://modelcontextprotocol.io/"]}),
])
def test_example_agent_executes_remote_tools(remote, name, arguments):
    llm = ToolResultLLM([ScriptedLLM.tool_call(name, arguments)])
    confirmations = []

    def confirm(tool, params):
        confirmations.append((tool, params))
        return True

    result = asyncio.run(example.search("Explain MCP", llm, confirm=confirm))
    assert "MCP connects AI applications" in result.final_output
    assert "https://modelcontextprotocol.io/" in result.final_output
    assert {s["function"]["name"] for s in llm.schema} == {"parallel_web_search", "parallel_web_fetch"}
    tool, params = confirmations[0]
    assert tool.requires_confirmation and tool.external_content
    assert tool.origin == "mcp:parallel"
    assert params == arguments
    calls = [json.loads(r.content) for r in remote if r.method == "POST"]
    assert [c["method"] for c in calls] == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    assert calls[-1]["params"] == {"name": name.removeprefix("parallel_"), "arguments": arguments}
    for request in remote:
        assert str(request.url) == "https://search.parallel.ai/mcp"
        assert request.headers["user-agent"] == f"agentmaker/{__version__}"
        assert "authorization" not in request.headers


def test_example_declined_confirmation_prevents_remote_call(remote):
    llm = ToolResultLLM([ScriptedLLM.tool_call("parallel_web_search", {
        "objective": "Explain MCP", "search_queries": ["MCP"],
    })])
    asyncio.run(example.search("Explain MCP", llm, confirm=lambda tool, params: False))
    assert not any(json.loads(r.content).get("method") == "tools/call" for r in remote if r.method == "POST")


def test_example_cli_runs_with_terminal_confirmation(remote, monkeypatch, capsys):
    llm = ToolResultLLM([ScriptedLLM.tool_call("parallel_web_search", {
        "objective": "Explain MCP", "search_queries": ["MCP"],
    })])
    monkeypatch.setattr(example, "LLMClient", lambda provider: llm)
    monkeypatch.setattr("sys.argv", [str(EXAMPLE), "Explain MCP"])
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    asyncio.run(example.main())
    assert "MCP connects AI applications" in capsys.readouterr().out
