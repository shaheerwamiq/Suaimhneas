"""
Host, Client, Protocol & Server --> Tool Providers

Minimal MCP server example.

Demonstrates the CORE structure of ANY MCP server -- the same pattern
you'd use for real Jira access -- using a mock in-memory "Jira" so it
runs with zero external accounts, tokens, or setup.

    AI Assistant  <--- MCP (JSON-RPC over stdio) --->  MCP Server  <--> Jira API (real version)

The assistant never needs to know how Jira's REST API works. It just
sees three tools below (list_issues, get_issue, create_issue) with a
name, a docstring (used as the description), and typed parameters. It
decides when to call them; this server does the actual work.

Requirements:
    pip install mcp

Sanity check it starts (it'll just idle waiting for a client on stdin):
    python mcp_jira_mock_server.py
    (Ctrl+C to stop)

To actually poke at it interactively, use the official MCP Inspector
(no Anthropic account needed, just Node.js):
    npx @modelcontextprotocol/inspector python mcp_jira_mock_server.py
This opens a browser UI where you can call list_issues/get_issue/create_issue
by hand and see the exact request/response JSON going over the protocol.

To use it from Claude Desktop instead, add it to your MCP config
(claude_desktop_config.json):
    {
      "mcpServers": {
        "jira-mock": {
          "command": "python",
          "args": ["/absolute/path/to/mcp_jira_mock_server.py"]
        }
      }
    }
"""

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Mock "Jira" data store.
# In a REAL Jira MCP server, these three functions would call the Jira REST
# API (e.g. via the `jira` python package or `requests`) using an API token
# read from an environment variable -- nothing else about this file changes.
# ---------------------------------------------------------------------------
FAKE_JIRA_DB = {
    "PROJ-1": {"summary": "Login page throws 500 on submit", "status": "To Do", "assignee": "unassigned"},
    "PROJ-2": {"summary": "Add dark mode toggle", "status": "In Progress", "assignee": "shaheer"},
    "PROJ-3": {"summary": "Export button is slow on large reports", "status": "Done", "assignee": "shaheer"},
}
_next_id = 4

# The string here ("jira-mock") is the server's name as clients will see it.
mcp = FastMCP("jira-mock")


@mcp.tool()
def list_issues(status: str = "") -> str:
    """List Jira issues. Optionally filter by status: 'To Do', 'In Progress', or 'Done'."""
    results = []
    for key, issue in FAKE_JIRA_DB.items():
        if status and issue["status"].lower() != status.lower():
            continue
        results.append(f"{key}: {issue['summary']} [{issue['status']}] (assignee: {issue['assignee']})")
    return "\n".join(results) if results else "No matching issues."


@mcp.tool()
def get_issue(issue_key: str) -> str:
    """Get full details for a single Jira issue by its key, e.g. 'PROJ-2'."""
    issue = FAKE_JIRA_DB.get(issue_key)
    if not issue:
        return f"No issue found with key {issue_key}."
    return (
        f"Key: {issue_key}\n"
        f"Summary: {issue['summary']}\n"
        f"Status: {issue['status']}\n"
        f"Assignee: {issue['assignee']}"
    )


@mcp.tool()
def create_issue(summary: str, assignee: str = "unassigned") -> str:
    """Create a new Jira issue with the given summary and optional assignee."""
    global _next_id
    key = f"PROJ-{_next_id}"
    _next_id += 1
    FAKE_JIRA_DB[key] = {"summary": summary, "status": "To Do", "assignee": assignee}
    return f"Created {key}: {summary} (assignee: {assignee})"


if __name__ == "__main__":
    # stdio transport: the client launches this script as a subprocess and
    # talks to it over stdin/stdout. This is the standard local-dev setup.
    mcp.run(transport="stdio")