#!/usr/bin/env python3

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path


# Change this to any directory you want the MCP server to access.
DIRECTORY = str(Path(__file__).parent.resolve())


async def main():
    # Launch the MCP server
    server = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            DIRECTORY,
        ],
    )

    print("Starting MCP server...\n")

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:

            # Initialize the connection
            await session.initialize()

            print("=" * 80)
            print("CONNECTED")
            print("=" * 80)

            # ------------------------------------------------------------------
            # Discover all available tools
            # ------------------------------------------------------------------

            response = await session.list_tools()

            print(f"\nFound {len(response.tools)} tools:\n")

            for tool in response.tools:
                print("-" * 80)
                print(f"Tool Name : {tool.name}")
                print(f"Description : {tool.description}")

                print("\nInput Schema:")
                print(tool.inputSchema)

            # ------------------------------------------------------------------
            # Example: list files
            # ------------------------------------------------------------------

            print("\n")
            print("=" * 80)
            print("Calling list_directory")
            print("=" * 80)

            try:
                result = await session.call_tool(
                    "list_directory",
                    {
                        "path": "."
                    }
                )

                print(result)

            except Exception as e:
                print("Tool invocation failed:")
                print(e)


if __name__ == "__main__":
    asyncio.run(main())