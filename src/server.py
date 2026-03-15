"""Unusual Whales MCP Server — analyze downloaded Unusual Whales data exports."""

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.data_loader import list_data_files, load_file
from src.tools import (
    register_dark_pool_tools,
    register_market_data_tools,
    register_options_flow_tools,
)

server = Server("unusual-whales-mcp")

# Registry for tool handlers
_tool_handlers: dict[str, callable] = {}


def _register_core_tools():
    """Register built-in tools for managing data files."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        tools = [
            Tool(
                name="list_data_files",
                description=(
                    "List all data files (CSV/JSON) available in the data directory. "
                    "Drop Unusual Whales exports into the data/ folder to make them available."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="query_data",
                description=(
                    "Load and query a data file. Returns the first N rows by default. "
                    "Supports filtering by column values."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the data file (e.g. 'options_flow.csv')",
                        },
                        "head": {
                            "type": "integer",
                            "description": "Number of rows to return (default: 20)",
                            "default": 20,
                        },
                        "sort_by": {
                            "type": "string",
                            "description": "Column name to sort by",
                        },
                        "ascending": {
                            "type": "boolean",
                            "description": "Sort ascending (default: false)",
                            "default": False,
                        },
                        "filter_column": {
                            "type": "string",
                            "description": "Column to filter on",
                        },
                        "filter_value": {
                            "type": "string",
                            "description": "Value to filter for (exact match)",
                        },
                    },
                    "required": ["filename"],
                },
            ),
            Tool(
                name="describe_data",
                description=(
                    "Get summary statistics and column info for a data file. "
                    "Shows dtypes, row count, and basic stats."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the data file",
                        },
                    },
                    "required": ["filename"],
                },
            ),
        ]
        # Add tools from registered modules
        for register_fn in [
            register_options_flow_tools,
            register_dark_pool_tools,
            register_market_data_tools,
        ]:
            tools.extend(register_fn())
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "list_data_files":
            files = list_data_files()
            if not files:
                return [TextContent(
                    type="text",
                    text="No data files found. Drop CSV/JSON exports from Unusual Whales into the data/ directory.",
                )]
            return [TextContent(type="text", text=json.dumps(files, indent=2))]

        elif name == "query_data":
            df = load_file(arguments["filename"])
            if "filter_column" in arguments and "filter_value" in arguments:
                col = arguments["filter_column"]
                val = arguments["filter_value"]
                df = df[df[col].astype(str) == str(val)]
            if "sort_by" in arguments:
                df = df.sort_values(
                    by=arguments["sort_by"],
                    ascending=arguments.get("ascending", False),
                )
            head = arguments.get("head", 20)
            result = df.head(head).to_json(orient="records", indent=2)
            return [TextContent(type="text", text=result)]

        elif name == "describe_data":
            df = load_file(arguments["filename"])
            info = {
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "summary": json.loads(df.describe(include="all").to_json()),
            }
            return [TextContent(type="text", text=json.dumps(info, indent=2))]

        # Delegate to registered tool handlers
        elif name in _tool_handlers:
            return await _tool_handlers[name](arguments)

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


_register_core_tools()


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
