"""Tools for analyzing Unusual Whales dark pool data."""

from mcp.types import Tool


def register_dark_pool_tools() -> list[Tool]:
    """Return tool definitions for dark pool analysis.

    These tools will be implemented as you identify the specific
    columns and format of your Unusual Whales dark pool exports.
    """
    return [
        Tool(
            name="analyze_dark_pool",
            description=(
                "Analyze dark pool activity from Unusual Whales exports. "
                "Identifies large block trades and institutional activity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Dark pool data CSV filename",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Filter by stock ticker symbol (optional)",
                    },
                    "min_size": {
                        "type": "integer",
                        "description": "Minimum trade size filter (optional)",
                    },
                },
                "required": ["filename"],
            },
        ),
    ]
