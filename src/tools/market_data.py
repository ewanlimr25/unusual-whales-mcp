"""Tools for analyzing general market data from Unusual Whales."""

from mcp.types import Tool


def register_market_data_tools() -> list[Tool]:
    """Return tool definitions for market data analysis.

    These tools will be implemented as you identify the specific
    columns and format of your Unusual Whales market data exports.
    """
    return [
        Tool(
            name="analyze_market_overview",
            description=(
                "Analyze market overview data from Unusual Whales exports. "
                "Provides sector performance, market breadth, and trend analysis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Market data CSV filename",
                    },
                },
                "required": ["filename"],
            },
        ),
    ]
