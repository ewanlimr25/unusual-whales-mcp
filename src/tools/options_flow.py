"""Tools for analyzing Unusual Whales options flow data."""

from mcp.types import Tool


def register_options_flow_tools() -> list[Tool]:
    """Return tool definitions for options flow analysis.

    These tools will be implemented as you identify the specific
    columns and format of your Unusual Whales options flow exports.
    """
    return [
        Tool(
            name="analyze_options_flow",
            description=(
                "Analyze options flow data from Unusual Whales exports. "
                "Finds unusual activity by volume, premium, or sentiment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Options flow CSV filename",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Filter by stock ticker symbol (optional)",
                    },
                    "min_premium": {
                        "type": "number",
                        "description": "Minimum premium filter in dollars (optional)",
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["bullish", "bearish", "neutral"],
                        "description": "Filter by sentiment (optional)",
                    },
                },
                "required": ["filename"],
            },
        ),
    ]
