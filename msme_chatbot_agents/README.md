# Insurance Bot Agents

A Python package containing specialized AI agents for insurance domain tasks using LangChain, LangGraph, and MCP (Model Context Protocol).

## Features

- **ProcessPlannerAgent**: Plans and orchestrates complex insurance workflows
- **InsureSQLCoderAgent**: Generates and executes SQL queries for insurance data
- **InsurancePolicyAgent**: Handles insurance policy information and semantic search
- **EmailDraftAgent**: Creates professional email drafts for insurance communications
- **VisualizationAgent**: Generates charts and visualizations for insurance data

## Installation

### Install from local directory (editable mode)

For development, install in editable mode so changes are immediately reflected:

```bash
# From the insurance_bot_agents directory
pip install -e .

# Or from the parent directory
pip install -e insurance_bot_agents/
```

### Install from local directory (standard mode)

```bash
# From the parent directory
pip install insurance_bot_agents/

# Or directly from the directory
cd insurance_bot_agents
pip install .
```

### Install with development dependencies

```bash
pip install -e ".[dev]"
```

## Usage

After installation, you can import and use the agents in your Python code:

```python
from insurance_bot_agents import (
    ProcessPlannerAgent,
    InsureSQLCoderAgent,
    InsurancePolicyAgent,
    EmailDraftAgent,
    VisualizationAgent,
    AgentEnum
)

# Import tools
from insurance_bot_agents import (
    SQL_query_exec,
    send_email,
    semantic_search_tool,
    mcp
)

# Example: Initialize an agent
planner = ProcessPlannerAgent()

# Use the agent in your workflow
# ... your code here ...
```

## Package Structure

```
insurance_bot_agents/
├── __init__.py              # Package exports and version info
├── mcp_agents.py            # Main agent definitions
├── mcp_agent_tools.py       # Tool implementations
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

## Requirements

- Python >= 3.12
- See `pyproject.toml` for full dependency list

## Development

### Setting up for development

```bash
# Clone the repository
git clone <your-repo-url>
cd insurance_bot_agents

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Running tests

```bash
pytest tests/
```

## Environment Variables

Make sure to set up the following environment variables:

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI models)
- Database connection strings and other service credentials as needed

## License

[Your License Here]

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
