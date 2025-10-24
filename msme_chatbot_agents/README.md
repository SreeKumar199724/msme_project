# MSME Chatbot Agents

A Python package containing specialized AI agents for MSME (Micro, Small, and Medium Enterprises) chatbot tasks using LangChain, LangGraph, and MCP (Model Context Protocol).

## Features

- **TranslatorAgent**: Bilingual translation agent for English and Telugu language support
- **MSMEGuidelinesAgent**: Semantic search agent for MSME guidelines, policies, and document retrieval
- **MCP Integration**: Full Model Context Protocol support for advanced tool usage
- **Swarm Architecture**: LangGraph Swarm-based multi-agent coordination

## Installation

### Install in Editable Mode (Recommended)

From the parent project directory:

```bash
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa
uv pip install -e msme_chatbot_agents
```

This installs the package in editable mode, so any changes to the source code are immediately available without reinstallation.

### Verify Installation

```bash
.venv\Scripts\python.exe -c "import msme_chatbot_agents; print('Success!')"
```

## Usage

After installation, you can import and use the agents in your Python code:

```python
from msme_chatbot_agents import (
    TranslatorAgent,
    MSMEGuidelinesAgent,
    AgentEnum,
    ProcessSubTask,
    ProcessFullPlan,
    ChartData
)

# Import tools
from msme_chatbot_agents import (
    semantic_search_tool,
    mcp
)

# Example: Initialize an agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

translator = TranslatorAgent(llm=llm)
guidelines_agent = MSMEGuidelinesAgent(llm=llm)

# Use the agent in your workflow
agent = translator.create_agent(hand_off_tools=[])
```

## Package Structure

```
msme_chatbot_agents/
├── __init__.py              # Package exports and version info
├── mcp_agents.py            # Main agent class definitions
├── mcp_agent_tools.py       # MCP tool implementations
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

## Requirements

- Python >= 3.12
- LangChain, LangGraph, FastMCP, and related dependencies
- See `pyproject.toml` for the complete dependency list

## Agents

### TranslatorAgent
Professional bilingual agent for translating between English and Telugu.

**Key Features:**
- Maintains user's preferred language throughout the session
- Provides consistent translations
- Context-aware language detection

### MSMEGuidelinesAgent
Semantic search agent for MSME policy and guideline retrieval.

**Key Features:**
- Vector database search using Qdrant
- OpenAI embeddings for semantic matching
- Policy-based question answering
- Document citation and references

## MCP Tools

The package includes MCP-enabled tools:

- `semantic_search_tool`: Searches MSME guidelines in vector database
- `SQL_query_exec`: Executes SQL queries on PostgreSQL (commented out by default)
- `send_email`: Sends emails via SMTP (commented out by default)

### Running as MCP Server

You can also run this package as a standalone MCP server:

```bash
.venv\Scripts\python.exe msme_chatbot_agents\mcp_agents.py
```

This starts an MCP server that communicates via stdin/stdout, useful for MCP client integrations.

## Environment Variables

Required environment variables (create a `.env` file):

```env
# LLM APIs
OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=your_endpoint

# Vector Database
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
EMBEDDING_MODEL_NAME=text-embedding-3-large
VECTOR_STORE_COLLECTION=msme_guidelines_docs

# Optional: Database (if using SQL tools)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_db
DB_USER=your_user
DB_PASSWORD=your_password

# Optional: Email (if using email tools)
SENDER_EMAIL=your_email
SENDER_PASSWORD=your_password
```

## Development

### Making Changes

Since the package is installed in editable mode, you can directly edit:
- `mcp_agents.py` - to modify agent behavior
- `mcp_agent_tools.py` - to add/modify tools
- `pyproject.toml` - to update dependencies

Changes take effect immediately without reinstallation.

### Reinstalling After Dependency Changes

If you modify `pyproject.toml` dependencies:

```bash
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa
uv pip install -e msme_chatbot_agents --force-reinstall
```

## Troubleshooting

### Module Not Found Error
```bash
# Reinstall the package
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa
uv pip install -e msme_chatbot_agents
```

### LangGraph MRO Error
Make sure you're using LangGraph 0.5.4 or higher:
```bash
uv sync  # Syncs all packages to correct versions
```

## Integration Example

See `backend/main.py` for a complete example of how this package is used in a FastAPI application with LangGraph Swarm coordination.

## Version

Current version: 0.1.0

## License

[Your License Here]

## Contributing

Contributions are welcome! Please ensure all changes are compatible with the existing architecture.
