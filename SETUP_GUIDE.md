# MSME Chatbot Setup Guide

## Project Structure

```
msme_chatbot_qa/
├── .venv/                          # Main virtual environment (USE THIS)
├── backend/
│   └── main.py                     # FastAPI backend
├── msme_chatbot_agents/            # Agent package (installed as editable)
│   ├── __init__.py
│   ├── mcp_agents.py              # Agent classes
│   ├── mcp_agent_tools.py         # MCP tools
│   └── pyproject.toml             # Package configuration
└── pyproject.toml                  # Main project configuration
```

## Installation Steps

### Step 1: Navigate to Main Project Directory
```bash
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa
```

### Step 2: Install the Agents Package (Editable Mode)
```bash
uv pip install -e msme_chatbot_agents
```

**What this does:**
- Installs `msme_chatbot_agents` as a Python package
- Uses `-e` (editable mode) so changes to the code are immediately available
- Installs into the main `.venv` virtual environment

### Step 3: Verify Installation
```bash
.venv/Scripts/python.exe -c "import msme_chatbot_agents; print('Success!')"
```

## Usage in Your Code

### In backend/main.py (or any other file):

```python
# Import the package
import msme_chatbot_agents
from msme_chatbot_agents.mcp_agents import TranslatorAgent, MSMEGuidelinesAgent
from msme_chatbot_agents.mcp_agent_tools import semantic_search_tool, mcp

# Use the agents
llm = AzureChatOpenAI(...)
translator = TranslatorAgent(llm)
guidelines_agent = MSMEGuidelinesAgent(llm)
```

## Understanding the Components

### 1. **mcp_agents.py**
- Contains agent class definitions (TranslatorAgent, MSMEGuidelinesAgent)
- Each agent has a `create_agent()` method that returns a LangGraph agent
- Can be run as an MCP server: `python mcp_agents.py`

### 2. **mcp_agent_tools.py**
- Contains tool definitions decorated with `@mcp.tool()`
- Tools: `SQL_query_exec`, `send_email`, `semantic_search_tool`
- Creates an MCP (Model Context Protocol) server object

### 3. **How backend/main.py Uses Them**
- Imports agent classes from `msme_chatbot_agents.mcp_agents`
- Dynamically instantiates agents based on configuration
- Creates a LangGraph swarm with handoff tools
- Serves via FastAPI endpoints

## Running the Backend

```bash
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa
.venv/Scripts/python.exe backend/main.py
```

Or with uvicorn:
```bash
.venv/Scripts/uvicorn.exe backend.main:app --host 0.0.0.0 --port 8503 --reload
```

## Testing MCP Server Standalone

If you want to run `mcp_agents.py` as a standalone MCP server:

```bash
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa
.venv/Scripts/python.exe msme_chatbot_agents/mcp_agents.py
```

This starts an MCP server that communicates via stdin/stdout (used by MCP clients).

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError: No module named 'msme_chatbot_agents'`:
```bash
# Reinstall the package
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa
uv pip install -e msme_chatbot_agents
```

### Version Conflicts
If you see MRO errors with langgraph:
```bash
# Sync all packages to correct versions
uv sync
```

### Old Virtual Environment
If there's a nested `.venv` in `msme_chatbot_agents/`:
```bash
# Remove it (we only use the main .venv)
powershell -Command "Remove-Item -Recurse -Force 'msme_chatbot_agents\.venv'"
```

## Key Points

1. **One Virtual Environment**: Use only `C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa\.venv`
2. **Editable Install**: The `-e` flag means changes to code are immediately available
3. **Package Structure**: `msme_chatbot_agents` is a proper Python package with `__init__.py` and `pyproject.toml`
4. **MCP vs Regular Import**:
   - As a package: Import and use in Python code
   - As MCP server: Run standalone for MCP protocol communication

## Current Status

✓ Package installed successfully in main venv
✓ All imports working correctly
✓ Backend can import and use agents
✓ Version compatibility resolved (langgraph 0.5.4)
