# Insurance Bot - AI-Powered Insurance Assistant

An intelligent chatbot system built with LangGraph and FastAPI that provides insurance-related assistance using a multi-agent swarm architecture with MCP (Model Context Protocol) integration.

## Project Overview

This project implements an advanced insurance bot that leverages multiple AI agents working together to handle various insurance-related queries and tasks. The system uses LangGraph's swarm architecture to coordinate between specialized agents, providing comprehensive and context-aware responses.

## Project Structure

```
insurance_bot/
├── backend/                          # Backend application
│   ├── main.py                       # FastAPI application with LangGraph workflow
│   ├── app.py                        # Alternative application entry point
│   ├── requirements.txt              # Python dependencies
│   ├── pyproject.toml                # Project configuration
│   ├── uv.lock                       # UV package manager lock file
│   ├── conda_packages.txt            # Conda package list
│   ├── Support_docs_json_format.txt  # Supporting documentation format
│   ├── mermaid_graph.png            # Workflow visualization
│   └── templates/                    # HTML templates
│
├── langgraph-chatbot-frontend/      # Frontend application
│   ├── index.html                    # Main HTML interface
│   ├── package.json                  # Node.js dependencies
│   ├── package-lock.json             # NPM lock file
│   ├── execution_doc.txt             # Execution documentation
│   └── node_modules/                 # Node dependencies
│
├── insurance_bot_agents/            # Agent modules
│   ├── mcp_agents.py                # MCP-enabled agent implementations
│   ├── mcp_agent_tools.py           # MCP agent tools and utilities
│   ├── agents.py                    # Base agent implementations
│   ├── agent_tools.py               # Agent tools and utilities
│   ├── requirements.txt             # Agent-specific dependencies
│   ├── remote_mcp_server_execution.txt  # MCP server documentation
│   └── __init__.py                  # Package initialization
│
├── .venv/                           # Python virtual environment
├── .gitignore                       # Git ignore rules
├── .python-version                  # Python version specification
└── readme.md                        # This file
```

## Key Features

- **Multi-Agent Architecture**: Uses LangGraph swarm to coordinate multiple specialized agents
- **MCP Integration**: Implements Model Context Protocol for enhanced agent capabilities
- **Persistent Memory**: MongoDB-based checkpointing for stateful conversations
- **Azure Integration**:
  - Azure OpenAI (GPT-4o) for language model
  - Azure Cognitive Services Speech for text-to-speech and speech-to-text
- **Real-time Chat**: FastAPI backend with async support
- **Interactive Frontend**: Web-based chat interface with Chart.js visualization support
- **Structured Outputs**: JSON-formatted responses with optional chart data

## Technology Stack

### Backend
- **Framework**: FastAPI 0.115.11
- **LLM**: Azure OpenAI (GPT-4o) via langchain-openai
- **Orchestration**: LangGraph 0.5.2 with Swarm architecture
- **Database**: MongoDB (AsyncMongoDBSaver for checkpointing)
- **Speech**: Azure Cognitive Services Speech 1.43.0
- **MCP**: langchain-mcp-adapters 0.1.7, mcp 1.10.1

### Frontend
- **HTML/JavaScript**: Vanilla JS with Chart.js
- **HTTP Server**: Node.js-based server
- **Real-time Communication**: Fetch API for backend communication

### Agent Framework
- LangChain 0.3.26
- LangGraph Swarm 0.0.12
- LangGraph Checkpoint MongoDB 0.1.4
- Custom MCP-enabled agents

## Installation

### Prerequisites
- Python 3.11+
- Node.js 16+
- MongoDB instance (local or cloud)
- Azure OpenAI API credentials
- Azure Speech Services credentials

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
```bash
python -m venv ../.venv
source ../.venv/bin/activate  # On Windows: ..\.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables (create `.env` file in root):
```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key

# Azure Speech
SPEECH_KEY=your_speech_key
SPEECH_REGION=your_region

# MongoDB
MONGODB_URI=your_mongodb_uri
ASYNC_MONGODB_URI=your_async_mongodb_uri
SUPPORT_DOC_DB_NAME=your_db_name
SUPPORT_DOC_COLLECTION=your_collection_name
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd langgraph-chatbot-frontend
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### Start Backend Server
```bash
cd backend
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8501
```

The backend will be available at `http://localhost:8501`

### Start Frontend Server
```bash
cd langgraph-chatbot-frontend
# Use your preferred HTTP server, e.g.:
npx http-server -p 3000
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### POST /chat
Send a message to the chatbot.

**Request Body:**
```json
{
  "user_input": "Your question here",
  "thread_id": "unique-thread-id"
}
```

**Response:**
```json
{
  "assistant_response": "Text response from assistant",
  "graph_data": {
    // Optional Chart.js compatible data
  }
}
```

## Agent Architecture

The system uses a multi-agent swarm architecture with the following components:

1. **Process Planner Agent**: Coordinates task execution and plans responses
2. **Specialized Agents**: Domain-specific agents loaded dynamically from `insurance_bot_agents`
3. **MCP Integration**: Agents can utilize MCP servers for enhanced capabilities
4. **Hand-off Mechanism**: Agents can transfer control to specialized agents as needed

### Agent Features
- MongoDB-backed support documentation
- Dynamic agent loading and instantiation
- Structured output support
- Context-aware handoffs between agents

## Features in Detail

### Speech Integration
- **Text-to-Speech**: Convert assistant responses to audio (Azure Neural Voice)
- **Speech-to-Text**: Voice input support via microphone

### Visualization Support
- Chart.js integration for data visualization
- Dynamic chart generation based on assistant responses
- Download chart functionality

### Persistent Conversations
- Thread-based conversation tracking
- MongoDB checkpointing for state persistence
- Async MongoDB support for scalability

## Development

### Adding New Agents

1. Create agent class in `insurance_bot_agents/mcp_agents.py`
2. Implement required methods:
   - `__init__(self, llm, support_doc=None)`
   - `create_agent(self, hand_off_tools)`
   - Define `name` and `hand_off_description` attributes
3. Agents are automatically discovered and loaded

### Project Dependencies

Key dependencies:
- `langgraph-swarm`: Agent swarm orchestration
- `langchain-openai`: Azure OpenAI integration
- `fastapi`: Web framework
- `pymongo` & `motor`: MongoDB drivers
- `azure-cognitiveservices-speech`: Speech services

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `insurance_bot_agents` is in Python path
2. **MongoDB Connection**: Verify `MONGODB_URI` and network access
3. **Azure Credentials**: Check environment variables are set correctly
4. **Port Conflicts**: Ensure ports 8501 (backend) and 3000 (frontend) are available

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]
