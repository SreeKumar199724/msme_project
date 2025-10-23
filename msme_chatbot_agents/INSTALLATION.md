# Installation Guide for insurance-bot-agents

## Quick Start

### Option 1: Install in Editable Mode (Recommended for Development)

This allows you to make changes to the code and have them immediately reflected without reinstalling:

```bash
# Navigate to the package directory
cd C:\Users\user\PROJECT-002-INSURANCE-BOT\insurance_bot\insurance_bot_agents

# Install in editable mode
pip install -e .
```

Or from the parent directory:

```bash
cd C:\Users\user\PROJECT-002-INSURANCE-BOT\insurance_bot
pip install -e insurance_bot_agents/
```

### Option 2: Standard Installation

For production use or when you don't need to modify the code:

```bash
cd C:\Users\user\PROJECT-002-INSURANCE-BOT\insurance_bot
pip install insurance_bot_agents/
```

### Option 3: Install with Development Dependencies

Includes additional tools for testing and development:

```bash
pip install -e "insurance_bot_agents[dev]"
```

## Verifying Installation

After installation, verify that the package is installed correctly:

```bash
# Check if package is installed
pip show insurance-bot-agents

# Or list all installed packages
pip list | grep insurance-bot-agents
```

## Testing the Installation

Create a test Python file to verify the imports work:

```python
# test_import.py
from insurance_bot_agents import (
    ProcessPlannerAgent,
    InsureSQLCoderAgent,
    InsurancePolicyAgent,
    EmailDraftAgent,
    VisualizationAgent,
    AgentEnum
)

print("✓ All agents imported successfully!")
print(f"Available agents: {list(AgentEnum)}")

# Test importing tools
from insurance_bot_agents import (
    SQL_query_exec,
    send_email,
    semantic_search_tool
)

print("✓ All tools imported successfully!")
```

Run the test:

```bash
python test_import.py
```

## Using in Other Projects

Once installed, you can use the package in any Python script or project within the same environment:

```python
# In your project file (e.g., backend/main.py)
from insurance_bot_agents import ProcessPlannerAgent, InsureSQLCoderAgent
from langchain_openai import ChatOpenAI

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create agent instances
planner = ProcessPlannerAgent(llm=llm)
sql_agent = InsureSQLCoderAgent(llm=llm, support_doc={
    "table_name": "insurance_data",
    "table_structure": "..."
})

# Use in your workflow
# ...
```

## Uninstalling

To remove the package:

```bash
pip uninstall insurance-bot-agents
```

## Updating the Package

If you've made changes and need to update:

### For Editable Installation
Changes are automatically reflected - no action needed!

### For Standard Installation
Reinstall the package:

```bash
pip install --upgrade insurance_bot_agents/
```

Or force reinstall:

```bash
pip install --force-reinstall insurance_bot_agents/
```

## Troubleshooting

### Import Errors

If you encounter import errors:

1. Check that the package is installed:
   ```bash
   pip list | grep insurance-bot-agents
   ```

2. Verify you're using the correct Python environment:
   ```bash
   python --version
   which python  # On Windows: where python
   ```

3. Try reinstalling:
   ```bash
   pip uninstall insurance-bot-agents
   pip install -e insurance_bot_agents/
   ```

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# View dependency tree
pip show insurance-bot-agents

# Check for conflicts
pip check
```

### Path Issues on Windows

If you encounter path issues on Windows, use absolute paths:

```bash
pip install -e "C:\Users\user\PROJECT-002-INSURANCE-BOT\insurance_bot\insurance_bot_agents"
```

## Environment Setup

Make sure you have all required environment variables:

```bash
# Create or update .env file
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
DATABASE_URL=your_database_url
# Add other required variables
```

## Next Steps

After successful installation:

1. Set up your environment variables
2. Configure your database connections
3. Test individual agents
4. Integrate into your main application

For more details, see the main [README.md](README.md).
