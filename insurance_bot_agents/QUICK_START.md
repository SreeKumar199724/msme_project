# Quick Start Guide

## Installation (One Command)

```bash
cd C:\Users\user\PROJECT-002-INSURANCE-BOT\insurance_bot\backend
uv pip install -e ../insurance_bot_agents/
```

## Import and Use

```python
from insurance_bot_agents import (
    ProcessPlannerAgent,
    InsureSQLCoderAgent,
    InsurancePolicyAgent,
    EmailDraftAgent,
    VisualizationAgent,
    AgentEnum
)
```

## Quick Test

```bash
cd insurance_bot_agents
python test_package_import.py
```

## Usage Example

```python
from langchain_openai import ChatOpenAI
from insurance_bot_agents import InsureSQLCoderAgent

llm = ChatOpenAI(model="gpt-4")

agent = InsureSQLCoderAgent(
    llm=llm,
    support_doc={
        "table_name": "open_invoices_data",
        "table_structure": "sno, customer, invoice_number..."
    }
)

# Use agent in your LangGraph workflow
```

## Update Package After Changes

Since it's installed in editable mode (`-e`), any changes you make to the code are immediately reflected - no need to reinstall!

## Reinstall If Needed

```bash
cd C:\Users\user\PROJECT-002-INSURANCE-BOT\insurance_bot\backend
uv pip uninstall insurance-bot-agents
uv pip install -e ../insurance_bot_agents/
```

That's it! You're ready to use the package anywhere in your project.
