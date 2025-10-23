"""
Diagnostic script to test InsurancePolicyAgent registration
"""
import os
import sys
from dotenv import load_dotenv
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the agents module
from msme_chatbot_agents import mcp_agents
import inspect

print("=" * 60)
print("AGENT REGISTRATION DIAGNOSTIC")
print("=" * 60)

# Step 1: Check what agents are discovered
print("\n1. Discovering agents using inspect.getmembers:")
agent_list = []
for name, obj in inspect.getmembers(mcp_agents):
    if inspect.isclass(obj) and name.endswith('Agent'):
        agent_list.append(name)
        print(f"   ✓ Found: {name}")

print(f"\n   Total agents found: {len(agent_list)}")
print(f"   Agent list: {agent_list}")

# Step 2: Check if InsurancePolicyAgent specifically exists
print("\n2. Checking InsurancePolicyAgent:")
if hasattr(mcp_agents, 'InsurancePolicyAgent'):
    print("   ✓ InsurancePolicyAgent class EXISTS in mcp_agents module")
    agent_class = getattr(mcp_agents, 'InsurancePolicyAgent')
    print(f"   Class: {agent_class}")

    # Try to instantiate it
    try:
        agent_instance = agent_class(llm=None, support_doc=None)
        print(f"   ✓ Successfully instantiated")
        print(f"   - Agent name: {agent_instance.name}")
        print(f"   - Handoff description: {agent_instance.hand_off_description}")
        print(f"   - Tools: {[tool.__name__ if hasattr(tool, '__name__') else str(tool) for tool in agent_instance.tools]}")
    except Exception as e:
        print(f"   ✗ Failed to instantiate: {e}")
else:
    print("   ✗ InsurancePolicyAgent class NOT FOUND in mcp_agents module")

# Step 3: Check AgentEnum
print("\n3. Checking AgentEnum:")
if hasattr(mcp_agents, 'AgentEnum'):
    agent_enum = mcp_agents.AgentEnum
    print(f"   ✓ AgentEnum exists")
    print(f"   Available enum values:")
    for member in agent_enum:
        print(f"      - {member.name}: {member.value}")
        if 'Insurance' in member.value or 'Insurance' in member.name:
            print(f"        ^ This might relate to InsurancePolicyAgent")
else:
    print("   ✗ AgentEnum NOT FOUND")

# Step 4: Simulate workflow builder process
print("\n4. Simulating workflow builder process:")
print(f"   Agent list to be processed: {agent_list}")
print("\n   Checking if each agent would be found by build_workflow:")
module = mcp_agents
for agent_name in agent_list:
    if hasattr(module, agent_name):
        print(f"   ✓ {agent_name}: Would be found by hasattr(module, agent_name)")
    else:
        print(f"   ✗ {agent_name}: Would NOT be found by hasattr(module, agent_name)")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
