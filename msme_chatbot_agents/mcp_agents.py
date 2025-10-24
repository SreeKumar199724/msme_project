import os
from dotenv import load_dotenv
from typing import Annotated, List, Optional, Union
from typing_extensions import TypedDict,Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from contextlib import asynccontextmanager


from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent, InjectedState,ToolNode, tools_condition
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_swarm import create_handoff_tool, create_swarm
import inspect
import asyncio
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

# from msme_chatbot_agents.agent_tools import * #without MCP
# from msme_chatbot_agents.mcp_agent_tools import generate_chart_response
# from .mcp_agent_tools import SQL_query_exec, send_email,semantic_search_tool, mcp
from msme_chatbot_agents.mcp_agent_tools import semantic_search_tool, mcp 


tools = [semantic_search_tool]


# Listing of agents for process planner #Add whenever new agents are created
class AgentEnum(str, Enum):
    TranslatorAgent = "TranslatorAgent"
    MSMEGuidelinesAgent = "MSMEGuidelinesAgent"

#Creating a template/format for each sub tasks
class ProcessSubTask(BaseModel):
    task_details: str
    assigned_agent: AgentEnum # we want to assign the task to the agent

#Final Template for all the tasks
class ProcessFullPlan(BaseModel):
    main_task: str
    subtasks: List[ProcessSubTask]
    is_greeting: bool

class ChartData(BaseModel):
    """Pydantic model for chart data structure"""
    chart_type: Literal["bar", "line", "pie", "doughnut", "radar", "polarArea", "scatter"] = Field(
        description="Type of chart to display",
        default="bar"
    )
    labels: List[str] = Field(
        description="Labels for the chart axes or categories",
        min_items=1
    )
    data_records: List[float] = Field(
        description="Numerical data points corresponding to the labels",
        min_items=1
    )
    
#Agent for showing the process plan
class TranslatorAgent:
    def __init__(self, llm,support_doc=None):
        self.llm = llm
        # self.tools = [SQL_query_exec,send_email]+mcp_tools
        self.tools = []
        self.name = "TranslatorAgent"
        self.hand_off_description = "Transfer to TranslatorAgent for translating user requests from English to Telugu and vice versa"
        self.support_doc = support_doc
        self.prompt = self._create_prompt()

    def _create_prompt(self):
            prompt_template = f"""You are a professional, bilingual (English and Telugu) .

                                **Primary Functions:**
                                1.  Respond to user queries about the AP MSME and send to the document search tool for retrieving information.
                                2.  Accurately translate the user's request between English and Telugu before processing the information, if necessary.

                                **CRITICAL LANGUAGE MEMORY RULE:**
                                * Analyze the language of the user's **first message** (English or Telugu). This language is the user's persistent preference for this entire chat session.
                                * You **MUST** respond in this preferred language for every subsequent message, regardless of the language the user types in, to ensure a consistent conversational flow.
                                * Only switch the response language if the user explicitly instructs you to do so (e.g., "Respond in English now").

                                **Initial Action:** Greet the user and briefly introduce yourself in a English language and ask for the preferred language, then immediately analyze their first response to establish the preferred language.
                                """
            return prompt_template


    def create_agent(self,hand_off_tools: list):
        hand_off_tools.extend(self.tools)
        return create_react_agent(self.llm, tools=hand_off_tools,prompt=self.prompt,name=self.name)



#Agent for Insurance data
class MSMEGuidelinesAgent:
    def __init__(self, llm,support_doc=None):
        self.llm = llm
        # self.tools = [SQL_query_exec,send_email]+mcp_tools
        self.tools = [semantic_search_tool]
        self.name = "MSMEGuidelinesAgent"
        self.hand_off_description = "Transfer to MSMEGuidelinesAgent for questions on the MSME guidelines related information by searching the policy, guidelines and other documents stored in vector db"
        self.support_doc = support_doc
        self.prompt = self._create_prompt()

    def _create_prompt(self):
            prompt_template = """You are an MSME Guidelines Document Search Assistant. 
            Whenever user asks for a question related to some MSME guideline, policies and other questions related to MSME, use semantic search results from the msme_guidelines_docs documents to answer customer questions accurately.

            Core Rules
            - Search the vector database before answering
            - Only use information retrieved from the database - never guess or fabricate
            - Cite specific policy sections when providing information
            - If information isn't found, state this clearly

            ## Response Structure
            1. Direct answer to the question
            2. Key details from policy documents (include exact figures, percentages, timeframes)
            3. Important conditions, exclusions, or limitations
            4. Document reference/citation

            ## Search Strategy
            - Use specific terms from the customer's question
            - If results are insufficient, rephrase and search with alternative terms
            - For broad questions, break into multiple specific searches

            ## Style
            - Clear, professional, and customer-friendly
            - Explain MSME jargon in simple terms
            - Be empathetic but factual
            - Never make assumptions about any process or eligibility

            ## Key Guidelines
            - Accuracy over completeness - better to say "not found" than provide wrong information
            - Include ALL relevant conditions and exceptions
            - For ambiguous queries, ask clarifying questions before searching
            - Stay factual - no opinions, only guidelines and policies statements

            Retrieved guideline or policies information will be provided with each query. Base your answer strictly on that information.
            """
            return prompt_template


    def create_agent(self,hand_off_tools: list):
        hand_off_tools.extend(self.tools)
        return create_react_agent(self.llm, tools=hand_off_tools,prompt=self.prompt,name=self.name)



async def main():
    print("Starting MCP agent...")
    await mcp.run()