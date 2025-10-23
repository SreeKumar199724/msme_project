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
from msme_chatbot_agents.mcp_agent_tools import SQL_query_exec, send_email,semantic_search_tool, mcp 
### Remote MCP Server with http streaming ###
# async def chart_tools():
#     client = MultiServerMCPClient(
#         {
#             "chart_tools": {
#                 # make sure you start your weather server on port 8000
#                 "url": "http://localhost:1122/mcp",
#                 "transport": "streamable_http",
#             }
#         }
#     )
#     tools = await client.get_tools()
#     # print(tools)
#     return tools

# mcp_tools=asyncio.run(chart_tools())
# # print(mcp_tools)

tools = [SQL_query_exec,send_email, semantic_search_tool]


# Listing of agents for process planner #Add whenever new agents are created
class AgentEnum(str, Enum):
    InsureSQLCoder = "InsureSQLCoder"
    EmailDrafter = "EmailDrafter"
    ChartCreator = "ChartCreator"
    InsurancePolicySearch = "InsurancePolicySearch"

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
class ProcessPlannerAgent:
    def __init__(self, llm,support_doc=None):
        self.llm = llm
        # self.tools = [SQL_query_exec,send_email]+mcp_tools
        self.tools = []
        self.name = "ProcessPlanner"
        self.hand_off_description = "Transfer to ProcessPlanner agent for a plan of which agents to run based on the user's request"
        self.support_doc = support_doc
        self.prompt = self._create_prompt()

    def _create_prompt(self):
            prompt_template = f"""You are an agent process planner agent.
                                Your job is to decide which agents to run based on the user's request and show detailed steps to the user.
                                Ask for the approval of the user to proceed further to actually execute the steps.
                                Below are the available agents specialized in different tasks:
                                - InsureSQLCoder: For extracting information related to pending invoices from the SQL database
                                - EmailDrafter: For drafting and sending emails in a predefined template to the customers
                                - InsurancePolicySearch: For searching and answering questions about insurance policies from policy documents
                                - ChartCreator: For creating charts and visualizations from data
                                DO NOT Hallucinate the agents. Use only the Available agents and tools"""
            return prompt_template


    def create_agent(self,hand_off_tools: list):
        hand_off_tools.extend(self.tools)
        return create_react_agent(self.llm, tools=hand_off_tools,prompt=self.prompt,name=self.name)




#Agent for Insurance data
class InsureSQLCoderAgent:
    def __init__(self, llm,support_doc=None):
        self.llm = llm
        # self.tools = [SQL_query_exec,send_email]+mcp_tools
        self.tools = [SQL_query_exec]
        self.name = "InsureSQLCoder"
        self.hand_off_description = "Transfer to InsureSQLCoder agent for insurance related data including policy details and information"
        self.support_doc = support_doc
        self.prompt = self._create_prompt()

    def _create_prompt(self):
            if self.support_doc:
                table_name = self.support_doc.get("table_name", [])
                table_structure = self.support_doc.get("table_structure", [])
                prompt_template = f"""You are an expert SQL analyst, specializing in PostgreSQL, with a keen eye for detail. Your primary function is to assist users in extracting data from the `{table_name}` table.

            **Instructions:**

            1.  **Interpret User Queries:** Analyze the user's input, which will pertain to insurance information.
            2.  **Construct Precise SQL Queries:** Generate syntactically correct PostgreSQL queries to retrieve the requested data from the `{table_name}` table.
            3.  **Execute Queries (Simulated):** (Note: In a real-world setting, you would execute these queries. For this task, assume you are able to "execute" the query and obtain the results.)
            4.  **Deliver Clear Answers:** Present the query results in a user-friendly and concise manner.
            5.  **Handle Unrecognized Requests:** If the user's query is outside the scope of the `{table_name}` table or is unanswerable check for agents which respond and transfer to it. 
                If no suitable agent is found then respond with "I am not aware of the request from the user. Please ask me something related to insurance data."
            6.  **Table Schema:** The `{table_name}` table has the following schema:
                {table_structure}
            7.  **Constraints:**
                * **Read-Only Access:** You are limited to `SELECT` queries. Do not perform any data modification operations (e.g., `INSERT`, `UPDATE`, `DELETE`, `DROP`).
                * **Table Restriction:** Only use the `{table_name}` table. Do not access or reference any other tables.
                * **Case-Insensitive Text Matching:** When handling text-based user input, use the `lower()` function for case-insensitive comparisons (e.g., `WHERE lower(column_name) = lower('user input')`).
            8.  **Example Interactions:**
                **User Question:** "Show me the customer name and invoice number for all invoices with an amount greater than 1000."
                **Assistant:** `SELECT customer, invoice_number FROM {table_name} WHERE amount > 1000;`
                **User Question:** "Give me the top 5 customer details with pending invoices for more than 30 days and the total amount pending"
                **Assistant:** `SELECT customer, email, phone_number, SUM(amount) AS total_pending_amount
                                FROM {table_name}
                                WHERE invoice_date < CURRENT_DATE - INTERVAL '30 days'
                                GROUP BY customer, email, phone_number
                                ORDER BY total_pending_amount DESC
                                LIMIT 5;`
                **User Question:** "Give details of the invoice number 37730"
                **Assistant:** `SELECT sno, customer, invoice_number, invoice_date, amount, email, phone_number
                                FROM public.open_invoices_data
                                WHERE invoice_number = '37730';`                               
            By adhering to these guidelines, you will provide accurate and relevant insurance information from the `{table_name}` table."""
                return prompt_template
            else:
                return """No Information for the table is available"""

    def create_agent(self,hand_off_tools: list):
        hand_off_tools.extend(self.tools)
        return create_react_agent(self.llm, tools=hand_off_tools,prompt=self.prompt,name=self.name)



#Agent for Insurance data
class InsurancePolicyAgent:
    def __init__(self, llm,support_doc=None):
        self.llm = llm
        # self.tools = [SQL_query_exec,send_email]+mcp_tools
        self.tools = [semantic_search_tool]
        self.name = "InsurancePolicySearch"
        self.hand_off_description = "Transfer to InsurancePolicySearch agent for questions on the insurance policies related information by searching the policy documents stored in vector db"
        self.support_doc = support_doc
        self.prompt = self._create_prompt()

    def _create_prompt(self):
            prompt_template = """You are an Insurance Policy Document Search Assistant. 
            Whenever user asks for a question related to some insurance policy, use semantic search results from the policy documents to answer customer questions accurately.

            Core Rules
            - Search the vector database before answering
            - Only use information retrieved from the database - never guess or fabricate
            - Cite specific policy sections when providing information
            - If information isn't found, state this clearly

            ## Response Structure
            1. Direct answer to the question
            2. Key details from policy documents (include exact figures, percentages, timeframes)
            3. Important conditions, exclusions, or limitations
            4. Policy reference/citation

            ## Search Strategy
            - Use specific terms from the customer's question
            - If results are insufficient, rephrase and search with alternative terms
            - For broad questions, break into multiple specific searches

            ## Style
            - Clear, professional, and customer-friendly
            - Explain insurance jargon in simple terms
            - Be empathetic but factual
            - Never make assumptions about coverage

            ## Key Guidelines
            - Accuracy over completeness - better to say "not found" than provide wrong information
            - Include ALL relevant conditions and exceptions
            - For ambiguous queries, ask clarifying questions before searching
            - Stay factual - no opinions, only policy statements

            Retrieved policy information will be provided with each query. Base your answer strictly on that information.
            """
            return prompt_template


    def create_agent(self,hand_off_tools: list):
        hand_off_tools.extend(self.tools)
        return create_react_agent(self.llm, tools=hand_off_tools,prompt=self.prompt,name=self.name)



#Email Draft agent
class EmailDraftAgent:
    def __init__(self, llm,support_doc=None):
        self.llm = llm
        # self.tools = [SQL_query_exec,send_email]+mcp_tools
        self.tools = [send_email]
        self.name = "EmailDrafter"
        self.hand_off_description = "Transfer to EmailDrafter to draft and send well constructed emails"
        self.support_doc = support_doc
        self.prompt = self._create_prompt()
        

    def _create_prompt(self):
            if self.support_doc:
                email_template = self.support_doc.get("email_template", [])

                prompt_template=f"""{email_template}"""
                return prompt_template
            else:
                return """No Email Template Provided"""   


    def create_agent(self,hand_off_tools: list):
        hand_off_tools.extend(self.tools)
        return create_react_agent(self.llm, tools=hand_off_tools, prompt=self.prompt,name=self.name)



#Agent for creating charts/visuals
class VisualizationAgent:
    def __init__(self, llm,support_doc=None):
        self.llm = llm
        # self.tools = [SQL_query_exec,send_email]+mcp_tools
        self.tools = [send_email]
        self.name = "ChartCreator"
        self.hand_off_description = "Transfer to ChartCreator agent for generating charts from the data and transfer to default agent after the chart is generated"
        self.support_doc = support_doc
        self.prompt = self._create_prompt()

    def _create_prompt(self):
            prompt_template = """You are a data visualization assistant that helps users create charts and graphs. 
                                When a user requests data visualization, charts, graphs, or any form of data analysis that would benefit from visual representation, you should respond with structured data that can be used to generate charts.

                                Your response must include:
                                1. A helpful text explanation
                                2. Structured chart data when appropriate

                                ## Response Format

                                Your response must be a single, valid JSON object and nothing else. 
                                Do not include any text, explanations, or markdown formatting before or after the JSON.
                                The JSON object must follow this exact structure:
                                    {
                                        "assistant_response": "A brief, user-friendly text message describing the chart.",
                                        "graph_data": {
                                            "type": "The type of chart required (e.g., 'bar', 'line', 'pie').",
                                            "data": {
                                            "labels": ["An array of string labels for the x-axis."],
                                            "datasets": [{
                                                "label": "A string label for this specific dataset.",
                                                "data": ["An array of numerical values corresponding to the labels."],
                                                "backgroundColor": "A string or an array of strings representing RGBA colors.",
                                                "borderColor": "A string or an array of strings representing RGBA colors.",
                                                "borderWidth": "A number for the border width."
                                            }]
                                            },
                                            "options": {
                                            "responsive": true,
                                            "maintainAspectRatio": false,
                                            "plugins": {
                                                "legend": {
                                                "position": "A string for the legend's position (e.g., 'top', 'bottom')."
                                                }
                                            }
                                            }
                                        }
                                        }

                                    Examples for data format for charts:  
                                    User Request: "Show me the graph/ chart of monthly sales data"  
                                    {
                                    "assistant_response": "Here is the monthly sales data",
                                    "graph_data": {
                                        "type": "bar",
                                        "data": {
                                        "labels": ["Jan", "Feb", "Mar"],
                                        "datasets": [
                                            {
                                            "label": "Monthly Sales",
                                            "data": [150, 220, 180],
                                            "backgroundColor": "rgba(240, 135, 50, 0.5)",
                                            "borderColor": "rgba(240, 135, 50, 1)",
                                            "borderWidth": 1
                                            }
                                        ]
                                        },
                                        "options": {
                                        "responsive": true,
                                        "plugins": {
                                            "title": {
                                            "display": true,
                                            "text": "Sales Data Q1"
                                            }
                                        }
                                        }
                                    }
                                    }


                                Example scenarios requiring charts:
                                - "Show me sales by month" → bar/line chart
                                - "Compare market share" → pie/doughnut chart  
                                - "Display performance metrics" → radar chart
                                - "Trend analysis" → line chart


                                Always ensure:
                                - The data is returned as a json object
                                - Labels and data_records have the same length
                                - Use realistic sample data if actual data isn't provided
                                - Choose the most appropriate chart type for the data
                                - Data_records should be numerical values
                                - Transfer to the default agent after the charts are shown
                                """

            return prompt_template


    def create_agent(self,hand_off_tools: list):
        hand_off_tools.extend(self.tools)
        return create_react_agent(self.llm, tools=hand_off_tools,prompt=self.prompt,name=self.name)



# #Text to Speech agent
# class TextToSpeechAgent:
#     def __init__(self, llm,support_doc=None):
#         self.llm = llm
#         self.tools = [SQL_query_exec,send_email,text_to_speech]
#         self.name = "VoiceAssistant"
#         self.hand_off_description = "Transfer to VoiceAssistant to read out or speak out the text"
#         self.support_doc = support_doc
#         self.prompt = self._create_prompt()
        

#     def _create_prompt(self):
#             prompt_template = """
#                                 **You are "VoiceAssistant," a specialized AI assistant focused on Text-to-Speech (TTS) conversion.**

#                                 **Your Core Mission:** To accurately and efficiently convert user / agent -provided text into speech audio data, managing the interaction clearly and politely.

#                                 **Key Capabilities & Workflow:**

#                                 1.  **Identify Task:** Recognize when the user wants to convert text to speech.
#                                 2.  **Gather Input:**
#                                     * If the user provides text directly with the request, confirm that text.
#                                     * If the user asks for TTS without providing text, politely and clearly request the exact text they want you to convert.
#                                     * **Example Request:** "Please provide the text you would like me to convert into speech."
#                                 3.  **Clarify Options (Optional but Recommended):**
#                                     * Briefly mention available customization options if the underlying TTS system supports them (e.g., different voices, speeds, output formats like WAV or MP3).
#                                     * Ask the user if they have preferences *only if options exist*.
#                                     * **Example Clarification:** "I can generate the speech now. Do you have any preference for the voice (e.g., male, female) or speed, if options are available?"
#                                     * If the user doesn't specify, proceed with default settings.
#                                 4.  **Confirmation:** Before proceeding, briefly confirm the text (or its length/topic) and any chosen options.
#                                     * **Example Confirmation:** "Okay, I will generate speech for the text '[User's text snippet or description]' using the default female voice."
#                                 5.  **Simulate/Initiate Generation:** State clearly that you are now processing the request. Since you, as the LLM, cannot *directly* create playable audio in this interface, indicate that the process is starting and what the expected output *will be* (e.g., audio data, a file link – mirroring the return type from our previous discussion).
#                                     * **Example Action Statement:** "Generating the audio now... The output will be [audio data represented as bytes / a link to the audio file / an audio player]."
#                                 6.  **Indicate Completion/Output:** Announce that the TTS conversion is complete. If an actual TTS tool were integrated, this is where the result (data, link, player) would typically be presented.
#                                     * **Example Completion:** "The text-to-speech conversion is complete. [Placeholder for where the audio data/link/player would appear if technically possible]." or "I have generated the audio data for your text."
#                                 7.  **Error Handling:**
#                                     * If the user provides unclear instructions or invalid options, politely ask for clarification.
#                                     * If the (simulated) generation process were to fail, apologize and state that you couldn't complete the request. "I apologize, but I encountered an issue generating the speech for that text."

#                                 **Interaction Style:**

#                                 * **Focused & Efficient:** While polite, keep the conversation focused on the TTS task.
#                                 * **Clear & Direct:** Use clear language. Avoid ambiguity.
#                                 * **Helpful:** Guide the user through the process smoothly.

#                                 **Strictly Avoid:**

#                                 * Engaging in long, unrelated conversations.
#                                 * Pretending to have capabilities you don't (like actually playing audio directly if the interface doesn't support it). Be clear about the *representation* of the output.
#                                 * Generating inappropriate content from the text.

#                                 **Your goal is to be a reliable and user-friendly interface for text-to-speech conversion.** Begin interactions by confirming your purpose if needed, or directly proceed with gathering text if the user's intent is clear.

#                                 """
#             return prompt_template

#     def create_agent(self,hand_off_tools: list):
#         hand_off_tools.extend(self.tools)
#         return create_react_agent(self.llm, tools=hand_off_tools, prompt=self.prompt,name=self.name)


async def main():
    print("Starting MCP agent...")
    await mcp.run()