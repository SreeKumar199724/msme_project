from typing import Annotated,List, Optional, Union, Dict, Any
from typing_extensions import TypedDict, Literal
from pydantic import BaseModel, Field, model_validator
from enum import Enum
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import importlib
import json
import uuid
import inspect
import logging
from IPython.display import Image, display
from PIL import Image as PILImage

from langgraph.graph import StateGraph, START, END, MessagesState, add_messages
from langchain_openai import ChatOpenAI
from langgraph_swarm import create_handoff_tool, create_swarm, SwarmState
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
import azure.cognitiveservices.speech as speechsdk

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Add parent directory to path to import msme_chatbot_agents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

#import the custom created agents
# from msme_chatbot_agents import mcp_agents
# from msme_chatbot_agents import mcp_agent_tools
import msme_chatbot_agents
from msme_chatbot_agents.mcp_agents import *
# from msme_chatbot_agents import ProcessPlannerAgent, InsurancePolicyAgent, InsureSQLCoderAgent, ProcessFullPlan, VisualizationAgent
# from msme_chatbot_agents import mcp_agents



load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ##OpenAI LLM
# llm = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     api_key=os.getenv("OPENAI_API_KEY")
# )

#Azure OpenAI model
try:
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",  # or your deployment
        api_version="2024-12-01-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    logger.info("Azure OpenAI LLM initialized.")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI LLM: {e}")



class ChatResponse(BaseModel):
    """Complete response model that includes text and optional chart data"""
    assistant_response: str = Field(description="Text response from the assistant")
    # Change the type from 'ChartData' to a flexible dictionary
    graph_data: Optional[Dict[str, Any]] = Field(
        description="Optional chart data object for Chart.js", default=None
    )


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    bot_messages: Annotated[list, add_messages]
    chart_request: bool = False
    structured_output: Optional[ChatResponse] = None



class WorkflowBuilder:
    def __init__(self, llm):
        self.llm = llm
        self.support_mongodb_client = None

 
    def build_workflow(self, *agent_names, checkpointer):
        agents_needed = []
        agent_names_list = []
        

        module_name = "msme_chatbot_agents.mcp_agents"
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logger.error(f"Could not import agent module '{module_name}': {e}")
            return None

        hand_off_tools_dict = {}


        #MongoDB Connection along with supporting Documents for prompts
        # --- MongoDB Connection ---
        mongodb_uri=os.getenv("MONGODB_URI")
        if not mongodb_uri:
            logger.error("MONGODB_URI environment variable not set.")
            return None
        
        # support_mongodb_client  = None
        try:
            self.support_mongodb_client = MongoClient(mongodb_uri, server_api=ServerApi('1'),
                                                      maxPoolSize=10,minPoolSize=1)
            self.support_mongodb_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")

        
        # --- End MongoDB Connection ---

        support_doc_db=os.getenv("SUPPORT_DOC_DB_NAME")
        support_doc_col=os.getenv("SUPPORT_DOC_COLLECTION")
        # db = mongodb_client[support_doc_db] 
        # collection = db[support_doc_col]
        if support_doc_db and support_doc_col:
            try:
                db = self.support_mongodb_client[support_doc_db] 
                collection = db[support_doc_col]
                logger.info(f"Connected to support document collection: {support_doc_db}.{support_doc_col}")
            except Exception as e:
                logger.warning(f"Could not connect to support document collection: {e}")
        else:
            logger.warning("Support document DB or collection not configured. Proceeding without support docs.")


        #Creating a list of hand-off tools for all the given agents
        for agent_name in agent_names:
            try:
                if hasattr(module,agent_name):
                    agent1 = getattr(module, agent_name)
                    doc = collection.find_one({"_id": agent_name})
                    if doc:
                        doc.pop("_id", None) # remove mongoDB id
                        agent = agent1(llm=self.llm, support_doc=doc)   ##Instantiating the agent class with support docs
                        # print(f"{agent_name} class is instantiated with support docs")
                        try:
                            hand_off_tools_dict[agent_name] = create_handoff_tool(agent_name=agent.name, description=agent.hand_off_description)
                            logger.info(f"Created handoff tool for {agent_name}")
                        except AttributeError:
                            logger.warning(f"Agent class '{agent_name}' found but lacks expected attributes (e.g., name, hand_off_description). Skipping handoff tool.")
                        except Exception as e:
                            logger.error(f"Error creating handoff tool for {agent_name}: {e}")

                    else:
                        agent = agent1(llm=self.llm, support_doc=None)   ##Instantiating the agent class without support docs
                        try:
                            hand_off_tools_dict[agent_name] = create_handoff_tool(agent_name=agent.name, description=agent.hand_off_description)
                            logger.info(f"Created handoff tool for {agent_name}")
                        except AttributeError:
                            logger.warning(f"Agent class '{agent_name}' found but lacks expected attributes (e.g., name, hand_off_description). Skipping handoff tool.")
                        except Exception as e:
                            logger.error(f"Error creating handoff tool for {agent_name}: {e}")
                          
                
            except AttributeError:
                print(f"Warning: Agent '{agent_name}' not found in 'agents'. Skipping.")
       
        #Instantiating agents with the requried supporting docs stored in MongoDB and the hand-off tools
        for agent_name in agent_names:
            try:
                if hasattr(module,agent_name):
                    agent1 = getattr(module, agent_name)
                    doc = collection.find_one({"_id": agent_name})
                    if doc:
                        doc.pop("_id", None) # remove mongoDB id
                        agent = agent1(self.llm,support_doc=doc) ##Instantiating the agent class with support docs
                        logger.info(f"Instantiated agent: {agent_name} with support docs.")
                        hand_off_tools = [value for key, value in hand_off_tools_dict.items() if key != agent_name]
                        agents_needed.append(agent.create_agent(hand_off_tools))
                        agent_names_list.append(agent.name)

                        
                    else:  #This step is to define the process planner agent to show the steps in a structured format
                        if agent_name=="ProcessPlannerAgent" and 'ProcessFullPlan' in locals():
                            try: 
                             agent_llm = self.llm.with_structured_output(ProcessFullPlan, include_raw=True)
                             logger.info("Using structured output LLM for ProcessPlannerAgent.")
                            except Exception as e:
                             logger.error(f"Failed to configure structured output for ProcessPlannerAgent: {e}. Using default LLM.")
                             agent_llm = self.llm

                            agent = agent1(llm=agent_llm,support_doc=None)  ##Instantiating the agent class without support docs
                            logger.info(f"Instantiated agent: {agent_name} without support docs.")
                            hand_off_tools = [value for key, value in hand_off_tools_dict.items() if key != agent_name]
                            agents_needed.append(agent.create_agent(hand_off_tools))
                            agent_names_list.append(agent.name)
     
                        else:
                            agent = agent1(self.llm,support_doc=None)  ##Instantiating the agent class without support docs
                            logger.info(f"Instantiated agent: {agent_name} without support docs.")
                            hand_off_tools = [value for key, value in hand_off_tools_dict.items() if key != agent_name]
                            agents_needed.append(agent.create_agent(hand_off_tools))
                            agent_names_list.append(agent.name)
                                             

            except AttributeError:
                logger.error(f"Warning: Agent '{agent_name}' not found in 'agents'. Skipping.")
        if not agents_needed:
            logger.error("No valid agents could be created. Workflow cannot be built.")
            return None
        logger.info(f"Agents prepared for swarm: {agent_names_list}")

        
        logger.info(f"Successfully created {len(agents_needed)} agent instances")
        logger.info(f"Agents prepared for swarm: {agent_names_list}")
        
        # Using Synchronous MongoDB for (Short Term) Persistent memory- checkpointer_db created in MongoDB
        # checkpointer = MongoDBSaver(mongodb_client) 
     

        try:
            default_agent = "ProcessPlanner" if "ProcessPlanner" in agent_names_list else agent_names_list[0]
            logger.info(f"Using '{default_agent}' as default active agent.")

            graph_builder = create_swarm(
                agents=agents_needed, default_active_agent="ProcessPlanner"
            )

            logger.info("LangGraph swarm flow created")

      
            graph = graph_builder.compile(checkpointer=checkpointer)
            logger.info("LangGraph swarm compiled successfully with checkpointer.")
            return graph
        except Exception as e:
            logger.error(f"Failed to create or compile swarm graph: {e}")
            return None
    
 

# generating unique thread id for each session
def generate_thread_id():
    return str(uuid.uuid4())  # Generates a unique UUID

thread_id = generate_thread_id()

config = {"configurable": {"thread_id": thread_id}}


## Text to speech function
def text_to_speech(text_input: str):
    """
    Speaks out/ reads out the given text input

    Args:
        text_input (str): The input text that needs to converted into audio or speech

    Returns:
        bool: True if the audio was created successfully, False otherwise.
    """

    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=os.getenv("SPEECH_KEY"), region=os.getenv('SPEECH_REGION'))
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    # The neural multilingual voice can speak different languages based on the input text.
    speech_config.speech_synthesis_voice_name='en-US-AvaMultilingualNeural'

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    
    speech_synthesis_result = speech_synthesizer.speak_text_async(text_input).get()
    
    return speech_synthesis_result

## Speech to text
def recognize_from_microphone():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    speech_config.speech_recognition_language="en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
    return speech_recognition_result.text


agent_list = []
for name, obj in inspect.getmembers(msme_chatbot_agents.mcp_agents):
    if inspect.isclass(obj) and name.endswith('Agent'):
        agent_list.append(name)


#Async Mongodb checkpointer initiation
checkpointer = None
async def initialize_checkpointer():
    """Initialize the async MongoDB checkpointer and return the actual saver instance"""
    async_mongodb_uri = os.getenv("ASYNC_MONGODB_URI")
    if not async_mongodb_uri:
        raise ValueError("ASYNC_MONGODB_URI environment variable not set.")
    
    # Create the context manager
    checkpointer_cm = AsyncMongoDBSaver.from_conn_string(async_mongodb_uri)
    
    # Enter the context manager to get the actual checkpointer instance
    checkpointer_instance = await checkpointer_cm.__aenter__()
    
    return checkpointer_instance, checkpointer_cm

async def initialize_application():
    global graph, builder, checkpointer, checkpointer_context
    try:
        # Get both the checkpointer instance and its context manager
        checkpointer, checkpointer_context = await initialize_checkpointer()
        logger.info("Asynchronous MongoDB Checkpointer initialized.")
        
        # Initialize workflow builder
        builder = WorkflowBuilder(llm)
        graph = builder.build_workflow(*agent_list, checkpointer=checkpointer)

        if graph is None:
            logger.error("Failed to build the graph. Exiting.")
            raise RuntimeError("Workflow graph initialization failed.")

        try:
            display(Image(graph.get_graph().draw_mermaid_png()))
            image_data = graph.get_graph().draw_mermaid_png()
            with open("mermaid_graph.png", "wb") as f:
                f.write(image_data)
        except Exception as e:
            logger.warning(f"Could not save graph visualization: {e}")
            
        return graph
    
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    global builder, checkpointer, checkpointer_context
    # Startup
    logger.info("Starting up FastAPI application...")
    try:
        await initialize_application()
        logger.info("Application initialized successfully")
        yield
    finally:
        # Shutdown - cleanup resources
        logger.info("Shutting down FastAPI application...")
        
       
        # Clean up async checkpointer
        if checkpointer_context:
            try:
                await checkpointer_context.__aexit__(None, None, None)
                logger.info("Async checkpointer context closed successfully")
            except Exception as e:
                logger.error(f"Error closing checkpointer context: {e}")

# --- FastAPI Application ---
app = FastAPI(
    title="LangGraph Swarm Chatbot API",
    description="API endpoint for interacting with the LangGraph Agent Swarm",
    lifespan=lifespan
)


# CORS (Cross-Origin Resource Sharing) Middleware
# Allows requests from your Streamlit frontend (adjust origins if needed)
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3003",
    "http://localhost:8503",

    # Add any other origins if necessary (e.g., deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    user_input: str
    thread_id: str # Important for maintaining stateful conversations

class VoiceResponse(BaseModel):
    """Response model for voice input endpoint"""
    text: str = Field(description="Transcribed text from speech")
    success: bool = Field(description="Whether transcription was successful")
    error: Optional[str] = Field(description="Error message if any", default=None)


@app.post("/voice-input", response_model=VoiceResponse)
async def voice_input_endpoint():
    """
    Endpoint to capture voice input from microphone and convert to text using Azure Speech Services
    """
    try:
        logger.info("Voice input endpoint called")
        transcribed_text = recognize_from_microphone()

        if transcribed_text:
            logger.info(f"Successfully transcribed: {transcribed_text}")
            return VoiceResponse(text=transcribed_text, success=True)
        else:
            logger.warning("No text transcribed from voice input")
            return VoiceResponse(text="", success=False, error="No speech detected")

    except Exception as e:
        logger.error(f"Error in voice input endpoint: {e}", exc_info=True)
        return VoiceResponse(text="", success=False, error=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_input: UserInput):
    global graph
    logger.info(f"Received request for thread_id: {user_input.thread_id}")
    config = {"configurable": {"thread_id": user_input.thread_id}}
    input_message = {"messages": [HumanMessage(content=user_input.user_input)]}

    try:
        final_state = None
        async for event in graph.astream(input_message, config=config):
            final_state = event

        if final_state:
            last_message = None
            state_messages = None
            for key, value in final_state.items():
                if isinstance(value, dict) and "messages" in value:
                    state_messages = value["messages"]
                    break
            
            if state_messages is None and "messages" in final_state:
                state_messages = final_state["messages"]

            if state_messages:
                for msg in reversed(state_messages):
                    if isinstance(msg, AIMessage):
                        last_message = msg
                        break

            if last_message:
                logger.info(f"Raw LLM response: {last_message.content}")
                
             
                try:
                   
                    llm_output = json.loads(last_message.content)
                    
                   
                    return ChatResponse(
                        assistant_response=llm_output.get("assistant_response", "Here's the information you requested."),
                        graph_data=llm_output.get("graph_data") # This is now a direct pass-through
                    )

                except json.JSONDecodeError:
                    # If LLM returns plain text, not JSON
                    logger.warning("LLM output was not valid JSON. Returning as plain text.")
                    ast_response = "There's a mismatch in the chart format. Kindly explain the data to be plotted"
                    # return ChatResponse(assistant_response=str(ast_response), graph_data=None)
                    return ChatResponse(assistant_response=str(last_message.content), graph_data=None)


        # Fallback response if no message is found
        logger.error(f"Graph streaming finished without a final state for thread_id: {user_input.thread_id}")
        return ChatResponse(assistant_response="I'm sorry, I couldn't generate a response.")

    except Exception as e:
        logger.error(f"Error during graph interaction for thread_id {user_input.thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# python main.py ---- command to execute the backend


# --- Run with Uvicorn ---
# Example command: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# The __main__ block is convenient for simple execution but 'uvicorn' command is standard for deployment
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with Uvicorn...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8503,
        reload=True,
        log_level="info",
        access_log=True
    )
# --- End Run ---