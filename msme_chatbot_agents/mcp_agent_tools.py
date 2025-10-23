from langchain_core.tools import tool
from typing import List, Dict, Any, Optional, Literal, Union, Annotated
# from langgraph.prebuilt import create_react_agent, InjectedState,ToolNode, tools_condition
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from langchain.tools.retriever import create_retriever_tool
# from langgraph.graph import StateGraph, START, END, MessagesState
import psycopg2
import json
import os
import pandas as pd
from langchain_core.messages import ToolMessage
from langchain_experimental.utilities import PythonREPL
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import sys
import asyncio
import azure.cognitiveservices.speech as speechsdk

import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp = FastMCP("Insurance_Agent_Tools")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#SQL query executing tool
@mcp.tool()
def SQL_query_exec(sql_query: str) -> str:
    """ This function takes SQL query as input and executes the query on the postgreSQL data that contains insurance information.
    Args:
        public.insurance_data: The table that contains the insurance information
        sql_query (str): The sql query from the agent
    Returns:
        results: Tabular data after execution of the SQL query"""
    
    print(f"{sql_query}")
    # print(f"{State}")
    load_dotenv()
    
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        print("Database connection successful")
        # print(f"{sql_query}")

        cursor = conn.cursor()

        try:
            cursor.execute(sql_query)
            conn.commit()

            if cursor.description:  # If it's a SELECT query
                column_names = [desc[0] for desc in cursor.description]
                results_list = cursor.fetchall()
                results_df = pd.DataFrame(results_list, columns=column_names)
                result_str = results_df.to_string(index=False)
                print(result_str)
                return result_str


            else:
                return "No results"


        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            conn.rollback()
            return None

        finally:
            cursor.close()

    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
print("SQL query exec executed")

#email sending tool



@mcp.tool()
def send_email(receiver_email, subject, body,tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Sends an email using the specified SMTP server.

    Args:
        receiver_email (str): The recipient's email address.
        subject (str): The email subject.
        body (str): The email body.

    Returns:
        bool: True if the email was sent successfully, False otherwise.
    """
    load_dotenv()
    
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))
        smtp_server="smtp.gmail.com"
        smtp_port=587

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)
            server.send_message(msg)

        return Command(
        update={
            # update the message history
            "messages": [
                ToolMessage(
                    f"Successfully sent the email to {receiver_email}", tool_call_id=tool_call_id
                )
            ],
        }
    )

    except Exception as e:
        print(f"Email sending failed: {e}")
        return False


#Semantic search tool
@mcp.tool()
def semantic_search_tool(query: str) ->  str:
    """
    Performs semantic search on a Qdrant vector database using OpenAI embeddings.
    
    This function connects to a Qdrant vector store, converts the input query into
    an embedding using OpenAI's text-embedding-3-large model, and retrieves the
    most similar document from the 'semantic_search_policy' collection.
    
    Args:
        query (str): The search query string to find similar documents for.
        
    Returns:
        str: The content of the most similar document found in the vector store.
             Returns an empty string if no results are found or if an error occurs.
        
    Raises:
        ValueError: If required environment variables (QDRANT_URL, QDRANT_API_KEY, 
                   OPENAI_API_KEY) are not set.
        Exception: For any connection or search errors with Qdrant.
        
    Environment Variables Required:
        - QDRANT_URL: The URL of your Qdrant instance
        - QDRANT_API_KEY: API key for Qdrant authentication
        - OPENAI_API_KEY: API key for OpenAI embeddings
        
    Example:
        >>> result = semantic_search_tool("What are the surrender charges?")
        >>> print(result)
        "Surrender charges apply when premiums are withdrawn..."
    """
    try:
        # Load environment variables
        load_dotenv()
      
        
        # Validate required environment variables
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        # embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")
        
        # Connect to Qdrant
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=True)
        
        # Initialize vector store
        collection_name = "semantic_search_policy"
        # logger.info(f"Using collection: {collection_name}")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        
        # Perform similarity search
        results = vector_store.similarity_search(query, k=1)
        
       
        # Return the top result content
        if results:
            return results[0].page_content
        else:
            return ""
            
    except Exception as e:
        raise
print("Semantic search tool executed")



# #text-to-speech tool
# @mcp.tool()
# def text_to_speech(text_input: str):
#     """
#     Speaks out/ reads out the given text input

#     Args:
#         text_input (str): The input text that needs to converted into audio or speech

#     Returns:
#         bool: True if the audio was created successfully, False otherwise.
#     """
#     load_dotenv()

#     # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
#     speech_config = speechsdk.SpeechConfig(subscription=os.getenv("SPEECH_KEY"), region=os.getenv('SPEECH_REGION'))
#     audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

#     # The neural multilingual voice can speak different languages based on the input text.
#     speech_config.speech_synthesis_voice_name='en-US-AvaMultilingualNeural'

#     speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    
#     speech_synthesis_result = speech_synthesizer.speak_text_async(text_input).get()
    
#     return speech_synthesis_result


#     # return Command(
#     #     update={
#     #         # update the message history
#     #         "messages": [
#     #             ToolMessage(
#     #                 "Successfully converted to audio", tool_call_id=tool_call_id
#     #             )
#     #         ],
#     #     }
#     # )



