import os
import sys
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Union
import json
import logging
from dataclasses import dataclass

import autogen
from autogen import (
    UserProxyAgent,
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from chromadb.utils import embedding_functions
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
import pandas as pd
from autogen.coding import LocalCommandLineCodeExecutor

# Configurations of the LLM models
from config.settings import OPENAI_API_KEY, anthropic_api_key

# config_list = [{'model': 'gpt-4o', 'api_key': OPENAI_API_KEY}]
# llm_config = {"model": "gpt-4", 'api_key': OPENAI_API_KEY}  #gpt-3.5-turbo, gpt-4o

llm_config = {"model": "llama2:7b", "base_url": "http://localhost:11434/v1", "api_key": "ollama"}


# llm_config = {"model": "claude-3-5-sonnet-20240620", 'api_key': anthropic_api_key, 'api_type': 'anthropic'}
import requests
from langchain.tools import BaseTool, StructuredTool#, Tool, tool
from pydantic import Extra
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
# import pexpect
import os
import subprocess
# import params
import tempfile

from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor


import autogen


#{"model": "llama2:7b", "base_url": "http://localhost:11434/v1", "api_key": "ollama"}
llm_config_local = {"config_list": [{
    "model": "llama2:7b", 
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama"
}]}

bob = autogen.AssistantAgent(
    name="Bob",
    system_message=""""
      You love telling jokes. After Alice feedback improve the joke. 
      Say 'TERMINATE' when you have improved the joke.
    """,
    llm_config=llm_config_local
)

alice = autogen.AssistantAgent(
    name="Alice",
    system_message="Criticise the joke.",
    llm_config=llm_config_local
)

def termination_message(msg):
    return "TERMINATE" in str(msg.get("content", ""))

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={"use_docker": False},
    is_termination_msg=termination_message,
    human_input_mode="NEVER"
)

groupchat = autogen.GroupChat(
    agents=[bob, alice, user_proxy],
    speaker_selection_method="round_robin",
    messages=[]
)

manager = autogen.GroupChatManager(
    groupchat=groupchat, 
    code_execution_config={"use_docker": False},
    llm_config=llm_config,
    is_termination_msg=termination_message
)

user_proxy.initiate_chat(
    manager, 
    message="Tell a joke"
)