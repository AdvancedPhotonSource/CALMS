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
from config.settings import OPENAI_API_KEY, anthropic_api_key, mem0_key
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
from mem0 import MemoryClient
import PyPDF2

# Function for scraping PDFs
def pdf_to_text(pdf_file: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_file (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Adding a memory client 
client = MemoryClient(api_key=mem0_key)

# config_list = [{'model': 'gpt-4o', 'api_key': OPENAI_API_KEY}]
# llm_config = {"model": "gpt-4", 'api_key': OPENAI_API_KEY}  #gpt-3.5-turbo, gpt-4o

# llm_config = {"model": "llama2:7b", "base_url": "http://localhost:11434/v1", "api_type": "ollama"}


llm_config = {"model": "claude-3-5-sonnet-20240620", 'api_key': anthropic_api_key, 'api_type': 'anthropic'}


with open('C:/Users/cnmuser/Desktop/CALMS/polybot_experiment.py', 'r') as polybot_file:
    POLYBOT_FILE = ''.join(polybot_file.readlines())

POLYBOT_FILE_FILTER = POLYBOT_FILE.replace("{", "")
POLYBOT_FILE_FILTER = POLYBOT_FILE_FILTER.replace("}", "")
POLYBOT_FILE_LINES = len(POLYBOT_FILE.split('\n'))

POLYBOT_RUN_FILE_PATH = "C:/Users/Public/robot/N9_demo_3d/polybot_screenshots/polybot_screenshots.py"
POLYBOT_RUN_FILE = ''.join(open(POLYBOT_RUN_FILE_PATH).readlines())
POLYBOT_RUN_FILE_FILTER = POLYBOT_RUN_FILE.replace("{", "").replace("}", "")
POLYBOT_RUN_FILE_LINES = len(POLYBOT_RUN_FILE.split('\n'))

"""
===============================
Python Execution Tools
===============================
"""
def exec_cmd(py_str: str):
    """
    Placeholder for the function. While in testing, just keeping it as a print statement
    """
    print(py_str)
    
    return "Command Executed"


def lint_cmd(py_str: str, lint_fp, py_pfx = None): # = 'agent_scripts/tmp_lint.py'
    import subprocess
    """
    Helper function to enable linting.
    Creates a file, prepends text to it, lints it, then removes the file.
        py_str: string to lint
        py_pfx: prefix to add to string. Used if appending py_str to an existing python file
    """
    with open(lint_fp, 'w') as lint_file:
        if py_pfx is not None:
            lint_file.write(py_pfx)
            lint_file.write("\n")
        lint_file.write(py_str)


    # Pylint's internal reporter API fails on so just use subprocess which sesm to be more reliable
    result = subprocess.run([r"c:/Users/Public/robot/polybot-env/python.exe", "-m", "pylint", lint_fp, "-d R,C,W"], stdout=subprocess.PIPE)
    
    #"C:\Users\cnmuser\.conda\envs\calms\python.exe"
    result_str = result.stdout.decode('utf-8')

    with open(lint_fp, 'w') as lint_file:
        pass
    # os.remove(lint_fp)

    result_str_split = result_str.split('\n')
    result_str = '\n'.join(result_str_split[1:])

    return result_str

def filter_pylint_lines(lint_output, start_ln):
    """
    Filter out the pylint lines that are not needed for the output
    """
    filtered_ouput = []
    for line in lint_output.split('\n'):
        if line.startswith("*********"):
            filtered_ouput.append(line)

        line_split = line.split(':') 
        if len(line_split) > 1:
            if line_split[1].isdigit():
                if int(line.split(':')[1]) > start_ln:
                    filtered_ouput.append(line)

    return '\n'.join(filtered_ouput)


"""
===============================
Polybot Tools
===============================
"""

def polybot_exec_cmd(py_str: str):
    with open('C:/Users/cnmuser/Desktop/CALMS/polybot_commands.py', 'r') as polybot_file:
        POLYBOT_FILE = ''.join(polybot_file.readlines())
        POLYBOT_FILE_FILTER = POLYBOT_FILE.replace("{", "")
        POLYBOT_FILE_FILTER = POLYBOT_FILE_FILTER.replace("}", "")

    f"""After the code is checked for error. It takes in a python string and execs it in the environment described by the script."
        + "The script will contain objects and functions used to interact with the instrument. "
        + "Here are some rules to follow: \n"
        + "Before running the experiment create a new python file with all the library imports (robotics, loca, rack_status, proc, pandas, etc.) or any other list that is required."
        + "Check if the requested polymer is available in the rack_status and then directly proceed with the experimental excecution"
        + "Some useful commands and instructions are provided below \n\n + {POLYBOT_FILE_FILTER} """
    
    POLYBOT_RUN_FILE_PATH = "C:/Users/Public/robot/N9_demo_3d/polybot_screenshots/polybot_screenshots.py"
    file_path = POLYBOT_RUN_FILE_PATH
    
    # Write the command to the file
    with open(file_path, 'a') as file:
        file.write(py_str + '\n')
    
    return "Command Saved"


def polybot_linter(py_str: str):
    """
    Linting tool for Polybot. Prepends the Polybot file.

    Tool for checking the quality and correctness of the code. Always call this tool first before writing or executing any code."
    +  "Takes in a python string and lints it."
    + " Always run the linter to check the code before running it."
    + " The output will provide suggestions on how to improve the code."
    + " Attempt to correct the code based on the linter output."
    + " Rewrite the code until there are no errors. "
    + " Otherwise, fix the code and check again using linter."
    """
    print("running linter......")
    POLYBOT_RUN_FILE_PATH = "C:/Users/Public/robot/N9_demo_3d/polybot_screenshots/polybot_screenshots.py"
    lint_fp = POLYBOT_RUN_FILE_PATH # 'agent_scripts/tmp_lint.py' #POLYBOT_RUN_FILE_PATH
    POLYBOT_RUN_FILE = ''.join(open(POLYBOT_RUN_FILE_PATH).readlines())
    POLYBOT_RUN_FILE_FILTER = POLYBOT_RUN_FILE.replace("{", "").replace("}", "")
    lint_output = lint_cmd(py_str, lint_fp, py_pfx=POLYBOT_RUN_FILE_FILTER)
    # lint_output = filter_pylint_lines(lint_output, POLYBOT_RUN_FILE_LINES)
    
    if ':' not in lint_output:
        lint_output += '\nNo errors.'
        
    return lint_output

# temp_dir = tempfile.TemporaryDirectory()
workdir = "C:/Users/Public/robot/N9_demo_3d/polybot_screenshots/polybot_screenshots_run"

# Local command line code executor.
executor = LocalCommandLineCodeExecutor(
    timeout=120,  # Timeout for each code execution in seconds.
    work_dir=workdir,  
    functions=[lint_cmd, polybot_linter], #polybot_exec_cmd
)

# Agent with code writing capabilities
code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

# assignt the default coder system message
code_writer_agent_system_message = code_writer_agent.system_message
code_writer_agent_system_message += executor.format_functions_for_prompt()
print(code_writer_agent_system_message)

code_writer_agent = ConversableAgent(
    name="code_writer_agent",
    system_message=code_writer_agent_system_message + POLYBOT_FILE,
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

code_review_agent = ConversableAgent(
    name="code_reviewer_agent",
    system_message="Your task is to review the code writen from the code writer agent and provide feedback for corrections",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="ALWAYS",
)

polybot_admin = UserProxyAgent(
    name="admin",
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    system_message="admin. You pose the task. Return 'TERMINATE' in the end when everything is over.",
    llm_config = llm_config,
    code_execution_config={"executor": executor},
)


message = "Pick up polymer A and move it to the clamp."

groupchat = autogen.GroupChat(
    agents=[polybot_admin, code_review_agent, code_review_agent], messages=[], max_round=50
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

chat_result = polybot_admin.initiate_chat(
    code_writer_agent,
    message=message,
)

