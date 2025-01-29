import os
import sys
import json
import autogen
from autogen import (
    UserProxyAgent,
    AssistantAgent,
    ConversableAgent,
    register_function,
    GroupChat,
    GroupChatManager
)

from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from chromadb.utils import embedding_functions
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
import pandas as pd
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.agentchat.contrib.capabilities.teachability import Teachability
import PyPDF2
import autogen_llm


# Configurations of the LLM models
from config.settings import OPENAI_API_KEY, anthropic_api_key


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
# client = MemoryClient(api_key=mem0_key)

# config_list = [{'model': 'gpt-4o', 'api_key': OPENAI_API_KEY}]
# llm_config = {"model": "gpt-4", 'api_key': OPENAI_API_KEY}  #gpt-3.5-turbo, gpt-4o

# llm_config = {"model": "llama2:7b", "base_url": "http://localhost:11434/v1", "api_type": "ollama"}


# llm_config = {"model": "claude-3-5-sonnet-20240620", 'api_key': anthropic_api_key, 'api_type': 'anthropic'}

llm_config = {
    "model": "gpt4turbo",
    "model_client_cls": "ArgoModelClient",
    'temp': 0
}

with open('C:/Users/cnmuser/Desktop/CALMS/polybot_experiment.py', 'r') as polybot_file:
    POLYBOT_FILE = ''.join(polybot_file.readlines())

# temp_dir = tempfile.TemporaryDirectory()
workdir = "C:/Users/Public/robot/N9_demo_3d/polybot_screenshots/polybot_screenshots_run"

# Local command line code executor.
executor = LocalCommandLineCodeExecutor(
    timeout=120,  # Timeout for each code execution in seconds.
    work_dir=workdir,  
    # functions=[lint_cmd, polybot_linter], #polybot_exec_cmd
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
# code_writer_agent_system_message += executor.format_functions_for_prompt()
# print(code_writer_agent_system_message)


code_writer_agent = ConversableAgent(
    name="code_writer_agent",
    system_message=code_writer_agent_system_message + POLYBOT_FILE,
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="ALWAYS",
)

code_review_agent = ConversableAgent(
    name="code_reviewer_agent",
    system_message="""Your task is to review the code writen from the code writer agent and provide feedback for corrections.
    """,
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)


# Create PDF scrapper agent.
scraper_agent = ConversableAgent(
    "PDFScraper",
    llm_config=llm_config,
    system_message="You are a PDF scrapper and you can scrape any PDF using the tools provided if a PDF is provided for context."
    "After reading the text you can provide specific answers based on the context of the PDF file."
    "Returns 'TERMINATE' when the scraping is done.", # 
    human_input_mode="NEVER",
)

polybot_admin = UserProxyAgent(
    name="admin",
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="ALWAYS",
    system_message="admin. You pose the task. Return 'TERMINATE' in the end when everything is over.",
    llm_config = llm_config,
    # code_execution_config={"executor": executor},
    code_execution_config=False)

# Register the function with the agents.
register_function(
    pdf_to_text,
    caller=scraper_agent,
    executor=polybot_admin,
    name="scrape_pdf",
    description="Scrape PDF files and return the content.",
)


prompt_1 = f"""Write the execution code to move the vial with PEDOT:PSS defined as polymer A to the clamp holder. 
        """

# prompt_1a = """Write the code to move the vial with polymer A to the clamp location."""

prompt_2 = f"""Write the execution code to pick up a substrate and move it to the coating station."""
# # prompt_2 = "Write Python execution code to pick up a substrate and move it to the coating station using a robotic arm."

# prompt_2a = """Write the code to pick up a substrate and move it to the coating stage."""

prompt_3 = """Write the execution code to create a polymer film using only PEDOT:PSS defined as polymer A. 
        Extract the best range of the film processing conditions from the paper PEDOT_PSS_manuscript.pdf.
        """

# prompt_3a = """Write the code to create a polymer film with only PEDOT:PSS defined as polymer A. 
# Identify the best processing conditions from the paper PEDOT_PSS_manuscript.pdf."""

# Unmark these lines to add teachability to the models
# teachability = Teachability(
#     verbosity=0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
#     reset_db=False,
#     path_to_db_dir="./teachability_db_gpt-4o",
#     recall_threshold=5,  # Higher numbers allow more (but less relevant) memos to be recalled.
# )

code_writer_agent.register_model_client(autogen_llm.ArgoModelClient)
# code_review_agent.register_model_client(autogen_llm.ArgoModelClient)
# scraper_agent.register_model_client(autogen_llm.ArgoModelClient)
polybot_admin.register_model_client(autogen_llm.ArgoModelClient)


groupchat = autogen.GroupChat(
    agents=[polybot_admin, code_writer_agent, code_review_agent, scraper_agent], messages=[], max_round=20  #prev_chat
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
manager.register_model_client(autogen_llm.ArgoModelClient)

# teachability.add_to_agent(code_writer_agent)
# teachability.add_to_agent(code_review_agent)
# teachability.add_to_agent(manager)

chat_result = polybot_admin.initiate_chat(
    manager,
    message=prompt_1 #, clear_history=False
)

