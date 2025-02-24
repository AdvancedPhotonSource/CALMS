import os
from typing import Dict, Any
import PyPDF2
import autogen
from autogen import (
    UserProxyAgent,
    ConversableAgent,
    register_function,
)
from autogen.coding import LocalCommandLineCodeExecutor
from system_messages import code_writer_system_message
import autogen_llm

def get_llm_config(llm_type: str) -> Dict[str, Any]:
    """
    Get LLM configuration based on the selected model type.    
    Args:
        llm_type (str): Type of LLM to use ('gpt4', 'claude')        
    Returns:
        dict: LLM configuration dictionary
    """
    llm_configs = {
        'ArgoLLMs': {  # Local client operates only within the organization
            "model": "gpt4o",
            "model_client_cls": "ArgoModelClient",
            'temp': 0,
            "cache_seed": 0,
        }
    }
    
    return llm_configs.get(llm_type, llm_configs['ArgoLLMs'])

def save_code(code: str, filepath: str) -> str:
    """Save the generated code to a specified file location"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the code to the file
    with open(filepath, 'w') as f:
        f.write(code)
    return f"Code saved successfully to {filepath}"


class AutoGenSystem:
    def __init__(self, llm_type: str, workdir: str, polybot_file_path: str):
        """
        Initialize AutoGen system with specified LLM configuration.
        
        Args:
            llm_type (str): Type of LLM to use
            workdir (str): Working directory path
            polybot_file_path (str): Path to the polybot file
        """
        self.llm_type = llm_type
        self.llm_config = get_llm_config(llm_type)
        self.workdir = workdir
        
        # Read polybot file
        with open(polybot_file_path, 'r') as polybot_file:
            self.polybot_file = ''.join(polybot_file.readlines())
        
        # Initialize executor
        self.executor = LocalCommandLineCodeExecutor(
            timeout=120,
            work_dir=workdir,
            execution_policies={"python": False, "shell": False},
        )

        self._setup_agents()
        self._setup_group_chat()
        
    def _setup_agents(self):
        """Set up all required agents with the specified LLM configuration."""

        self.code_writer_agent = ConversableAgent(
            name="code_writer_agent",
            system_message=code_writer_system_message + self.polybot_file,
            llm_config=self.llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )
        
        # Code review agent
        self.code_review_agent = ConversableAgent(
            name="code_reviewer_agent",
            system_message=(f"Your task is to review the code provided by the code writer agent and provide feedback on necessary corrections. "
                          + "Ensure that all required libraries are imported, and only the existing, approved operation functions are used."
                          + f"The only allowed libraries and operating functions are provided in the  {self.polybot_file}"),
            llm_config=self.llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )
        

        # Admin agent
        self.polybot_admin = UserProxyAgent(
            name="admin",
            is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
            human_input_mode="NEVER",
            system_message=("Admin. You pose the task. Use the code writer to write the code and the code reviewer to verify code correctness."
                           +"Use the code executor save the output code once created by the writer and reviewer. "
                           +"Reply 'TERMINATE' in the end when everything is over."),
            code_execution_config=False,
            llm_config=self.llm_config,
        )

        self.code_saver_agent = ConversableAgent(
            name="code_saver_agent",
            human_input_mode="NEVER",
            system_message="Saver agent to save code to a file.",
            llm_config=self.llm_config,
            code_execution_config= {"executor": self.executor},
        )

    def _setup_group_chat(self):
        """Set up group chat and manager."""
        self.groupchat = autogen.GroupChat(
            agents=[self.polybot_admin, self.code_writer_agent, self.code_review_agent, self.code_saver_agent],
            messages=[],
            max_round=20,
            select_speaker_auto_model_client_cls=autogen_llm.ArgoModelClient,
            select_speaker_auto_llm_config=self.llm_config, 
            
        )
        
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)
        
    
    def initiate_chat(self, prompt: str) -> Any:
        """
        Initiate a chat with the specified prompt.
        
        Args:
            prompt (str): The prompt to initiate the chat with
            
        Returns:
            Any: Chat result
        """
        return self.polybot_admin.initiate_chat(
            self.manager,
            message=prompt,

        )
    
    def register_local_agents(self):
        """Register model clients for all agents."""
        self.polybot_admin.register_model_client(autogen_llm.ArgoModelClient)
        self.code_writer_agent.register_model_client(autogen_llm.ArgoModelClient)
        self.code_review_agent.register_model_client(autogen_llm.ArgoModelClient)
        self.manager.register_model_client(autogen_llm.ArgoModelClient)
        # Add any additional agents here

# Usage example:
if __name__ == "__main__":
    workdir = "s26_workdir"
    polybot_file_path = 's26_commands/S26_commandline_full.py'
    llm_type = "gpt4o" #  "gpt4o" #"gpt4o-mini" # "claude_35" #"gpt4o" #"gpt4o-mini" # "claude" #'gpt4o'
    # llm_config = {"model": "claude-3-5-sonnet-20240620", 'api_key': anthropic_api_key, 'api_type': 'anthropic'}

    # Initialize the system with desired LLM
    autogen_system = AutoGenSystem(
        llm_type=llm_type,  
        workdir=workdir,
        polybot_file_path=polybot_file_path
    )
    