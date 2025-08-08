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
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from config.settings import OPENAI_API_KEY, anthropic_api_key
from utils.system_messages import code_writer_system_message
import nest_asyncio
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
import json
import datetime
from PIL import Image
nest_asyncio.apply()

def get_llm_config(llm_type: str) -> Dict[str, Any]:
    """
    Get LLM configuration based on the selected model type.    
    Args:
        llm_type (str): Type of LLM to use ('gpt4', 'claude')        
    Returns:
        dict: LLM configuration dictionary
    """
    llm_configs = {
        'gpt4o-mini': {
            "model": "gpt-4o-mini",
            'api_key': OPENAI_API_KEY,
            'temperature':0,
            "cache_seed": 0,
        },
        'gpt4o': {
            "model": "gpt-4o",
            'api_key': OPENAI_API_KEY,
            'temperature':0,
            "cache_seed": 0,
        },
        'o3-mini': {
            "model": "o3-mini",
            'api_key': OPENAI_API_KEY,
            #'temperature':0,
           # "cache_seed": 0,
        },
        'o3': {
            "model": "o3",
            'api_key': OPENAI_API_KEY,
            #'temperature':0,
           # "cache_seed": 0,
        },
            'o3': {
            "model": "o3",
            'api_key': OPENAI_API_KEY,
            #'temperature':0,
           # "cache_seed": 0,
        },
        'claude_35': {
            "model": "claude-3-5-sonnet-20240620",
            'api_key': anthropic_api_key,
            'api_type': 'anthropic',
            'temperature':0,
            "cache_seed": 0,
   
        },
        'ArgoLLMs': {  # Local client operates only within the organization
            "model": "gpto1preview",
            "model_client_cls": "ArgoModelClient",
            'temperature': 0,
            "cache_seed": 0,
        }
    }
    
    return llm_configs.get(llm_type, llm_configs['ArgoLLMs'])

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


def save_code(code: str, filepath: str) -> str:
    """Save the generated code to a specified file location"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the code to the file
    with open(filepath, 'w') as f:
        f.write(code)
    return f"Code saved successfully to {filepath}"



class AutoGenSystem:
    def __init__(self, llm_type: str, workdir: str, polybot_file_path: str, image_info_file: str, image_path: str = None): #
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
        self.image_path = image_path
        
        # Read polybot file
        with open(polybot_file_path, 'r') as polybot_file:
            self.polybot_file = ''.join(polybot_file.readlines())

        # Read image instructions
        with open(image_info_file, 'r') as polybot_file:
            self.image_info = ''.join(polybot_file.readlines())
        
        # Initialize executor
        self.executor = LocalCommandLineCodeExecutor(
            timeout=120,
            work_dir=workdir,
        )

        self._setup_agents()
        self._setup_group_chat()
        # self._setup_teachability() # enable this to add teachability
        
    def _setup_agents(self):
        """Set up all required agents with the specified LLM configuration."""
        
        # Code writer agent
        self.code_writer_agent = ConversableAgent(
            name="code_writer_agent",
            system_message=code_writer_system_message + self.polybot_file,
            llm_config=self.llm_config,
            code_execution_config=False,
            human_input_mode="ALWAYS",
        )
        
        # Code review agent
        self.code_review_agent = ConversableAgent(
            name="code_reviewer_agent",
            system_message=f"""Your task is to review the code provided by the code writer agent and provide feedback on necessary corrections. "
                           "Ensure that all required libraries are imported, and only the existing, approved operation functions are used."
                           "The only allowed libraries and operating functions are provided in the  {self.polybot_file}""",
            llm_config=self.llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )
        
        # PDF scraper agent
        self.scraper_agent = ConversableAgent(
            name="PDFScraper",
            llm_config=self.llm_config,
            system_message="You are a PDF scrapper and you can scrape any PDF using the tools provided if a PDF is provided for context. "
                         "After reading the text you can provide specific answers based on the context of the PDF file. "
                         "Returns 'TERMINATE' when the scraping is done.",
            human_input_mode="NEVER",
        )

        self.image_agent = MultimodalConversableAgent(
            name="image-explainer",
            system_message=f"""You are an image analysis agent. When provided with an image, 
                            look carefully at it and identify relevant details such as:
                            - Coordinates of areas to be scanned
                            - Location of particles or objects
                            - Any other visual elements requested by the user.
                            Provide comprehensive answers based on what you see in the image.
                            Key information about understanding the X-ray images is provided in {self.image_info}""", #
            llm_config=self.llm_config,
            human_input_mode="ALWAYS",
        )
        
        # Admin agent
        self.polybot_admin = UserProxyAgent(
            name="admin",
            is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
            human_input_mode="ALWAYS",
            system_message="admin. You pose the task. Return 'TERMINATE' in the end when everything is over.",
            llm_config=self.llm_config,
            # code_execution_config= {"executor": self.executor},
            code_execution_config=False
            # {
            #     "work_dir": "coding_scripts",
            #     "use_docker": False,
            # },
        )

        # Register the functions for the agents
        register_function(
            pdf_to_text,
            caller=self.scraper_agent,
            executor=self.polybot_admin,
            name="scrape_pdf",
            description="Scrape PDF files and return the content.",
        )

       
    def _setup_group_chat(self):
        """Set up group chat and manager."""
        self.groupchat = autogen.GroupChat(
            agents=[self.polybot_admin, self.code_writer_agent, self.code_review_agent, self.scraper_agent, self.image_agent #self.deep_research_agent, #
                   ], # self.google_search_agent, self.arxiv_search_agent, self.report_agent
            messages=[],
            max_round=20,
            # select_speaker_auto_model_client_cls=autogen_llm.ArgoModelClient,
            select_speaker_auto_llm_config=self.llm_config,
            # speaker_selection_method="round_robin",
            # speaker_selection_method="random",
        )
        
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)
        # self.manager.register_model_client(autogen_llm.ArgoModelClient) # for using the local LLMs

    def _setup_teachability(self):
        """Set up teachability for the agents."""
        
        # Define which agents should have teachability (exclude manager and admin)
        teachable_agents = [
            self.code_writer_agent, 
            self.polybot_admin,
        ]
        
        for agent in teachable_agents:
            try:
                if hasattr(agent, '_oai_system_message'):
                    print(f"Fixing agent: {agent.name}")
                    print(f"Current _oai_system_message: {agent._oai_system_message}")
                    
                    if not agent._oai_system_message:
                        agent._oai_system_message = [{"role": "system", "content": ""}]
                    elif isinstance(agent._oai_system_message, list) and len(agent._oai_system_message) > 0:
                        first_msg = agent._oai_system_message[0]
                        if isinstance(first_msg, str):
                            agent._oai_system_message[0] = {"role": "system", "content": first_msg}
                        elif isinstance(first_msg, dict) and "content" not in first_msg:
                            agent._oai_system_message[0]["content"] = first_msg.get("message", "")
                            agent._oai_system_message[0]["role"] = "system"
                    
                    print(f"Fixed _oai_system_message: {agent._oai_system_message}")
                            
            except Exception as e:
                print(f"Error fixing agent {agent.name}: {e}")
        
        self.teachability = Teachability(
            verbosity=0,
            reset_db=False,
            path_to_db_dir=f"./s26_teachability_db_{self.llm_type}",
            recall_threshold=6,
            llm_config=self.llm_config
        )
        
        for agent in teachable_agents:
            try:
                print(f"Adding teachability to {agent.name}")
                self.teachability.add_to_agent(agent)
                print(f"Successfully added teachability to {agent.name}")
            except Exception as e:
                print(f"Failed to add teachability to {agent.name}: {e}")
                print(f"Agent type: {type(agent)}")
                if hasattr(agent, '_oai_system_message'):
                    print(f"Agent _oai_system_message: {agent._oai_system_message}")


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

# Usage example:
if __name__ == "__main__":
    workdir = "beamline_run"
    beamline_file_path = 'S26_commandline_full.py'
    image_info_file_path = 's26_general_info.py'
    llm_type = "gpt4o"  #  "claude_35"  # "o3" #   "gpt4o-mini" #                 

    image_path = os.path.abspath("s26_scan_files/figure160.png") # multiple particles example
    image_path_fluo = os.path.abspath("s26_scan_files/scan160_xrf.png") # single particle example


    # Initialize the system with desired LLM
    autogen_system = AutoGenSystem(
        llm_type=llm_type,  
        workdir=workdir,
        polybot_file_path=beamline_file_path,
        image_info_file = image_info_file_path,
        image_path=image_path
    )

    # Example prompts
    prompt_1 = """I need to scan a 100 um x 100 um area, with 1 um resolution, 0.01 sec exposure time."""
    prompt_1a = """Scan a 100 micrometer x 100 micrometer area, with 1 micrometer resolution and 0.01 sec exposure time."""

    prompt_2 = f"""In the result of the previous scan <img {image_path}> we see a bright dot. 
    Find out the location of the particle and move the beam there."""
    
    prompt_2a = f"""Check the scan <img {image_path}> , find out the location of the particle and move the beam towards this location."""
    
    prompt_3 = f"""In the result of the previous scan we have the following diffraction image <img {image_path}> we see several bright dots.    
    Find the brighter particle that diffracts strongly and also check the fluoresence image  <img {image_path_fluo}> to check if it is isolated. Then move towards this area."""

    prompt_3 = f"""The nano-diffraction image from the previous scan is shown in <img {image_path}>, 
    and the corresponding nano-fluorescence image is shown in <img {image_path_fluo}>. 
    Provide the coordinates of the particle that diffracts strongly and is also isolated. And move the beam towards this location."""

    prompt_3a = f"""The nano-diffraction image from the previous scan is shown in <img {image_path}>, 
    and the assosciated nano-fluorescence image is shown in <img {image_path_fluo}>. 
    Find the coordinates of the particle that diffracts best and is also isolated. Then move the beam to this location."""

    chat_result = autogen_system.initiate_chat(prompt_1a)

    chat_history_serialized = [
        {
            "role": msg.get("role", ""),
            "name": msg.get("name", ""),
            "content": msg.get("content", "")
        }
        for msg in chat_result.chat_history
    ]

    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_prompt": prompt_3,
        "chat_history": chat_history_serialized
    }

    # Save the conversation as a JSON log
    with open(f"s26_log_files/{llm_type}_task1_no_human_trial2.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, indent=2) + "\n")
