A repository for implementing and testing an autonomous agentic pipeline on a real robotic environment using an N9 robotic station (https://www.northrobotics.com/robots).



### Installation

1. Clone the repository:
```bash
git clone https://github.com/katerinavr/SDL-Agents.git
cd SDL-Agents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add the API keys (config/settings.py):
```bash
OPENAI_API_KEY = ""
anthropic_api_key = ""
```

4. Run the Gradio app:
```bash
python app.py
```

## Components

### Core Files
- `app.py`: Gradio app integrating an option for using human feedback, and an option for displaying live video of the robotic station when a USB camera is connected and has an associated URL address. 
- `n9_robot_operation_commands.py`: Defines the set of available robot operation commands
- `S26_commandline_full.py` : Defines the set of allowed function to operate the X-ray nanoprobe
- `s26_general_info.py` : Description of how to interprete the nano-fluoresence and nano-diffraction X-ray images
- `params.py`: Contains configuration parameters and settings for the system
- `sdl_agents.py`: Main implementation of SDL agents
- `beamline_agents.py`: Main implementation of the agentic workflow in the X-ray nanoprobe

### Teachability Databases
- `teachability_db_gpt4o/`: Contains the ChromaDB with the saved input-output pairs after the human teachings using as a base model GPT-4o
- `teachability_db_gpt4o-mini/`: Contains the ChromaDB with the saved input-output pairs after the human teachings using as a base model GPT-4o-mini

## Examples

- `notebooks`: Contain examples of using the agentic pipeline to operate the N9 robot will tasks of increased complexity.


- `videos`: A video showing the agentic interface to operate the robotic equipment using a simple prompt can be found under assets\sdl_agents_calms_task1.mp4.

![demo](https://raw.githubusercontent.com/katerinavr/SDL-Agents/refs/heads/main/assets/sdl_agents_.gif)