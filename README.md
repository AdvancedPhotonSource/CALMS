# CALMS : Context-Aware Language Model for Science

CALMS is a retrieval and tool augmented large language model (LLM) to assist scientists, design experiments around and perform science using complex scientific instrumentation. 



https://github.com/mcherukara/CALMS/assets/20727490/6ed99a11-7923-4d44-9684-bffab525b4b6

<br/><br/>

### Getting started

1. conda create --name calms python=3.11.5

2. git clone https://github.com/mcherukara/CALMS

3. Navigate to the folder, activate your conda environment, then:
   
   pip install -r requirements_H100.txt 

4. Start the app:
- The VERY FIRST time you run each model, you will have to compute embeddings over the document stores. You can do this by setting init_docs = True in params.py before starting the chat app. This will take a LONG time but only needs to be run once
  
- python chat_app.py --openai
  
   > for OpenAI models (choose which one (GPT3.5, GPT4 etc. ) in params.py)

(OR)

- python chat_app.py —hf
  
   > for open-source models (choose which one (Vicuna etc.) in params.py)

   > Recommend at least 50 GB of GPU memory for LLAMA family of models

**Please note you will have to provide your own OpenAI and Materials Project API keys**

6. Navigate to localhost:2023 for the open-source model and localhost:2024 for the openai model
   > Ports can be set in chat_app.py

<br/><br/>

#### DISCLAIMER
The content presented in this paper has been generated using pre-trained Large Language Models (LLMs), specifically GPT 3.5 and [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), by injecting contextual prompts into these LLM pipelines through a retrieval and augmentation tool. The generated content is reported as is, without any manipulation or alteration of the LLM outputs. The authors acknowledge that LLM-generated content may contain errors, biases, or inaccuracies, which could significantly impact the scientific workflows in which they are incorporated. It is important to note that the current code base is not production-ready and requires additional checks and balances before being used for large-scale deployment. Furthermore, the authors disclaim any responsibility or liability for the accuracy, completeness, or reliability of LLM-generated content presented in this paper.
