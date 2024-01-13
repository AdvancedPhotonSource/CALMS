# CALMS : Context-Aware Language Model for Science

CALMS is a retrieval and tool augmented large language model (LLM) to assist scientists, design experiments around and perform science using complex scientific instrumentation. 



https://github.com/mcherukara/CALMS/assets/20727490/6ed99a11-7923-4d44-9684-bffab525b4b6

## Getting started

1. conda create --name calms python=3.11.5

2. git clone https://github.com/mcherukara/CALMS

3. Navigate to the folder, activate your conda environment, then:
pip install -r requirements_H100.txt 

4. git checkout argparse

- The VERY FIRST time you run each model, you will have to compute embeddings over the document stores. You can do this by setting init_docs = True in params.py before starting the chat app
—- this will take a LONG time but only needs to be run once

5. Start the app:
   
python chat_app.py --openai
-- for OpenAI models (choose which one in params.py)

(OR)

python chat_app.py —hf 
-- for open-source models (choose which one in params.py)
-- Recommend at least 50 GB of GPU memory for LLAMA family of models

Please note you will have to provide your own OpenAI and Materials Project API keys
