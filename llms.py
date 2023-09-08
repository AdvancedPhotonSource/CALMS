from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pydantic import Extra
import requests
import datetime
import os
import torch

from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class AnlLLM(LLM, extra=Extra.allow):

    def __init__(self, params):
        super().__init__()

        self.debug = params.anl_llm_debug 
        self.debug_fp = params.anl_llm_debug_fp
        
        with open(params.anl_llm_url_path, 'r') as url_f:
            self.anl_url = url_f.read().strip()

    @property
    def _llm_type(self) -> str:
        return "ANL LLM API"

    def _call(
        self,
        prompt: str,
        stop = None,
        run_manager = None,
    ) -> str:
        if self.debug:
            with open(self.debug_fp, 'a+') as debug_f:
                debug_f.write(f'\n\n{datetime.datetime.now()}\nPrompt:{prompt}')

        if stop is not None:
            print(stop)
            raise ValueError("stop kwargs are not permitted.")
        
        req_obj = {'user':'aps', 'prompt': prompt}
        result = requests.post(self.anl_url, json=req_obj)

        response = result.json()['response']

        if self.debug:
            with open(self.debug_fp, 'a+') as debug_f:
                debug_f.write(f'Response:{response}')

        return response

    @property
    def _identifying_params(self):
        return {}


def init_local_llm(params):
    #Create a local tokenizer copy the first time
    if os.path.isdir(params.tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(params.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(params.model_name)
        os.mkdir(params.tokenizer_path)
        tokenizer.save_pretrained(params.tokenizer_path)

    #Setup pipeline
    model = AutoModelForCausalLM.from_pretrained(params.model_name, 
                                                 device_map="auto", 
                                                 torch_dtype=torch.bfloat16)#, load_in_8bit=True)
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=params.seq_length,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2
    )


    #Setup LLM chain with memory and context
    return HuggingFacePipeline(pipeline=pipe)


def init_local_embeddings(params):
    return HuggingFaceEmbeddings(model_name=params.embedding_model_name)