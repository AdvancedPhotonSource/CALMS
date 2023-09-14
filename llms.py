
from pydantic import Extra
import requests
import datetime, os, shutil
import params

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

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

        if stop is None:
            stop_param = []
        else:
            stop_param = stop
        
        req_obj = {'user':'APS', 'prompt':[prompt], "stop":stop_param}
        result = requests.post(self.anl_url, json=req_obj)

        response = result.json()['response']

        if self.debug:
            with open(self.debug_fp, 'a+') as debug_f:
                debug_f.write(f'Response:{response}')

        return response

    @property
    def _identifying_params(self):
        return {}


def init_text_splitter():
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=params.chunk_size, 
                                                    chunk_overlap=params.chunk_overlap,
                                                    length_function = len,
                                                    separators = ['\n\n','\n', '.']
                                                    )
    return text_splitter


def init_facility_qa(embeddings, params):
    embed_path = params.embed_path

    if params.init_docs:
        text_splitter = init_text_splitter()

        if os.path.exists(embed_path):
            if params.overwrite_embeddings:
                shutil.rmtree(embed_path)
            else:
                raise ValueError("Existing Chroma Collection")

        all_texts = []
        for doc_path in params.doc_paths: #Iterate over text files in each path
            print ("Reading docs from", doc_path)
            for text_fp in os.listdir(doc_path):
                with open(os.path.join(doc_path, text_fp), 'r') as text_f:
                    book = text_f.read()
                texts = text_splitter.split_text(book)
                all_texts += texts

        docsearch = Chroma.from_texts(
            all_texts, embeddings, #metadatas=[{"source": str(i)} for i in range(len(all_texts))],
            persist_directory=embed_path
        )
        docsearch.persist()
    else:
        docsearch = Chroma(embedding_function=embeddings, persist_directory=embed_path)
    print ("Finished embedding documents")

    return docsearch