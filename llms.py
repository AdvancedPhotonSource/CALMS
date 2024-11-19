from typing import List
from pydantic import Extra
from tqdm import tqdm
import requests
import datetime, os, shutil
import params
import time

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings


class AnlLLM(LLM, extra=Extra.allow):

    def __init__(self, params):
        super().__init__()

        self.debug = params.anl_llm_debug 
        self.debug_fp = params.anl_llm_debug_fp

        self.temperature = 0.1
        self.top_p = 0.1
        
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

        print(f'Model = {params.anl_llm_model}')
        
        req_obj = {'user': params.anl_user, 
                   'model': params.anl_llm_model, 
                   'prompt': [prompt], 
                   'system': "",
                   'stop': stop_param, 
                   'temperature': self.temperature}
                   #'top_p': self.top_p}
        result = requests.post(self.anl_url, json=req_obj)
        if not result.ok:
            print(f"error {result.status_code} ({result.reason})")
            return

        response = result.json()['response']

        if self.debug:
            with open(self.debug_fp, 'a+') as debug_f:
                debug_f.write(f'Response:{response}')

        return response

    @property
    def _identifying_params(self):
        return {}


class ANLEmbeddingModel(Embeddings):
    def __init__(self, params):
        super().__init__()
        with open(params.anl_embed_url_path, 'r') as url_f:
            self.embed_url = url_f.read().strip()
        self.pagination = 16 # Limit imposed by OpenAI
        
    def embed_query(self, text: str):
        return self._query_api_single(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        output_embeds = []
        if len(texts) > self.pagination:
            pbar = tqdm(total=(len(texts)//self.pagination), desc='Batch Embed Calls')
        for i in range(0, len(texts), self.pagination):
            embeds_page = self._query_api_multiple(texts[i:i+self.pagination])
            if len(texts) > self.pagination:
                #time.sleep(3) # Prevent from overloading the API. 
                pbar.update(1)
            output_embeds += embeds_page

        if len(texts) > self.pagination:
            pbar.close() 

        return output_embeds
    
    def _query_api_multiple(self, texts: List[str]):
        req_obj = {'user':params.anl_user, 'model':'', 'prompt':texts, 'stop':[]}
        result = requests.post(self.embed_url, json=req_obj)
        if result.ok:
            return result.json()['embedding']
        print(f"error {result.status_code} ({result.reason})")

    def _query_api_single(self, text: str):
        req_obj = {'user':params.anl_user, 'model':'', 'prompt':[text], 'stop':[]}
        result = requests.post(self.embed_url, json=req_obj)
        if result.ok:
            return result.json()['embedding'][0]
        print(f"error {result.status_code} ({result.reason})")


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
    else:
        docsearch = Chroma(embedding_function=embeddings, persist_directory=embed_path)
    print ("Finished embedding documents")

    return docsearch

def write_list(all_texts):
    with open(params.pdf_text_path+'/pdf.txt', 'w') as file:
        for text in all_texts:
            file.write(text.page_content + '\n')
    file.close()

"""
===========================
NER Functionality
===========================
"""

import spacy

nlp = spacy.load('en_core_web_lg')

def get_subject(doc): #Extract subject
    for token in doc:
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]

def get_object(doc): #Extract object
    for token in doc:
        if ("dobj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            return doc[start:end]

def extract_proper_nouns(doc): #Extract proper nouns
    pos = [tok.i for tok in doc if tok.pos_ == "PROPN"]
    consecutives = []
    current = []
    for elt in pos:
        if len(current) == 0:
            current.append(elt)
        else:
            if current[-1] == elt - 1:
                current.append(elt)
            else:
                consecutives.append(current)
                current = [elt]
    if len(current) != 0:
        consecutives.append(current)
    return [doc[consecutive[0]:consecutive[-1]+1] for consecutive in consecutives]


def ner_hits(query): #Extract subject, object and NER strings

    doc = nlp(query)
    nouns = extract_proper_nouns(doc)
    subject = get_subject(doc)
    object = get_object(doc)

    all_nouns = []
    if nouns is not None:
        nouns = [noun.text.strip() for noun in nouns] #Spacy object to string
        for noun in nouns: 
            if len(noun)>params.min_NER_length: all_nouns.append(noun)

    if subject is not None:
        subject = subject.text.strip()
    else : subject = ""

    if object is not None:
        object = object.text.strip()
    else : object = ""
        
    print("Subject:", subject)
    print("Object:", object)
    print("Proper Nouns", all_nouns)

    uniques = list(set(all_nouns + [subject] + [object])) #Merge unique elements
    print ("Merged NER list: ", uniques)

    uniques = list(filter(lambda i: len(i) >= params.min_NER_length , uniques))
    print ("Filtered NER list: ", uniques) #Only consider NERs > a set length
    
    return uniques