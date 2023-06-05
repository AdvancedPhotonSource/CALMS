import os
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

device = "cuda" if torch.cuda.is_available() else "cpu"
#Need only 1 GPU if loading 8-bit model
print("Device:", device)

print("Using %d GPUs" %torch.cuda.device_count())

import gradio as gr
gr.close_all() #Close any existing open ports
import time, shutil
import params as p

#Create a local tokenizer copy the first time
if os.path.isdir(p.tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(p.tokenizer_path)
else:
    tokenizer = AutoTokenizer.from_pretrained("model_name")
    os.mkdir(p.tokenizer_path)
    tokenizer.save_pretrained(p.tokenizer_path)

#Setup pipeline
model = AutoModelForCausalLM.from_pretrained(p.model_name, device_map="auto")#, load_in_8bit=True)
pipe = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=p.seq_length,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2
)


#Load embedding model and use that to embed text from source
embeddings = HuggingFaceEmbeddings(model_name=p.embedding_model_name)
embed_path = 'embeds/%s' %(p.embedding_model_name)

if p.init_docs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=p.chunk_size, chunk_overlap=p.chunk_overlap)

    if os.path.exists(embed_path):
        #response = input("WARNING: DELETING EXISTING EMBEDDINGS. Type \"y\" to continue.")
#        if response.strip() == "y":
        if p.overwrite_embeddings:
            shutil.rmtree(embed_path)
        else:
            raise ValueError("Existing Chroma Collection")

    all_texts = []
    doc_path = 'APS-Science-Highlight'
    for text_fp in os.listdir(doc_path):
        with open(os.path.join(doc_path, text_fp), 'r') as text_f:
            book = text_f.read()
        texts = text_splitter.split_text(book)
        all_texts += texts

    docsearch = Chroma.from_texts(
        all_texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(all_texts))],
        persist_directory=embed_path
    )
    docsearch.persist()
else:
    docsearch = Chroma(persist_directory=embed_path)

#Method to find text with highest likely context
def get_context(query):
    
    docs = docsearch.similarity_search_with_score(query, k=p.N_hits)
    #Get context strings
    context=""
    for i in range(p.N_hits):
        context += docs[i][0].page_content +"\n"
        print (i+1, docs[i][0].page_content)
    return context



#Setup LLM chain with memory and context
local_llm = HuggingFacePipeline(pipeline=pipe)

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Context:
{context}

Current conversation:
{history}
Human: {input}
AI:"""

PROMPT = PromptTemplate(
    input_variables=["history", "input", "context"], template=template
)

memory = ConversationBufferWindowMemory(memory_key="history", 
                                        input_key = "input", 
                                        k=6)

conversation = LLMChain(
        prompt=PROMPT,
        llm=local_llm, 
        verbose=True, 
        memory=memory
)


#Setup Gradio app
with gr.Blocks(css="footer {visibility: hidden}", title="APS ChatBot") as demo:
    gr.Markdown("""
    # Welcome to the APS AI Assistant!
    I was trained at Meta, taught to follow instructions at Stanford and am now learning about the APS. AMA!
    """
    )
    chatbot = gr.Chatbot(show_label=False).style(height="500")
    msg = gr.Textbox(label="Send a message with Enter")
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):

        user_message = history[-1][0] #History is list of tuple list. E.g. : [['Hi', 'Test'], ['Hello again', '']]
        bot_message = conversation.predict(input=user_message, context=get_context(user_message))
        #Pass user message and get context and pass to model
        history[-1][1] = "" #Replaces None with empty string -- Gradio code

        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.02)
            yield history
    

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: memory.clear(), None, chatbot, queue=False)
    gr.Markdown("""
    Made with ❤️ for 🧑‍🔬 by:
    Mathew J. Cherukara, Michael Prince @ APS
    Henry Chan, Aikaterini Vriza @ CNM
    Varuni K. Sastry @ DSL/ALCF
    """
    )


demo.queue()
demo.launch(server_name="0.0.0.0", server_port=2023)
