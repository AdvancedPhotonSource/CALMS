#pip install unstructured
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,2,3'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

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
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") #AutoTokenizer.from_pretrained("model_name")
    os.mkdir(p.tokenizer_path)
    tokenizer.save_pretrained(p.tokenizer_path)

#Setup pipeline
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")#p.model_name)#, device_map="auto")#, load_in_8bit=True)
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
        with open(os.path.join(doc_path, text_fp), 'r', encoding="utf-8") as text_f:
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

print ("Finished embedding documents")


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

def init_chain():
    memory = ConversationBufferWindowMemory(memory_key="history", 
                                            input_key = "input", 
                                            k=6)

    conversation = LLMChain(
            prompt=PROMPT,
            llm=local_llm, 
            verbose=True, 
            memory=memory
    )

    return memory, conversation


#Setup Gradio app

#Common Methods 
def user(user_message, history):
    return "", history + [[user_message, None]]

def init_chat_layout():
    chatbot = gr.Chatbot(show_label=False).style(height="500")
    msg = gr.Textbox(label="Send a message with Enter")
    clear = gr.Button("Clear")

    return chatbot, msg, clear

#Page layout
with gr.Blocks(css="footer {visibility: hidden}", title="APS ChatBot") as demo:

    #Header
    gr.Markdown("""
    # Hi! I am the APS AI Assistant
    ### I was trained at Meta, taught to follow instructions at Stanford and am now learning about the APS. AMA!
    """
    )
    gr.Markdown("""
    * Use the General Chat to AMA. E.g. write some code for you, create a recipe from ingredients etc. 
    * Use the APS Q&A to ask me questions specific to the APS, I will look up answers using the documentation my trainers have provided me. 
    * Use the Document Q&A to ask me questions about a document you provide.
    """)

    #General chat tab
    with gr.Tab("General Chat"):

        memory1, conversation1 = init_chain() #Init chain
        chatbot, msg, clear = init_chat_layout() #Init layout

        def bot_no_context(history):
            user_message = history[-1][0] #History is list of tuple list. E.g. : [['Hi', 'Test'], ['Hello again', '']]
            bot_message = conversation1.predict(input=user_message, context="")
            #Pass user message and get context and pass to model
            history[-1][1] = "" #Replaces None with empty string -- Gradio code

            for character in bot_message:
                history[-1][1] += character
                time.sleep(0.02)
                yield history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_no_context, chatbot, chatbot #Use bot without context
        )
        clear.click(lambda: memory1.clear(), None, chatbot, queue=False)

    #APS Q&A tab
    with gr.Tab("Document Q&A"):
        gr.Markdown("""
        Q&A over uploaded document
        """
        )
        def loading_pdf():
            return "Loading..."

        def pdf_changes(pdf_docs):
            for pdf_doc in pdf_docs:
                loader = OnlinePDFLoader(pdf_doc.name)
                documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(documents)
                db = Chroma.from_documents(texts, embeddings)
                retriever = db.as_retriever()
                llm = local_llm 
                global qa
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
                #result = qa({"query": 'what is the strategy of APS Scientific Computing?'})
                return "Ready"

        def add_text(history, text):
            history = history + [(text, None)]
            return history, ""

        def bot(history):
            response = infer(history[-1][0])
            history[-1][1] = response['result']
            return history

        def infer(question):
            query = question
            result = qa({"query": query})
            return result


        title = """
        <div style="text-align: center;max-width: 700px;">
            <h1>Chat with PDF</h1>
            <p style="text-align: center;">Upload one or more PDFs from your computer, click the "Load PDFs to LangChain" button, <br />
            when everything is ready, you can start asking questions about the pdf ;)</p>
            <a style="display:inline-block; margin-left: 1em"></a>
        </div>
        """

        with gr.Column(elem_id="col-container"):
          gr.HTML(title)
        
          with gr.Column():
              pdf_doc = gr.File(label="Load PDFs", file_types=['.pdf'], type="file", file_count = 'multiple')
              #repo_id = gr.Dropdown(label="LLM", choices=["eachadea/vicuna-13b-1.1", "bigscience/bloomz"], value="eachadea/vicuna-13b-1.1")
              with gr.Row():
                  langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                  load_pdf = gr.Button("Load pdf to langchain")
          
          chatbot = gr.Chatbot([], elem_id="chatbot").style(height=350)
          question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
          submit_btn = gr.Button("Send message")

        load_pdf.click(pdf_changes, inputs=[pdf_doc], outputs=[langchain_status], queue=False)
        question.submit(add_text, [chatbot, question], [chatbot, question]).then(
            bot, chatbot, chatbot
        )
        submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
            bot, chatbot, chatbot
        )
    
         
    with gr.Tab("Tips & Tricks"):
        gr.Markdown("""
        1. I am not as powerful as GPT-4 or ChatGPT and I am running on cheap GPUs, if I get stuck, you can type "please continue" or similar and I will attempt to complete my thought.
        2. If I don't give a satisfactory answer, try rephrasing your question. For e.g. 'Can I do high energy diffraction at the APS?' instead of 'Where can I do high energy diffraction at the APS?
        """
        )



    #Footer
    gr.Markdown("""
    ##### Made with ❤️ for 🧑‍🔬 by:
    Mathew J. Cherukara, Michael Prince @ APS
    Henry Chan, Aikaterini Vriza @ CNM
    Varuni K. Sastry @ DSL/ALCF
    """
    )


demo.queue()
demo.launch(server_name="0.0.0.0", server_port=2023)
