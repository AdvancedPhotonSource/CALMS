import os
import params
if params.set_visible_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = params.visible_devices

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
import gradio as gr
import time, shutil

#Load embedding model and use that to embed text from source
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("Using %d GPUs" %torch.cuda.device_count())
gr.close_all() #Close any existing open ports


"""
===========================
Initialization
===========================
"""

def init_llm(params):
    #Create a local tokenizer copy the first time
    if os.path.isdir(params.tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(params.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("model_name")
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

    #Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=params.embedding_model_name)

    #Setup LLM chain with memory and context
    return HuggingFacePipeline(pipeline=pipe), embeddings


def init_aps_qa(embeddings, params):
    embed_path = 'embeds/%s' %(params.embedding_model_name)

    if params.init_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=params.chunk_size, chunk_overlap=params.chunk_overlap)

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
            all_texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(all_texts))],
            persist_directory=embed_path
        )
        docsearch.persist()
    else:
        docsearch = Chroma(embedding_function=embeddings, persist_directory=embed_path)
    print ("Finished embedding documents")

    return docsearch


"""
===========================
Chat Functionality
===========================
"""

class Chat():
    def __init__(self, llm, embedding, doc_store):
        self.llm = llm 
        self.embedding = embedding
        self.memory, self.conversation = self._init_chain()
        self.doc_store = doc_store


    def _init_chain(self):
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
                llm=self.llm, 
                verbose=True, 
                memory=memory
        )

        return memory, conversation


    #Method to find text with highest likely context
    def _get_context(self, query, doc_store):
        docs = doc_store.similarity_search_with_score(query, k=params.N_hits)
        #Get context strings
        context=""
        for i in range(params.N_hits):
            context += docs[i][0].page_content +"\n"
            print (i+1, docs[i][0].page_content)
        return context
    
    
    def generate_response(self, history, debug_output):
        user_message = history[-1][0] #History is list of tuple list. E.g. : [['Hi', 'Test'], ['Hello again', '']]

        if self.doc_store is None:
            context = ""
        else:
            context = self._get_context(user_message, self.doc_store)

        if debug_output:
            inputs = self.conversation.prep_inputs({'input': user_message, 'context':context})
            prompt = self.conversation.prep_prompts([inputs])[0][0].text

        bot_message = self.conversation.predict(input=user_message, context=context)
        #Pass user message and get context and pass to model
        history[-1][1] = "" #Replaces None with empty string -- Gradio code

        if debug_output:
            bot_message = f'---Prompt---\n\n {prompt} \n\n---Response---\n\n {bot_message}'

        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.02)
            yield history

    def add_message(self, user_message, history):
        return "", history + [[user_message, None]]


class PDFChat(Chat):
    def update_pdf_docstore(self, pdf_docs):
        all_pdfs = []
        for pdf_doc in pdf_docs:
            loader = OnlinePDFLoader(pdf_doc.name)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=params.chunk_size, 
                                                           chunk_overlap=params.chunk_overlap)
            texts = text_splitter.split_documents(documents)
            all_pdfs += texts
        embed_path = 'embeds/pdf'
        db = Chroma.from_documents(all_pdfs, self.embedding, metadatas=[{"source": str(i)} for i in range(len(all_pdfs))],
            persist_directory=embed_path) #Compute embeddings over pdf using embedding model specified in params file
        db.persist()

        self.doc_store = db

        return "PDF Ready"


"""
===========================
UI/Frontend
===========================
"""
def init_chat_layout():
    chatbot = gr.Chatbot(show_label=False, elem_id="chatbot")#.style(height="500")
    with gr.Row():
        with gr.Column(scale=0.85):
            msg = gr.Textbox(show_label = False,
                placeholder="Send a message with Enter")
        with gr.Column(scale=0.15, min_width=0):
            submit_btn = gr.Button("Send")
    clear = gr.Button("Clear")
    disp_prompt = gr.Checkbox(label='Debug: Display Prompt')
    
    return chatbot, msg, clear, disp_prompt, submit_btn


def main_interface(params, llm, embeddings):
    #Page layout
    with gr.Blocks(css="footer {visibility: hidden}", title="APS ChatBot") as demo:

        #Header
        gr.Markdown("""
        ## Hi! I am CALMS, the APS' AI Assistant
        I was trained at Meta, taught to follow instructions at Stanford and am now learning about the APS. AMA!
        """
        )
        gr.Markdown("""
        * Use the General Chat to AMA. E.g. write some code for you, create a recipe from ingredients etc. 
        * Use the APS Q&A to ask me questions specific to the APS, I will look up answers using the documentation my trainers have provided me. 
        * Use the Document Q&A to ask me questions about a document you provide.
        """)

        #General chat tab
        with gr.Tab("General Chat"):
            chatbot, msg, clear, disp_prompt, submit_btn = init_chat_layout() #Init layout

            chat_general = Chat(llm, embeddings, doc_store=None)

            msg.submit(chat_general.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_general.generate_response, [chatbot, disp_prompt], chatbot #Use bot without context
            )
            submit_btn.click(chat_general.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_general.generate_response, [chatbot, disp_prompt], chatbot #Use bot without context
            )
            clear.click(lambda: chat_general.memory.clear(), None, chatbot, queue=False)

        #APS Q&A tab
        with gr.Tab("APS Q&A"):
            chatbot, msg, clear, disp_prompt, submit_btn = init_chat_layout() #Init layout

            aps_qa_docstore = init_aps_qa(embeddings, params)
            chat_qa = Chat(llm, embeddings, doc_store=aps_qa_docstore)

            #Pass an empty string to context when don't want domain specific context
            msg.submit(chat_qa.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_qa.generate_response, [chatbot, disp_prompt], chatbot #Use bot with context
            )
            submit_btn.click(chat_qa.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_qa.generate_response, [chatbot, disp_prompt], chatbot #Use bot with context
            )
        
            clear.click(lambda: chat_qa.memory.clear(), None, chatbot, queue=False)

        #Document Q&A tab
        with gr.Tab("Document Q&A"):
            gr.Markdown("""
            """
            )

            title = """
            <div style="text-align: center;max-width: 700px;">
                <h1>Chat with PDF</h1>
                <p style="text-align: center;">Upload one or more PDFs from your computer, click the "Load PDFs" button, <br />
                when everything is ready, you can start asking questions about the pdf</p>
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
                    load_pdf = gr.Button("Load PDF")
            
            chatbot, msg, clear, disp_prompt, submit_btn = init_chat_layout() #Init layout

            chat_pdf = PDFChat(llm, embeddings, doc_store=None)

            load_pdf.click(chat_pdf.update_pdf_docstore, inputs=[pdf_doc], outputs=[langchain_status], queue=False)
            msg.submit(chat_pdf.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_pdf.generate_response, [chatbot, disp_prompt], chatbot #Use bot with context
            )
            submit_btn.click(chat_pdf.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_pdf.generate_response, [chatbot, disp_prompt], chatbot #Use bot with context
            )
            clear.click(lambda: chat_pdf.memory.clear(), None, chatbot, queue=False)
    
        with gr.Tab("Tips & Tricks"):
            gr.Markdown("""
            1. I am not as powerful as GPT-4 or ChatGPT and I am running on cheap GPUs, if I get stuck, you can type "please continue" or similar and I will attempt to complete my thought.
            2. If I don't give a satisfactory answer, try rephrasing your question. For e.g. 'Can I do high energy diffraction at the APS?' instead of 'Where can I do high energy diffraction at the APS?
            3. Avoid using acronyms, e.g. say coherent diffraction imaging instead of CDI.
            4. CALMS is an acronym for Context-Aware Language Model for Science. 
                        
            """
            )

        #Footer
        gr.Markdown("""
        ##### Made with ❤️ for 🧑‍🔬 by:
        Mathew J. Cherukara, Michael Prince @ APS<br>
        Henry Chan, Aikaterini Vriza, Tao Zhou @ CNM<br>
        Varuni K. Sastry @ DSL/ALCF
        """
        )
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=2023)


if __name__ == '__main__':
    llm, embeddings = init_llm(params)
    main_interface(params, llm, embeddings)


