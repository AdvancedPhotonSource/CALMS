import os, time, shutil
import params, dfrac_tools, llms

if params.set_visible_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = params.visible_devices

import torch

from langchain import PromptTemplate, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.document_loaders import OnlinePDFLoader
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings 

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("Using %d GPUs" %torch.cuda.device_count())

#Cleanups
gr.close_all() #Close any existing open ports
if os.path.exists(params.pdf_path):
    shutil.rmtree(params.pdf_path) 


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

#Setup embedding model
def init_local_embeddings(params):
    return HuggingFaceEmbeddings(model_name=params.embedding_model_name)



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
        print ("Context hits found", len(docs))
        for i in range(min(params.N_hits, len(docs))):
            if docs[i][1]<params.similarity_cutoff:
                context += docs[i][0].page_content +"\n"
                print (i+1, len(docs[i][0].page_content), docs[i][1], docs[i][0].page_content)
            else:
                print ("\n\nIGNORING CONTENT of score %.2f" %docs[i][1],len(docs[i][0].page_content), docs[i][0].page_content)
                
        return context
    
    
    def generate_response(self, history, debug_output):
        user_message = history[-1][0] #History is list of tuple list. E.g. : [['Hi', 'Test'], ['Hello again', '']]
        all_user_messages = [x[0] for x in history]
        print(all_user_messages)

        if self.doc_store is None:
            context = ""
        else:
            context = ""
            for message in all_user_messages:
             context += self._get_context(message, self.doc_store)

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
            #time.sleep(0.02)
            #yield history
        return history

    def add_message(self, user_message, history):
        return "", history + [[user_message, None]]


class PDFChat(Chat):
    def update_pdf_docstore(self, pdf_docs):
        all_pdfs = []
        for pdf_doc in pdf_docs:
            loader = OnlinePDFLoader(pdf_doc.name)
            documents = loader.load()
            text_splitter = llms.init_text_splitter()
            texts = text_splitter.split_documents(documents)
            all_pdfs += texts
        embed_path = params.pdf_path
        db = Chroma.from_documents(all_pdfs, self.embedding, #metadatas=[{"source": str(i)} for i in range(len(all_pdfs))],
            persist_directory=embed_path) #Compute embeddings over pdf using embedding model specified in params file
        db.persist()

        self.doc_store = db

        return "PDF Ready"
    

class ToolChat(Chat):
    """
    Implements an agentexector in a chat context. The agentexecutor is called in a fundimentally
    differnet way than the other chains, so custom implementaiton for much of the class.
    """
    def _init_chain(self):
        tools = [
            dfrac_tools.DiffractometerAIO(params.spec_init)   
        ]

        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=6)
        conversation = initialize_agent(tools, 
                                       self.llm, 
                                       agent='conversational-react-description', 
                                       verbose=True, 
                                       handle_parsing_errors='Check your output and make sure it conforms!',
                                       max_iterations=5,
                                       memory=memory)
        return memory, conversation
    
    def generate_response(self, history, debug_output):
        user_message = history[-1][0] #History is list of tuple list. E.g. : [['Hi', 'Test'], ['Hello again', '']]

        # TODO: Implement debug output for langchain agents. Might have to use a callback?
        print(f'User input: {user_message}')
        bot_message = self.conversation.run(user_message)
        #Pass user message and get context and pass to model
        history[-1][1] = "" #Replaces None with empty string -- Gradio code

        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.02)
            yield history



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
        ## Hi! I am CALMS, a Scientific AI Assistant
        #### I was trained at Meta, taught to follow instructions at Stanford and am now learning about the DOE User Facilities. AMA!
        """
        )
        gr.Markdown("""
        * Use the General Chat to AMA. E.g. write some code for you, create a recipe from ingredients etc. 
        * Use the Facility Q&A to ask me questions specific to the DOE facilities, I will look up answers using the documentation my trainers have provided me. 
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
        with gr.Tab("Facility Q&A"):
            chatbot, msg, clear, disp_prompt, submit_btn = init_chat_layout() #Init layout

            facility_qa_docstore = llms.init_facility_qa(embeddings, params)
            chat_qa = Chat(llm, embeddings, doc_store=facility_qa_docstore)

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
        
        with gr.Tab("Tool Agent"):
            chatbot, msg, clear, disp_prompt_tool, submit_btn = init_chat_layout() #Init layout

            tool_qa = ToolChat(llm, embeddings, None)

            #Pass an empty string to context when don't want domain specific context
            msg.submit(tool_qa.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                tool_qa.generate_response, [chatbot, disp_prompt_tool], chatbot #Use bot with context
            )
            submit_btn.click(tool_qa.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                tool_qa.generate_response, [chatbot, disp_prompt_tool], chatbot #Use bot with context
            )
        
            clear.click(lambda: tool_qa.memory.clear(), None, chatbot, queue=False)

    
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
    if params.llm_type == 'anl':
        llm = llms.AnlLLM(params)
    elif params.llm_type == 'hf':
        llm = init_local_llm(params)
    
    embeddings = init_local_embeddings(params)
    main_interface(params, llm, embeddings)


