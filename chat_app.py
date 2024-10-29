import os, time, shutil, subprocess
import params

if params.set_visible_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = params.visible_devices

import llms, prompts, bot_tools

import torch

from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import gradio as gr
from gradio import ChatMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("Using %d GPUs" %torch.cuda.device_count())

#Cleanups
gr.close_all() #Close any existing open ports'

def clean_pdf_paths():
    if os.path.exists(params.pdf_path): #Remove any PDF embeddings
        shutil.rmtree(params.pdf_path, ignore_errors=True)
    if os.path.exists(params.pdf_text_path): #Remove any raw PDF text
        shutil.rmtree(params.pdf_text_path, ignore_errors=True)
    os.mkdir(params.pdf_text_path)


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
        temperature=0,
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
def get_model():
    return params.anl_llm_model

def change_model(model_id):
    params.anl_llm_model = model_id

class Chat():
    def __init__(self, llm, embedding, doc_store):
        self.llm = llm 
        self.embedding = embedding
        self.doc_store = doc_store
        self.is_PDF = False #Flag to use NER over right set of docs. Changed in update_pdf_docstore


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

        return conversation


    #Method to find text with highest likely context
    def _get_context(self, query, doc_store):

        # Context retrieval from embeddings
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

        #Context retrieval from NER
        ners = llms.ner_hits(query) #Get unique named entities of > some length from query
        ner_hits = []

        #Set path from where to get NER context hits
        if self.is_PDF:
            doc_path = params.pdf_text_path
            print("Getting NER hits from PDF context")
        else: 
            doc_path = params.doc_path_root
            clean_pdf_paths() #Make sure PDF folders are clean to avoid context leak
            print("Getting NER hits from facility context")

        for ner in ners: #Grep NEs from raw text
            try: 
                hit = subprocess.check_output("grep -r -i -h '%s' %s/" %(ner, doc_path), shell=True).decode()
                hits = hit.split("\n") #split all the grep results into indiv strings
                ner_hits.extend(hits)
            except subprocess.CalledProcessError as err:
                if err.returncode > 1:
                    print ("No hits found for: ", ner) 
                    continue
                #Exit values: 0 One or more lines were selected. 1 No lines were selected. >1 An error occurred.
        #print ("NERs", ner_hits)

        ner_hits.sort(key=len, reverse=True) #Sort by length of hits
        #print ("Sorted NERs", ner_hits)

        for i in range(min(params.N_NER_hits, len(ner_hits))):
            print ("Selected NER hit %d : " %i, ner_hits[i])
            context += ner_hits[i]

        return context
    
    
    def generate_response(self, history, debug_output, convo_state, doc_state = None):
        user_message = history[-1]['content'] #History is list of tuple list. E.g. : [['Hi', 'Test'], ['Hello again', '']]
        all_user_messages = [x['content'] for x in history]

        if convo_state is None:
            convo_state = self._init_chain()

        if self.doc_store is not None:
            context = ""
            for message in all_user_messages:
             context += self._get_context(message, self.doc_store)
        elif doc_state is not None:
            context = ""
            for message in all_user_messages:
                context += self._get_context(message, doc_state)
        else:
            context = ""

        if debug_output:
            inputs = convo_state.prep_inputs({'input': user_message, 'context':context})
            prompt = convo_state.prep_prompts([inputs])[0][0].text

        bot_message = convo_state.predict(input=user_message, context=context)
        

        if debug_output:
            bot_message = f'---Prompt---\n\n {prompt} \n\n---Response---\n\n {bot_message}'

        print(history)
        print(convo_state)
        history.append(
            ChatMessage(role='assistant', content=bot_message)
        )
      
        return history, convo_state

    def add_message(self, user_message, history):
        history.append(
            ChatMessage(role='user', content=user_message)
        )
        return "", history
    
    def clear_memory(self, convo_state):
        if convo_state is not None:
            convo_state.memory.clear()
            return convo_state, None
        else:
            return None, None


class PDFChat(Chat):
    def update_pdf_docstore(self, pdf_docs, pdf_state):
        all_pdfs = []
        for pdf_doc in pdf_docs:
            loader = OnlinePDFLoader(pdf_doc.name)
            documents = loader.load()
            text_splitter = llms.init_text_splitter()
            texts = text_splitter.split_documents(documents)
            all_pdfs += texts
        llms.write_list(all_pdfs) #Write raw split text to file
        embed_path = params.pdf_path
        db = Chroma.from_documents(all_pdfs, self.embedding, #metadatas=[{"source": str(i)} for i in range(len(all_pdfs))],
            persist_directory=embed_path) #Compute embeddings over pdf using embedding model specified in params file

        return "PDF Ready", db
    

class ToolChat(Chat):
    """
    Implements an agentexector in a chat context. The agentexecutor is called in a fundimentally
    differnet way than the other chains, so custom implementaiton for much of the class.
    """
    def _init_chain(self):
        """
        tools = [
            dfrac_tools.DiffractometerAIO(params.spec_init)   
        ]
        """
        # TODO: CHANGE CREATION TYPE
        tools = [bot_tools.lattice_tool, bot_tools.diffractometer_tool]

        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=6)
        agent = create_json_chat_agent(
                                       tools=tools, 
                                       llm=self.llm,
                                       prompt=prompts.json_tool_prompt)

        agent_executor = AgentExecutor(
            agent=agent, tools=tools, handle_parsing_errors=True,
            max_iterations = 15,
            verbose=True
        )

        self.memory = memory
        self.conversation = agent_executor

        return memory, agent_executor
    
    def generate_response(self, history, debug_output):
        user_message = history[-1]['content'] #History is list of tuple list. E.g. : [['Hi', 'Test'], ['Hello again', '']]

        # Convert to langchain history
        lang_hist = []
        for message in history:
            if message['role'] == 'user':
                lang_hist.append(HumanMessage(content=message['content']))
            elif message['role'] == 'assistant':
                lang_hist.append(AIMessage(content=message['content']))
            else:
                raise ValueError(f'Unknown role in history {history}, {message['role']}. Add way to reolve.')

        # TODO: Implement debug output for langchain agents. Might have to use a callback?
        print(f'User input: {user_message}')
        response = self.conversation.invoke(
            {
                "input": user_message,
                "chat_history": lang_hist,
            }
        )

        bot_message = response['output']
        #Pass user message and get context and pass to model
        history.append(
            ChatMessage(role='assistant', content=bot_message)
        )

        return history
       

class S26ExecChat(ToolChat):
    """
    Implements an agentexector in a chat context. The agentexecutor is called in a fundimentally
    differnet way than the other chains, so custom implementaiton for much of the class.
    """
    def _init_chain(self):
        """
        tools = [
            dfrac_tools.DiffractometerAIO(params.spec_init)   
        ]
        """

        tools = [bot_tools.exec_cmd_tool] #, bot_tools.wolfram_tool

        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=6)
        agent = create_json_chat_agent(
                                       tools=tools, 
                                       llm=self.llm,
                                       prompt=prompts.json_tool_prompt)

        agent_executor = AgentExecutor(
            agent=agent, tools=tools, handle_parsing_errors=True,
            max_iterations = 15,
            verbose=True
        )
        
        self.memory = memory
        self.conversation = agent_executor

        return memory, agent_executor
    
        


class PolybotExecChat(ToolChat):
    def _init_chain(self):
        tools = [bot_tools.exec_polybot_tool, bot_tools.exec_polybot_lint_tool]

        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=7)

        agent = create_json_chat_agent(
                                       tools=tools, 
                                       llm=self.llm,
                                       prompt=prompts.json_tool_prompt)

        agent_executor = AgentExecutor(
            agent=agent, tools=tools, handle_parsing_errors=True,
            max_iterations = 15,
            verbose=True
        )
        
        self.memory = memory
        self.conversation = agent_executor
        
        return memory, agent_executor
    

"""
===========================
UI/Frontend
===========================
"""
def init_chat_layout():
    chatbot = gr.Chatbot(show_label=False, elem_id="chatbot", type='messages',
                         show_copy_button=True)#.style(height="500")
    with gr.Row():
        with gr.Column(scale=8):
            msg = gr.Textbox(show_label = False,
                placeholder="Send a message with Enter")
        with gr.Column(scale=2, min_width=0):
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
        """
        )
        gr.Markdown("""
        * Use the General Chat to AMA. E.g. write some code for you, create a recipe from ingredients etc. 
        * Use the Facility Q&A to ask me questions specific to the DOE facilities, I will look up answers using the documentation my trainers have provided me. 
        * Use the Document Q&A to ask me questions about a document you provide.
        """)

        if llm_type.huggingface:
            model_descr = f"local model: {params.model_name}"
        elif llm_type.openai:
            model_descr = f"ANL Hosted Model [{params.anl_llm_model}] (OpenAI)"
        else:
            model_descr = "Error! Unknown model"

        if embed_type.huggingface:
            embed_descr = f"Local model: {params.embedding_model_name}"
        elif embed_type.openai:
            embed_descr = f"ANL Hosted Model (OpenAI)"
        else:
            embed_descr = "Error! Unknown model"

        with gr.Row():
            openai_model_dd = gr.Dropdown(
                choices=['gpt35', 'gpt35large', 'gpt4', 'gpt4large', 'gpt4turbo', 'gpto1preview'],
                label='openai_model', 
                value=get_model,
                interactive=True,
                scale=1
            )
            openai_model_dd.change(change_model, inputs=[openai_model_dd])

            gr.Markdown('', scale=5)
        
        gr.Markdown(f"Context hits: {params.N_hits}\nNER hits: {params.N_NER_hits}")

        #General chat tab
        with gr.Tab("General Chat"):
            chatbot, msg, clear, disp_prompt, submit_btn = init_chat_layout() #Init layout

            chat_general = Chat(llm, embeddings, doc_store=None)
            chat_general_state = gr.State(None)

            msg.submit(chat_general.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_general.generate_response, [chatbot, disp_prompt, chat_general_state], [chatbot, chat_general_state] #Use bot without context
            )
            submit_btn.click(chat_general.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_general.generate_response, [chatbot, disp_prompt, chat_general_state], [chatbot, chat_general_state] #Use bot without context
            )
            clear.click(chat_general.clear_memory, [chat_general_state], [chat_general_state, chatbot])

        #APS Q&A tab
        with gr.Tab("Facility Q&A"):
            chatbot, msg, clear, disp_prompt, submit_btn = init_chat_layout() #Init layout

            facility_qa_docstore = llms.init_facility_qa(embeddings, params)
            chat_qa = Chat(llm, embeddings, doc_store=facility_qa_docstore)
            chat_qa_state = gr.State(None)

            #Pass an empty string to context when don't want domain specific context
            msg.submit(chat_qa.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_qa.generate_response, [chatbot, disp_prompt, chat_qa_state], [chatbot, chat_qa_state] #Use bot with context
            )
            submit_btn.click(chat_qa.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_qa.generate_response, [chatbot, disp_prompt, chat_qa_state], [chatbot, chat_qa_state] #Use bot with context
            )
            clear.click(chat_qa.clear_memory, [chat_qa_state], [chat_qa_state, chatbot])

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
                pdf_doc = gr.File(label="Load PDFs", file_types=['.pdf'], type="filepath", file_count = 'multiple')
                with gr.Row():
                    langchain_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                    load_pdf = gr.Button("Load PDF")
            
            chatbot, msg, clear, disp_prompt, submit_btn = init_chat_layout() #Init layout

            chat_pdf = PDFChat(llm, embeddings, doc_store=None)
            chat_pdf_state = gr.State(None)
            pdf_store_state = gr.State(None)

            load_pdf.click(chat_pdf.update_pdf_docstore, inputs=[pdf_doc, pdf_store_state], outputs=[langchain_status, pdf_store_state], queue=False)
            msg.submit(chat_pdf.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_pdf.generate_response, [chatbot, disp_prompt, chat_pdf_state, pdf_store_state], [chatbot, chat_pdf_state] #Use bot with context
            )
            submit_btn.click(chat_pdf.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_pdf.generate_response, [chatbot, disp_prompt, chat_pdf_state, pdf_store_state], [chatbot, chat_pdf_state] #Use bot with context
            )
            clear.click(chat_general.clear_memory, [chat_pdf_state], [chat_pdf_state, chatbot])
        
        with gr.Tab("S26 Agent"):
            chatbot, msg, clear, disp_prompt_tool, submit_btn = init_chat_layout() #Init layout

            tool_qa = S26ExecChat(llm, embeddings, None)
            tool_qa._init_chain()

            #Pass an empty string to context when don't want domain specific context
            msg.submit(tool_qa.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                tool_qa.generate_response, [chatbot, disp_prompt_tool], [chatbot] #Use bot with context
            )
            submit_btn.click(tool_qa.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                tool_qa.generate_response, [chatbot, disp_prompt_tool], [chatbot] #Use bot with context
            )
            clear.click(lambda: tool_qa.memory.clear(), None, chatbot, queue=False)

        with gr.Tab("Polybot Exec"):
            chatbot, msg, clear, disp_prompt_tool, submit_btn = init_chat_layout() #Init layout

            polybot_exec = PolybotExecChat(llm, embeddings, None)
            polybot_exec._init_chain()

            #Pass an empty string to context when don't want domain specific context
            msg.submit(polybot_exec.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                polybot_exec.generate_response, [chatbot, disp_prompt_tool], chatbot #Use bot with context
            )
            submit_btn.click(polybot_exec.add_message, [msg, chatbot], [msg, chatbot], queue=False).then(
                polybot_exec.generate_response, [chatbot, disp_prompt_tool], chatbot #Use bot with context
            )
        
            clear.click(lambda: polybot_exec.memory.clear(), None, chatbot, queue=False)


    
        with gr.Tab("Tips & Tricks"):
            gr.Markdown("""
            1. If I don't give a satisfactory answer, try rephrasing your question. For e.g. 'Can I do high energy diffraction at the APS?' instead of 'Where can I do high energy diffraction at the APS?
            2. Avoid using acronyms, e.g. say coherent diffraction imaging instead of CDI.
            3. CALMS is an acronym for Context-Aware Language Model for Science. 
                        
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
    demo.launch(server_name="0.0.0.0", server_port=params.port)


if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True) #One of mutuallu exclusive args is required
    group.add_argument('-hf', '--huggingface', action='store_true', help='Open-source model')
    group.add_argument('-o', '--openai', action='store_true', help='OpenAI Model')

    llm_type = parser.parse_args()
    print(llm_type)
    
    if llm_type.openai:
        llm = llms.AnlLLM(params)
        if params.anl_user == "":
            print("The ANL OpenAI API user parameter (anl_user) must be set in params.py! Exiting.")
            exit()
    elif llm_type.huggingface:
        llm = init_local_llm(params)
    else:
        raise AssertionError("LLM type must be huggingface or openai")

    #Embedding model parameters
    embed_type = llm_type # Can be different from llm_type

    #Embedding paths
    if embed_type.huggingface:
        params.embed_path = '%s/%s' %(params.base_path, params.embedding_model_name)
        embeddings = init_local_embeddings(params)

    elif embed_type.openai:
        if params.init_docs:
            input('WARNING: WILL INIT ALL DOCS WITH OPENAI EMBEDS. Press enter to continue')
        params.embed_path = f"{params.base_path}/anl_openai"
        embeddings = llms.ANLEmbeddingModel(params)
    
    params.pdf_path = '%s/pdf' %params.embed_path
    clean_pdf_paths() #Clear any PDF embeds and NER text

    main_interface(params, llm, embeddings)


