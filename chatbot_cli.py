import params
import llms
import bot_tools
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.messages import AIMessage, HumanMessage
import prompts

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
                raise ValueError(f"Unknown role in history {history}, {message['role']}. Add way to resolve.")

                #raise ValueError(f'Unknown role in history {history}, {message['role']}. Add way to resolve.')

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
            {'role':'assistant', 'content':bot_message}
        )

        return history
       
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

llm = llms.AnlLLM(params)
embeddings = llms.ANLEmbeddingModel(params)


polybot_exec = PolybotExecChat(llm, embeddings, None)
polybot_exec._init_chain()


chat_history = [{'content': 'Pick up the vial in rack 1', 'role': 'user'}]
#Pass an empty string to context when don't want domain specific context
chat_history = polybot_exec.generate_response(chat_history, True)

print(chat_history[-1])

