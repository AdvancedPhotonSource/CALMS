import gradio as gr
import os
import json
from typing import List, Tuple
from sdl_agents import AutoGenSystem
import autogen
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False" # to avoid the timed out
from queue import Queue
import threading
from time import sleep

# Global variables to manage human input flow
human_input_queue = Queue()
human_input_response = None
human_input_event = threading.Event()
CHECK_UPDATE_INTERVAL = 1 # seconds
uploaded_pdfs = []
require_human_input = False

# Variable for webcam live streaming
default_video_url = "http://127.0.0.1:8080/"

# GroupChatManager to capture messages
class CaptureGroupChatManager(autogen.GroupChatManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_messages = []
        
    def receive(self, message, sender, request_reply=None, silent=False):
        if not silent:
            agent_name = getattr(sender, "name", "Unknown")
            if isinstance(message, dict) and "content" in message:
                content = message["content"]
            else:
                content = str(message)
                
            if content:  
                self.captured_messages.append(f"{agent_name}: {content}")
        
        return super().receive(message, sender, request_reply, silent)

# Initialize AutoGen system
workdir = "polybot_exec_run"
os.makedirs(workdir, exist_ok=True)
polybot_file_path = 'utils/n9_robot_operation_commands.py'
llm_type = "gpt4o"  

# Create the AutoGen system
autogen_system = None 

# Store conversation history
conversation_history = []

def ensure_autogen_system():
    """Make sure the AutoGen system is initialized"""
    global autogen_system
    global require_human_input

    if autogen_system is not None:
        autogen_system.code_writer_agent.human_input_mode = "ALWAYS" if require_human_input else "NEVER"
        autogen_system.polybot_admin.human_input_mode = "ALWAYS" if require_human_input else "NEVER"
        return
    
    autogen_system = AutoGenSystem(
            llm_type=llm_type,
            workdir=workdir,
            polybot_file_path=polybot_file_path
        )
        
    # Set all agents to NEVER when we do not ask for human input
    autogen_system.code_writer_agent.human_input_mode = "ALWAYS" if require_human_input else "NEVER"
    autogen_system.code_review_agent.human_input_mode = "NEVER"
    autogen_system.scraper_agent.human_input_mode = "NEVER"
    autogen_system.polybot_admin.human_input_mode = "ALWAYS" if require_human_input else "NEVER"
    
    autogen_system.polybot_admin.get_human_input = lambda prompt: custom_get_human_input(prompt, "PolybotAdmin")
    autogen_system.code_writer_agent.get_human_input = lambda prompt: custom_get_human_input(prompt, "CodeWriter")

    # Replace the manager with our capturing version
    autogen_system.manager = CaptureGroupChatManager(
        groupchat=autogen_system.groupchat, 
        llm_config=autogen_system.llm_config
    )

def toggle_human_input(value):
    """Toggle the human input requirement flag"""
    global require_human_input
    require_human_input = value
    
    # Update the autogen system if it exists
    if autogen_system is not None:
        autogen_system.code_writer_agent.human_input_mode = "ALWAYS" if require_human_input else "NEVER"
        autogen_system.polybot_admin.human_input_mode = "ALWAYS" if require_human_input else "NEVER"
    
    return f"Human input requirement: {'Enabled' if require_human_input else 'Disabled'}"

def submit_human_input(response):
    """Submit human input from the UI"""
    global human_input_response
    global human_input_event
    global human_input_queue
   
    if not human_input_queue.empty():
        human_input_response = response
        human_input_event.set()  # Notify the waiting thread


def custom_get_human_input(prompt, agent_name):
    """Custom function to get human input through the UI"""
    global human_input_response, human_input_event, human_input_queue
    human_input_response = None

    human_input_event.clear()
    human_input_queue.put((agent_name, prompt))
    human_input_event.wait()
    human_input_queue.get()  # Remove the request from the queue
    
    return human_input_response

def check_for_human_input_requests():
    """Check if there are any pending human input requests"""
    if not human_input_queue.empty():
        agent_name, prompt = human_input_queue.get()
        human_input_queue.put((agent_name, prompt))  # Put it back for processing
        if autogen_system is not None:
            last_message = autogen_system.manager.captured_messages[-1]
        else:
            last_message = ""
        return f"**Human input needed from {agent_name}:**\n\n{last_message}\n\n{prompt}"
    return f"[No Requests Currently]"

def process_message(message: str, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:        
    print('autogen system initiallization')
    ensure_autogen_system()
    
    autogen_system.manager.captured_messages = []
    
    updated_history = history.copy() 
    updated_history.append((message, "Processing your request..."))

    agent_message = message
    if uploaded_pdfs:
        pdf_context = "The following PDFs have been uploaded and are available in the working directory:\n"
        for i, pdf in enumerate(uploaded_pdfs):
            pdf_context += f"{i+1}. {pdf['filename']} (located at: {pdf['path']})\n"
        
        agent_message = f"{pdf_context}\n\nUser request: {message}"

    # Start the chat in a separate thread so it doesn't block
    def run_chat():
        try:
            autogen_system.initiate_chat(agent_message)

            # Get the conversation output
            response = "\n\n".join(autogen_system.manager.captured_messages)

        except Exception as e:
            response = f"Error: {str(e)}"
            print(f"Exception during chat: {e}")
        
        # Update the last message with the actual response
        nonlocal updated_history
        updated_history[-1] = (response) #message, 
        
    # Start the chat thread
    chat_thread = threading.Thread(target=run_chat)
    chat_thread.start()
    
    # Wait briefly for any immediate human input requests
    chat_thread.join()    
    response = "\n\n".join(autogen_system.manager.captured_messages)
    updated_history[-1] = (message, response)
    
    # Store in global history
    conversation_history.append((message, response))    
    return updated_history

def clear_history():
    """Clear the conversation history"""
    global conversation_history
    conversation_history = []
    return []

def upload_pdf(file_path):
    """Handle PDF file upload"""
    global uploaded_pdfs
    
    # Save the uploaded PDF file
    if file_path is not None and isinstance(file_path, list) and len(file_path) > 0:
        file_obj = file_path[0]
        
        if isinstance(file_obj, str):
            filename = os.path.basename(file_obj)
            save_path = os.path.join(workdir, filename)
            with open(file_obj, 'rb') as source_file:
                file_content = source_file.read()
            with open(save_path, 'wb') as f:
                f.write(file_content)
            
            # Add the PDF to our tracking list
            uploaded_pdfs.append({
                "filename": filename,
                "path": save_path
            })
            
            return f"PDF uploaded: {filename}"
        
        return f"Received file object of type: {type(file_obj)}. Please check console for more details."
    
    return "No file uploaded or empty file list"

def update_webcam_url(url):
    """Update the webcam stream URL"""
    global webcam_url
    webcam_url = url
    return f"Webcam URL updated to: {url}"

# Function to update the video HTML 
def update_video_html(enabled, url):
    if enabled and url:
        return f"""
        <div style="width: 100%; height: 420px; border: 1px solid #ccc; overflow: hidden; position: relative;">
            <iframe src="{url}" style="width: 100%; height: 100%; border: none;" 
                   allowfullscreen></iframe>
        </div>
        """
    else:
        return """
        <div style="width: 100%; height: 240px; border: 1px solid #ccc; 
             display: flex; align-items: center; justify-content: center; background-color: #f5f5f5;">
            <p style="text-align: center;">No available video currently</p>
        </div>
        """

# Gradio interface
with gr.Blocks(title="SDL Agent Chat") as demo:
    gr.Markdown("# SDL Agents Chat Interface")
    gr.Markdown("Upload a PDF file for context (optional) and start chatting with the SDL agents.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                elem_id="chatbot"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Enter your message here...",
                    show_label=False,
                    container=False
                )
                submit_btn = gr.Button("Submit", variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("Clear History", variant="secondary")

            # Place holder for human input option
            human_input_toggle = gr.Checkbox(
                label="Require Human Input", 
                value=False
            )
    
            def toggle_human_input(value):
                global require_human_input
                require_human_input = value
                return f"Human input mode: {'Enabled' if value else 'Disabled'}"

            human_input_status = gr.Textbox(label="Status", value="Human input mode: Disabled", interactive=False)
            human_input_toggle.change(toggle_human_input, inputs=human_input_toggle, outputs=human_input_status)

            # Then your existing human input monitor section
            human_response = gr.Textbox(label='Human Input Monitor. Requests will show here while autogen is running.', interactive=False)
            with gr.Row():
                hr_msg = gr.Textbox(
                    placeholder="Enter input...",
                    show_label=False,
                    container=False
                )
            hr_submit = gr.Button("Submit Human Input", variant="primary")
        
            timer = gr.Timer(1, active=True)
            timer.tick(fn=check_for_human_input_requests, inputs=None, outputs=human_response)

        # Column for PDF/video  
        with gr.Column(scale=2):
            # Upload a PDF component
            pdf_upload = gr.File(label="Load PDFs", file_types=['.pdf'], type="filepath", file_count = 'multiple')
            pdf_status = gr.Textbox(label="Upload Status", interactive=False)

            # Live Stream Video components
            gr.Markdown("### Live Video Stream")
            video_toggle = gr.Checkbox(label="Enable Video Stream", value=False)
            video_url = gr.Textbox(
                label="Video Stream URL",
                placeholder="Enter video stream URL",
                value=default_video_url
            )
            
            # HTML component for displaying the video
            video_html = gr.HTML(update_video_html(False, default_video_url))
            
            # Connect the toggle and URL inputs to update the HTML
            video_toggle.change(
                fn=update_video_html,
                inputs=[video_toggle, video_url],
                outputs=video_html
            )
            
            video_url.change(
                fn=update_video_html,
                inputs=[video_toggle, video_url],
                outputs=video_html
            )

    submit_btn.click(
        lambda message, history: history + [(message, None)],  
        inputs=[msg, chatbot], 
        outputs=[chatbot]
    ).then(
        process_message, 
        inputs=[msg, chatbot], 
        outputs=[chatbot]
    ).then(
        lambda: "", 
        None, 
        msg
    )

    clear_btn.click(
        clear_history,
        outputs=[chatbot]
    )
    
    pdf_upload.upload(
        upload_pdf,
        inputs=[pdf_upload],
        outputs=[pdf_status]
    )
    
    msg.submit(
        lambda message, history: history + [(message, None)],  # Immediately display user message
        inputs=[msg, chatbot], 
        outputs=[chatbot]
    ).then(
        process_message,  # Then process normally
        inputs=[msg, chatbot], 
        outputs=[chatbot]
    ).then(
        lambda: "", 
        None, 
        msg
    )

    hr_submit.click(submit_human_input, inputs=hr_msg)
    hr_msg.submit(submit_human_input, inputs=hr_msg)


# Launch the app
demo.launch(share=False, debug=True)  
