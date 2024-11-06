#UI Params
port = 2025

# Env settings -- used for local model (hf)
set_visible_devices = True
visible_devices = '1'

#LLM parameters

# HuggingFace params
model_name = "lmsys/vicuna-13b-v1.5-16k"
tokenizer_path = "./tokenizer/%s" %model_name.split("/")[1]

seq_lengths = {"lmsys/vicuna-13b-v1.3" : 2048,
               "lmsys/vicuna-13b-v1.5-16k" : 16384}
seq_length = seq_lengths[model_name]

# OpenAI params
anl_llm_url_path = 'keys/ANL_LLM_URL'
anl_llm_debug = True
anl_llm_debug_fp = 'anl_outputs.log'
anl_user = "avriza"
# One of: gpt35, gpt35large, gpt4, gpt4large, gpt4turbo
anl_llm_model = 'gpt4turbo' 

anl_embed_url_path = 'keys/ANL_EMBED_URL'

embedding_model_name =   "all-mpnet-base-v2" #Highest scoring all-round, does 2800 sentences/s
chunk_size = 1024 #Size of chunks to break the text store into
chunk_overlap = 128 #How much overlap between chunks

#Embedding params
base_path = 'embeds/' 
init_docs = False #Recompute embeddings?
overwrite_embeddings = True #Overwrite embeddings if already exist? -- will raise val error of init_docs is True and this is not

#NER params
N_hits = 4 #How many hits of context to provide?
similarity_cutoff = 1.4 #Ignore context hits greater than this distance away. Empirical number.
N_NER_hits = 2 #How many NER hits to provide
min_NER_length = 5 #Only consider entities > 5 characters

#List of folders to add to doc store
doc_path_root = "DOC_STORE"
doc_paths = ["%s/APS-Science-Highlight" %doc_path_root, 
             "%s/APS-Docs" %doc_path_root, 
             "%s/ALCF-Docs" %doc_path_root,
             "%s/AIT-Docs" %doc_path_root,
             "%s/CNM-Docs" %doc_path_root,
             "%s/CNM-Science-Highlight" %doc_path_root
            ]
pdf_text_path = "%s/PDFs"  %doc_path_root#Store raw text from PDF for NER

#Spec Params
spec_init = True

# Tool Params
use_wolfram = False