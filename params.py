def check_model_type(model):
    if model != 'hf' and model !='anl':
        raise AssertionError("LLM type must be hf or anl")

# Env settings -- used for local model (hf)
set_visible_devices = True
visible_devices = '1'

#LLM parameters

#Options: 'hf' (local huggingface), 'anl' (anl-hosted LLM)
llm_type = 'hf'
check_model_type(llm_type)

# hf params
model_name = "lmsys/vicuna-13b-v1.5-16k"
tokenizer_path = "./tokenizer/%s" %model_name.split("/")[1]

seq_lengths = {"lmsys/vicuna-13b-v1.3" : 2048,
               "lmsys/vicuna-13b-v1.5-16k" : 16384}
seq_length = seq_lengths[model_name]

# anl params
anl_llm_url_path = 'keys/ANL_LLM_URL'
anl_llm_debug = True
anl_llm_debug_fp = 'anl_outputs.log'


#Embedding model parameters

#Options: 'hf' (local huggingface), 'anl' (anl-hosted LLM)
embed_type = llm_type # Can be different from llm_type
check_model_type(embed_type)
anl_embed_url_path = 'keys/ANL_EMBED_URL'

embedding_model_name =   "all-mpnet-base-v2" #Highest scoring all-round, does 2800 sentences/s
chunk_size = 1024 #Size of chunks to break the text store into
chunk_overlap = 128 #How much overlap between chunks

init_docs = False #Recompute embeddings?
if embed_type == 'anl' and init_docs:
    input('WARNING: WILL INIT ALL DOCS WITH OPENAI EMBEDS. Press enter to continue')

overwrite_embeddings = True #Overwrite embeddings if already exist? -- will raise val error of init_docs is True and this is not
N_hits = 4 #How many hits of context to provide?
similarity_cutoff = 1.4 #Ignore context hits greater than this distance away. Empirical number.
N_NER_hits = 2 #How many NER hits to provide
min_NER_length = 5 #Only consider entities > 5 characters

#List of folders to add to doc store
doc_paths = ["DOC_STORE/APS-Science-Highlight", 
             "DOC_STORE/APS-Docs", 
             "DOC_STORE/ALCF-Docs",
             "DOC_STORE/AIT-Docs",
             "DOC_STORE/CNM-Docs",
             "DOC_STORE/CNM-Science-Highlight"
            ]
pdf_text_path = "DOC_STORE/PDFs" #Store raw text from PDF for NER


#Embedding paths
base_path = 'embeds/' 
if embed_type == 'hf':
    embed_path = '%s/%s' %(base_path, embedding_model_name)
elif embed_type == 'anl':
    embed_path = f"{base_path}/anl_openai"
pdf_path = '%s/pdf' %embed_path

#Spec Params
spec_init = True

#Web UI port
if llm_type == 'hf':
    port = 2023
else:
    port = 2024
