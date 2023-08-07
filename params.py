# Env settings
set_visible_devices = True
visible_devices = '2,3'

#LLM parameters
#model_name = "eachadea/vicuna-13b-1.1"
model_name = "lmsys/vicuna-13b-v1.3"
tokenizer_path = "./tokenizer/"
seq_length = 2048 #LLM sequence length (Llama 2 max is 4k, Vicuna : 2048)


#Embedding model parameters
embedding_model_name =   "all-mpnet-base-v2" #Highest scoring all-round, does 2800 sentences/s
#embedding_model_name = "all-MiniLM-L6-v2" #93% of best score, 5X faster
chunk_size = 384 #Size of chunks to break the text store into
chunk_overlap = 64 #How much overlap between chunks

init_docs = False #Recompute embeddings?
overwrite_embeddings = True #Overwrite embeddings if already exist? -- will raise val error of init_docs is True and this is not
N_hits = 3 #How many hits of context to provide?

#List of folders to add to doc store
doc_paths = ["DOC_STORE/APS-Science-Highlight", "DOC_STORE/APS-Docs"]
#Embedding paths
pdf_path = 'embeds/pdf'
