# Env settings
set_visible_devices = True
visible_devices = '2,3'

#Load model and tokenizer
#model_name = "eachadea/vicuna-13b-1.1"
model_name = "lmsys/vicuna-13b-v1.3"
#model_name = "lmsys/vicuna-33b-v1.3"
tokenizer_path = "./tokenizer/"

chunk_size = 384 #Size of chunks to break the text store into
chunk_overlap = 32 #How much overlap between chunks
seq_length = 2048 #LLM sequence length (Vicuna max is 2048)

#Which embedding model to use?
embedding_model_name =   "all-mpnet-base-v2" #Highest scoring all-round, does 2800 sentences/s
#embedding_model_name = "all-MiniLM-L6-v2" #93% of best score, 5X faster

init_docs = False #Recompute embeddings?
overwrite_embeddings = True #Overwrite embeddings if already exist? -- will raise val error of init_docs is True and this is not
N_hits = 3 #How many hits of context to provide?
