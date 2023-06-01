#Load model and tokenizer
model_name = "eachadea/vicuna-13b-1.1"
tokenizer_path = "./tokenizer/"

chunk_size = 384 #Size of chunks to break the text store into
chunk_overlap = 32 #How much overlap between chunks
seq_length = 2048 #LLM sequence length (Vicuna max is 2048)

#Which embedding model to use?
embedding_model_name = "all-mpnet-base-v2" #Highest scoring all-round, does 2800 sentences/s
#embedding_model_name = "all-MiniLM-L6-v2" #93% of best score, 5X faster
init_docs = True #Recompute embeddings?
N_hits = 3 #How many hits of context to provide?