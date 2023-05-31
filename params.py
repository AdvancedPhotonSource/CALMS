#Load model and tokenizer
model_name = "eachadea/vicuna-13b-1.1"
tokenizer_path = "./tokenizer/"

chunk_size = 1024 #Size of chunks to break the text store into
chunk_overlap = 128 #How much overlap between chunks
seq_length = 2048 #LLM sequence length (Vicuna max is 2048)
embedding_model_name = "all-mpnet-base-v2" #Which embedding model to use?
init_docs = False #Recompute embeddings?