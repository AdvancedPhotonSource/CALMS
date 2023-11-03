from langchain.embeddings import HuggingFaceInstructEmbeddings

model_name = "hkunlp/instructor-xl"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceInstructEmbeddings(
model_name=model_name,
model_kwargs=model_kwargs,
encode_kwargs=encode_kwargs
)
