from langchain_experimental.text_splitter import SemanticChunker
from youtube_transcript_api import YouTubeTranscriptApi
from Embedding import get_embedding_model
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from Augmention import Augmention
load_dotenv()

video_id = input("Enter your url ")
video_id = video_id.split("v=")[-1].split("&")[0]

ytt_api = YouTubeTranscriptApi()
script = ytt_api.fetch(video_id, languages=['hi', 'en'])

full_text = " ".join([item.text for item in script])

embedding_model = get_embedding_model()

text_splitter = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.5
)

chunks = text_splitter.create_documents([full_text])
print(len(chunks))
vector_store = FAISS.from_documents(chunks, embedding_model)
print(vector_store.index_to_docstore_id)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 7})

# after retriever.invoke(...)
docs = retriever.invoke("What is Plan of action in this video")

context = "\n\n".join([doc.page_content for doc in docs])

answer = Augmention(
    context=context,
    Question=input("Enter your question ")
)

print(answer)
