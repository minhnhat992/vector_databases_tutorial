"""Use openai to go through vector database to anwser question."""

import os

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

from utils.car_etl import prepare_car_reviews_data
from utils.chroma_utils import build_chroma_collection

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_PATH = "data/archive/*"
CHROMA_PATH = "car_review_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "car_reviews"

# etl
chroma_car_reviews_dict = prepare_car_reviews_data(DATA_PATH)

# build collection
build_chroma_collection(
    chroma_path=CHROMA_PATH,
    collection_name=COLLECTION_NAME,
    embedding_func_name=EMBEDDING_FUNC_NAME,
    ids=chroma_car_reviews_dict["ids"],
    metadatas=chroma_car_reviews_dict["metadatas"],
    documents=chroma_car_reviews_dict["documents"],
)

# setup open ai
with open("data/creds/open_ai_secret.txt") as e:
    key = e.read()

client_open_ai = OpenAI(api_key=key)

# setup vector database collection
client_vector = chromadb.PersistentClient(CHROMA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_FUNC_NAME
)

collection = client_vector.get_collection(
    name=COLLECTION_NAME, embedding_function=embedding_func
)

# set up to use open ai query
context = """
 You are a customer success employee at a large
  car dealership. Use the following car reviews
  to answer questions: {}
 """

question = """
 What does the highest rated review  tell us about our dealship
 """
# pull from vector database good reviews
good_reviews = collection.query(
    query_texts=[question],
    n_results=10,
    include=["documents"],
    where={"Rating": {"$gte": 3}},
)

# prepare reviews format to use as context when asking open ai
reviews_str = ",".join(good_reviews["documents"][0])

good_review_summaries = client_open_ai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": context.format(reviews_str)},
        {"role": "user", "content": question},
    ],
    temperature=0,
    n=1,
)
good_review_summaries
