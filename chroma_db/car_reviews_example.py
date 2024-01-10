"""
Based on this example https://realpython.com/chromadb-vector-database/#practical-example-add-context-for-a-large-language-model-llm
"""
import chromadb
from chromadb.utils import embedding_functions

from utils.car_etl import prepare_car_reviews_data
from utils.chroma_utils import build_chroma_collection

DATA_PATH = "data/archive/*"
CHROMA_PATH = "car_reviews_embeddings"
EMBEDDING_FUNC_NAME = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "car_reviews"

# etl
chroma_car_reviews_dict = prepare_car_reviews_data(DATA_PATH)
chroma_car_reviews_dict.keys()

chroma_car_reviews_dict["ids"][-10]

print(chroma_car_reviews_dict["documents"][-10])

chroma_car_reviews_dict["metadatas"][-10]

# build collection
build_chroma_collection(
    chroma_path=CHROMA_PATH,
    collection_name=COLLECTION_NAME,
    embedding_func_name=EMBEDDING_FUNC_NAME,
    ids=chroma_car_reviews_dict["ids"],
    metadatas=chroma_car_reviews_dict["metadatas"],
    documents=chroma_car_reviews_dict["documents"],
)

# now run query
client = chromadb.PersistentClient(CHROMA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_FUNC_NAME
)
collection = client.get_collection(name=COLLECTION_NAME)

great_reviews = collection.query(
    query_texts=["find me bad reviews about Mazda car"],
    n_results=10,
    include=["documents", "distances", "metadatas"],
)

great_reviews["documents"][0]
