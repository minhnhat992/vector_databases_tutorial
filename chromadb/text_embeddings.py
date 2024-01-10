"""Using sentence-transformeds, we perform text embeddings."""
from sentence_transformers import SentenceTransformer

from cosine_similarity import compute_cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "The canine barked loudly.",
    "The dog made a noisy bark.",
    "He ate a lot of pizza.",
    "He devoured a large quantity of pizza pie.",
]

text_embeddings = model.encode(texts)

# Notice that text_embeddings is a NumPy array with the shape (4, 384), which means that it has 4 rows and 384
# columns. This is because you encoded 4 texts, and "all-MiniLM-L6-v2" generates 384-dimensional embeddings.
text_embeddings.shape

text_embeddings_dict = dict(zip(texts, list(text_embeddings)))
dog_text_1 = "The canine barked loudly."
dog_text_2 = "The dog made a noisy bark."

# calculate how similar the fistt 2 texts are
compute_cosine_similarity(
    text_embeddings_dict[dog_text_1], text_embeddings_dict[dog_text_2]
)

pizza_text_1 = "He ate a lot of pizza."
pizza_test_2 = "He devoured a large quantity of pizza pie."
compute_cosine_similarity(
    text_embeddings_dict[pizza_text_1], text_embeddings_dict[pizza_test_2]
)

compute_cosine_similarity(
    text_embeddings_dict[dog_text_1], text_embeddings_dict[pizza_text_1]
)
