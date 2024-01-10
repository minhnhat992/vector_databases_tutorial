"""
Spacy is a NLP package, we wil use that to create a NLP model
https://spacy.io/models

Import the language model, and calculate cosine between embeddings, to determine how similar different words are
"""

# need to run load model first. This is a medium size English language model
# in terminal : python -m spacy download en_core_web_md
import spacy

from utils.cosine_similarity import compute_cosine_similarity

nlp = spacy.load("en_core_web_md")

# a vector for the word dog
dog_embedding = nlp.vocab["dog"].vector
dog_embedding[0:10]

cat_embedding = nlp.vocab["cat"].vector
apple_embedding = nlp.vocab["apple"].vector
tasty_embedding = nlp.vocab["tasty"].vector
delicious_embedding = nlp.vocab["delicious"].vector
truck_embedding = nlp.vocab["truck"].vector

compute_cosine_similarity(dog_embedding, cat_embedding)
compute_cosine_similarity(apple_embedding, tasty_embedding)
compute_cosine_similarity(apple_embedding, truck_embedding)
compute_cosine_similarity(delicious_embedding, tasty_embedding)
