#EMBEDDINGS
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import os 
import tensorflow.compat.v2 as tf
from tensorflow_text import SentencepieceTokenizer

module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
model = hub.load(module_url)
print("module %s loaded" % module_url)

def embed(input):
    return model(input)

df = pd.read_csv("/app/data/processed_data.csv")
df_test = pd.read_csv("/app/data/processed_data_test.csv")

input_data = df["Lemmas"].fillna("").tolist() 
input_data_test = df_test["Lemmas"].fillna("").tolist()

tensor_inputs = tf.constant(input_data, dtype=tf.string)
tensor_inputs_test = tf.constant(input_data_test, dtype=tf.string)

embeddings = model(tensor_inputs).numpy()
embeddings_test = model(tensor_inputs_test).numpy()

df_embeddings = pd.DataFrame(embeddings)
df_embeddings_test = pd.DataFrame(embeddings_test)

print("Finished embeddings")

df_embeddings.to_csv("/app/data/embeddings.csv", index=False, header=False)
df_embeddings_test.to_csv("/app/data/embeddings_test.csv", index=False, header=False)

print("Saved embeddings")

df_cat = pd.read_csv("/app/data/processed_data_cat.csv")

categories_dict = df_cat.groupby("Sentence")["Lemmas"].apply(list).to_dict()

all_categories = []
all_embeddings = []
for category, keywords in categories_dict.items():

    keywords = [str(keyword) for keyword in keywords if keyword is not None]

    tensor_keywords = tf.constant(keywords, dtype=tf.string)
    keyword_embeddings = embed(tensor_keywords).numpy()

    mean_embedding = np.mean(keyword_embeddings.astype(float), axis=0).astype(float)
    
    all_categories.append(category)
    all_embeddings.append(mean_embedding)

df_keyword_embeddings = pd.DataFrame({
    "Category": all_categories,
    "Embeddings": all_embeddings
})

print("Finished keyword embeddings")

df_keyword_embeddings.to_csv("/app/data/embedded_keywords.csv", index=False)

print("Saved keyword embeddings")
