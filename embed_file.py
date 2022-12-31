import openai
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import pickle


def load_questions_answers(path, reverse_columns=False):
    if ".csv" in path:
        df = pd.read_csv(path).dropna()
    elif ".json" in path:
        df = pd.read_json(path_or_buf=path, lines=True)
    else:
      raise "Error, not supported file type"
    

    if "Embeddings" in df.columns:
      return df

    else:
        if not reverse_columns:
            df.rename(columns = {df.columns[0]:'Questions'}, inplace = True)
            df.rename(columns = {df.columns[1]:'Answers'}, inplace = True)
        else:
            df.rename(columns = {df.columns[1]:'Questions'}, inplace = True)
            df.rename(columns = {df.columns[0]:'Answers'}, inplace = True)
        return df



def get_embedding_from_text(text):
    embedding = get_embedding(
        text,
        engine="text-embedding-ada-002"
    )
    return embedding

def batch_embed(df):
    embeddings = []
    num=1
    for item in df["Questions"]:
        print("Embedding sentence:",num)
        num = num+1
        embeddings.append(get_embedding_from_text(item))

    df["Embeddings"] = embeddings
    return df

def save_pickle(df, path):
    pickle.dump(df, open(path, "wb"))


def embed_file(file_to_embed, target_path, openai_key, reverse_columns=False):
    
    openai.api_key = openai_key
    
    df = load_questions_answers(file_to_embed, reverse_columns=reverse_columns)

    
    if "Embeddings" not in df:
        df = batch_embed(df)
        save_pickle(df, target_path)
    else:
        print("Embeddingts already exists found")