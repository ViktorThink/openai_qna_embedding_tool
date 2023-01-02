import openai
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import copy
import pickle

def get_embedding_from_text(text):
    embedding = get_embedding(
        text,
        engine="text-embedding-ada-002"
    )
    return embedding

def text_search(text, df):
    embedding = get_embedding_from_text(text)

    for item in df:
      df["similarities"] = df.Embeddings.apply(lambda x: cosine_similarity(x, embedding))
    return df

def process(prompt, dialog, model_name):
    
    dialog_text=""
    for i in range(len(dialog)):
        if i % 2 == 0:
            dialog_text = dialog_text + "User: " + dialog[i]+"\n"
        else:
            dialog_text = dialog_text + "AI: " + dialog[i]+"\n"
    
    prompt=prompt.strip()+"\n"+dialog_text.strip() + "\n\nRewritten last message:"
    # print("prompt\n",prompt)
    reply = openai.Completion.create(
    model=model_name,
    prompt=prompt,
    temperature=0.1,
    top_p=1,
    frequency_penalty=0,
    stop=["#"],
    max_tokens=1000)
    
    reply = reply["choices"][0]["text"].strip()
    
    return reply
        
def get_top_reply(text, df, n=1, minimum_similarity=0.85):
    df = copy.copy(df)
    df = text_search(text, df)
    df = df.sort_values("similarities", ascending=False).head(n)
    if df.iloc[0]["similarities"] < minimum_similarity:
        top_question = "?"
        top_answer = "Sorry, I don't know the answer to that question."
    else:
        top_question = df["Questions"].iloc[0]
        top_answer = df["Answers"].iloc[0]

    return top_answer


def test_QnA(path, openai_key, num_replies=4, minimum_similarity=0.85,preprocess_prompt=None, postprocess_prompt=None, model_name="text-davinci-003"):
    
    if preprocess_prompt:
        with open(preprocess_prompt, 'r') as file:
            preprocess_prompt = file.read()
            
    if postprocess_prompt:
        with open(postprocess_prompt, 'r') as file:
            postprocess_prompt = file.read()
            
    openai.api_key = openai_key
    
    df = pickle.load(open(path, "rb"))
    dialog=[]
    for i in range(num_replies):
        dialog=dialog[-4:]
        from_user = input("Message: ")
        dialog.append(from_user)
        if preprocess_prompt:
            from_user = process(preprocess_prompt, dialog[-3:], model_name)
            dialog[-1] = from_user
            print("preprocess_prompt", from_user)
        reply = get_top_reply(from_user, df,minimum_similarity=minimum_similarity)
        
        print("Reply", reply)
        dialog.append(reply)
        if postprocess_prompt:
            reply = process(postprocess_prompt, dialog[-2:], model_name)
            dialog[-1] = reply
            print("postprocess_prompt:", reply)