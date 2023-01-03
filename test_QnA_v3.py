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

def pre_process(prompt, dialog, model_name):
    dialog=dialog[-3:]
    
    dialog_text=""
    for i in range(len(dialog)):
        if i % 2 == 0:
            dialog_text = dialog_text + "User: " + dialog[i]+"\n"
        else:
            dialog_text = dialog_text + "AI: " + dialog[i]+"\n"
    
    prompt=prompt.strip()+"\n"+dialog_text.strip() + "\n\nRelevant context:"
    print("prompt\n",prompt)
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

def post_process(prompt, dialog, model_name, info):
    dialog=dialog[-3:]
    dialog_text=""
    for i in range(len(dialog)):
        if i % 2 == 0:
            dialog_text = dialog_text + "User: " + dialog[i]+"\n"
        else:
            dialog_text = dialog_text + "AI: " + dialog[i]+"\n"
    
    prompt=prompt.strip()+"\nInfo: "+ info + "\n" + dialog_text.strip() + "\nAI:"
    print("prompt\n",prompt)
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
        
def get_info(text, df, n=3, minimum_similarity=0.85):
    df = copy.copy(df)
    df = text_search(text, df)
    df = df.sort_values("similarities", ascending=False).head(n)
    top_answer = ""
    for i in range(n):
        if df.iloc[i]["similarities"] > minimum_similarity:
            top_answer = top_answer + " " + df["Answers"].iloc[i]
    info= top_answer.strip()


    return info


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
        if len(dialog) > 1:
            context = pre_process(preprocess_prompt, dialog, model_name)
            from_user = from_user + " " + context
            print("preprocess_prompt", context)
            
        info = get_info(from_user, df,minimum_similarity=minimum_similarity)
        
        print("info", info)
        if info:
            reply = post_process(postprocess_prompt, dialog, model_name, info)
            print("postprocess_prompt:", reply)
        else:
            reply = "Sorry, I don't know the answer to that question."
            print("General answer", reply)
        dialog.append(reply)
            