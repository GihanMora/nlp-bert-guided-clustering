import os
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
#Enable matplotlib to be interactive (zoom etc)
import ast
import pandas as pd
from datetime import datetime
#Mean Pooling - Take attention mask into account for correct averaging
import torch
from transformers import AutoTokenizer, AutoModel, TFBertModel, BertTokenizer, BertModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # print('ime',input_mask_expanded)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    # print('se',sum_embeddings)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_mean_pooling_emb(sentences,tokenizer, model):
    number_of_gpu_cores = 2
    gpu_core = random.randint(0, number_of_gpu_cores-1)
    device = "cuda:"+str(gpu_core) if torch.cuda.is_available() else "cpu"
    print(device)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
    # Compute token embeddings
    with torch.no_grad():
        model = model.to(device)
        model_output = model(**encoded_input)

    sentence_embeddings_raw = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = sentence_embeddings_raw.tolist()

    return sentence_embeddings

if __name__ == "__main__":
    #input csv file with documents in 'text' column
    df = pd.read_csv(r"path to csv file")
    docs = df['text']
    dirname = os.path.dirname(__file__)

    #loading bert model and tokenizer
    model = BertModel.from_pretrained("bert-base-cased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    documents = []
    document_embeddings = []

    tt1 = datetime.now()
    for doc in docs:
        documents.append(doc)

        t1 = datetime.now()
        sentence_pieces = [doc]
        sentence_emb = get_mean_pooling_emb(sentence_pieces, tokenizer, model)
        document_embeddings.append(sentence_emb)
        # print(sentence_emb)
        t2 = datetime.now()
        print(t2-t1)

    tt2 = datetime.now()
    print(tt2 - tt1)

    out_df = pd.DataFrame()
    out_df['document'] = documents
    out_df['embedding'] = document_embeddings

    out_df.to_pickle(r"path to output file")