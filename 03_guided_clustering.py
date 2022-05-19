#importing required libraries
from datetime import datetime
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
from Core.profile_builder import get_mean_pooling_emb
from Core.scoring import calculate_scores
import pandas as pd


def get_nearest_neighbours(embeding, df):
    t1 = datetime.now()
    tuples = []

    for i, row_e in df.iterrows():
        # print(i)
        dis = cosine_similarity(row_e['embedding'], embeding)
        tuples.append([row_e['document'], dis, row_e['embedding']])

    s_tup = sorted(tuples, key=lambda x: x[1])  # sort tuples based on the cosine distance
    neaarest_neighbs_words = []
    neaarest_neighbs_embs = []
    neaarest_neighbs_labels = []
    for i, m in enumerate(s_tup[::-1]):
        # print(m)
        if (i < 5000):  # getting the nearest 5000 neighbours
            neaarest_neighbs_words.append(m[0])
            neaarest_neighbs_embs.append(m[2])
            neaarest_neighbs_labels.append(m[0])

    t2 = datetime.now()
    diff = t2 - t1
    print('time spent', diff)

    return [{},{'word':neaarest_neighbs_words,'embs':neaarest_neighbs_embs,'labels':neaarest_neighbs_labels}]

#input pickle should contain two columns document and embedding
df = pd.read_pickle(r"E:\Projects\Emotion_work_Gihan\nlp-emotion-AWARE\results\bigrams\merged_bigram_embeddings.pkl")
print(df.columns)



#generate nearest documents to a given seed and save results as a txt file
for group in ['seed1','seed2','seed3','seed4']:

    model = BertModel.from_pretrained("bert-base-cased")

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    group_emb = get_mean_pooling_emb([group], tokenizer, model)

    output = get_nearest_neighbours(group_emb,df)

    words = output[1]['word']

    f = open(r"path_to_save_results"+str(group)+".txt","w+")
    for w in words:
        f.writelines(w+'\n')
    f.close()