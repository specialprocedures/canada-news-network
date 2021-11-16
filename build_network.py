# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import combinations, groupby

import json
import numpy as np
import pandas as pd
import networkx as nx
from nltk.corpus import stopwords

sw = stopwords.words("english")
# %%

with open("all_entities.json", "r") as f:
    all_entities = json.load(f)

with open("all_aliases.json", "r") as f:
    all_aliases = json.load(f)

with open("input.json", "r") as f:
    data = json.load(f)
# %%


def get_vocab(all_entities, group):
    vocab = []
    for entity in all_entities:
        eg = entity["entity_group"]
        word = entity["word"]
        if (eg == group) and not (word in sw or word.isupper()):
            vocab.append(word)
    return list(set(vocab))


def get_tfidf(
    vocab, corpus=[i["Full text"] for i in data], max_features=500, ngram_range=(2, 3)
):
    tfidf = TfidfVectorizer(
        stop_words="english",
        vocabulary=vocab,
        ngram_range=ngram_range,
        min_df=0.01,
        lowercase=False,
        max_features=max_features,
    )
    X = tfidf.fit_transform(corpus)

    df = pd.DataFrame(
        tfidf.idf_, index=tfidf.get_feature_names(), columns=["idf_weights"]
    ).apply(lambda x: 1 / x)
    return df[df.idf_weights >= df.idf_weights.describe()["75%"]]


per, org, noun = [get_vocab(all_entities, i) for i in ["PER", "ORG", "NOUN"]]

per_df = get_tfidf(per, ngram_range=(2, 3))
org_df = get_tfidf(org, ngram_range=(2, 5))
noun_df = get_tfidf(org, ngram_range=(1, 1))
G = nx.Graph()
for df, cat in zip([per_df, org_df, noun_df], ["PER", "ORG", "NOUN"]):
    for item, row in df.iterrows():
        if item not in G.nodes:
            G.add_node(item, idf_weight=row["idf_weights"], cat=cat)
# %%
trimmed_words = []
for df in [per_df, org_df, noun_df]:
    trimmed_words.extend(df.index.to_list())

#%%
for doc_id in range(len(data)):
    doc_entities = [i for i in all_entities if i["doc_id"] == doc_id]
    num_sent = max(i["sentence"] for i in doc_entities)
    doc_aliases = all_aliases[doc_id]["items"]

    for sent_id in range(num_sent):
        sent_entities = [
            i["word"]
            for i in doc_entities
            if (i["sentence"] == sent_id) and (i["word"] in trimmed_words)
        ]

        for a, b in combinations(sent_entities, 2):

            if a in doc_aliases:
                a = doc_aliases[a]
            if b in doc_aliases:
                b = doc_aliases[b]
            if all(i in G.nodes for i in (a, b)):
                if (a, b) not in G.edges:
                    G.add_edge(a, b, weight=1)
                else:
                    G.edges[(a, b)]["weight"] += 1

#%%

nx.write_gexf(G, "canada.gexf")

# %%
