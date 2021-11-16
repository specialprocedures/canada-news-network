# %%# %%
# Core library imports
from itertools import combinations
import json

# Import progressbar
from tqdm import tqdm

# NLP imports
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Setup tokenizers


def make_pipeline(model_name: str) -> pipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    pipe = pipeline("token-classification",
                    device=0,
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy='average'
                    )
    return pipe


models = ["dslim/bert-large-NER",
          "vblagoje/bert-english-uncased-finetuned-pos"]

ner_pipe, pos_pipe = [make_pipeline(i) for i in models]

# %% Load data
with open('input.json', 'r') as f:
    d = json.load(f)

# %% Helper functions


def split_sentences(text: str) -> list:
    if "." in text:
        return [i for i in text.split('.') if len(i) > 1]
    else:
        return [text]


def classify_tokens(text: str, pipe: pipeline) -> list:
    out = []
    sents = split_sentences(text)
    results = pipe(sents)

    for n, s in enumerate(results):
        if isinstance(s, list):
            for entity in s:
                entity['sentence'] = n
                entity.pop('score')
                out.append(entity)
        elif isinstance(s, dict):
            s['sentence'] = n
            s.pop('score')
            out.append(s)
    return out


def get_alias(item: dict, alias_type: str) -> list:
    entities = item['entities']
    aliases = {}

    targets = [i['word'] for i in entities if i['entity_group'] == alias_type]
    for pair in combinations(targets, 2):
        a, b = sorted(pair, key=len)

        if (a in b) and (a != b) and (a not in aliases):
            aliases.update({a: b})
    return aliases


# Run models
for n, item in enumerate(tqdm(d)):
    item['id'] = n
    text = item['Full text']
    ner = classify_tokens(text, ner_pipe)
    pos = classify_tokens(text, pos_pipe)
    item['entities'] = ner + pos
    item['aliases'] = {}
    item['aliases']['PER'] = get_alias(item, 'PER')

# %%

# Create lists of all entities and aliases

all_aliases = []
all_entities = []

for item in d:
    for entity in item['entities']:
        for loc in ['start', 'end']:
            entity[loc] = int(entity[loc])

        entity['doc_id'] = item['id']
        all_entities.append(entity)

    doc_alias = {}
    doc_alias['doc_id'] = item['id']
    doc_alias['items'] = item['aliases']['PER']
    all_aliases.append(doc_alias)

# %%

# Dump to files

with open('processed.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=4, ensure_ascii=False)

with open('all_aliases.json', 'w', encoding='utf-8') as f:
    json.dump(all_aliases, f, indent=4, ensure_ascii=False)

with open('all_entities.json', 'w', encoding='utf-8') as f:
    json.dump(all_entities, f, indent=4, ensure_ascii=False)
# %%


# def get_ngrams(item):
#     entities = item['entities']
#     ngrams = []
#     n = 0
#     sentences = []
#     while True:
#         sentence = [i for i in entities if i['sentence'] == n]
#         n += 1
#         if len(sentence) == 0:
#             break
#         sentences.append(sentence)

#     for sentence in sentences:
#         for n in range(len(sentence)):
#             if sentence[n]['entity_group'] == 'NOUN':
#                 noun_set = [(sentence[n]['word'], sentence[n]['start'])]
#                 m = 1
#                 while True:
#                     loc = m + n
#                     print(loc)
#                     if loc >= len(sentence):
#                         m += 1
#                         break
#                     elif sentence[loc]['entity_group'] == 'NOUN':
#                         noun_set.append(
#                             (sentence[loc]['word'], sentence[n]['end']))
#                         m += 1
#                     else:
#                         break
#                 if len(noun_set) > 1:
#                     noun_text = " ".join([i[0] for i in noun_set])
#                     start = noun_set[0][1]
#                     end = noun_set[-1][1]
#                     if len(noun_text) > 3:
#                         entity = {
#                             "entity_group": "NOUN",
#                             "word": noun_text,
#                             "start": start,
#                             "end": end
#                         }

#                         ngrams.append(entity)
#         return ngrams

# %%
