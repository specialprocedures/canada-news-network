# %%
# Core library imports
from itertools import combinations, groupby
import json

# NLP imports
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# %%

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
