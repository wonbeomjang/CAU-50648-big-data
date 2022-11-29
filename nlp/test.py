import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

df = pd.read_parquet("../kaggle-pog-series-s01e01/dataset.parquet")

train_encodings = tokenizer(list(df["description_title"].values), truncation=True, padding=True, max_length=128)

