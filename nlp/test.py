from transformers import AlbertForSequenceClassification, AlbertTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

