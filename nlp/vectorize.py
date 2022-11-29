import os
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput


class NSMCDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, image_id):
        self.encodings = encodings
        self.labels = labels
        self.image_id = image_id

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item, self.image_id[idx]

    def __len__(self):
        return len(self.labels)


def get_loader(dataset):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    return train_loader


def vectorize(net: nn.Module, dataloader: DataLoader):
    with torch.no_grad():
        device = next(net.parameters()).device
        net = net.eval()
        result = []

        column_name = [f"nlp_{i}" for i in range(768)]

        pbar = tqdm(dataloader)
        for item, image_id in pbar:
            item: Tensor = item

            preds = net(**item)

            res = pd.DataFrame(preds.cpu(), columns=column_name)
            res["image_id"] = image_id

            result += [res]

    return pd.concat(result, axis=0, ignore_index=True)


class CustomDistillBert(DistilBertForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
                labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                    Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                    config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                    `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
                """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        return pooled_output


if __name__ == "__main__":
    config = DistilBertConfig.from_json_file("config.json")
    model = CustomDistillBert(config)
    model.pre_classifier = nn.Linear(model.config.dim, model.config.dim)
    model.classifier = nn.Linear(model.config.dim, 1)

    state_dict = torch.load("pytorch_model.bin", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    df = pd.read_parquet("../kaggle-pog-series-s01e01/dataset.parquet")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    embedding = tokenizer(list(df["description_title"].values), truncation=True, padding=True, max_length=128)
    image_id = list(df["description_title"].values)

    dataset = NSMCDataset(embedding, df["view_count"].values.astype(np.float32), image_id)
    data_loader = get_loader(dataset)
    result: pd.DataFrame = vectorize(model, data_loader)

    print(result)

    result.to_parquet("./nlp.parquet")
