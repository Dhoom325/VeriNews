import torch
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
import pandas as pd

df = pd.read_csv("datasets/isot/True.csv")
df["statement"] = ("Title: " + df["title"].fillna("") + " Article: " + df["text"].fillna(""))

from transformers import RobertaTokenizer, RobertaForSequenceClassification

model_path = "saved_model/roberta-welfake"

model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

model.eval()  # very important
print("Model loaded")
model = model.to(DEVICE)

texts = df["statement"].tolist()
def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=256
)

import numpy as np
from torch.utils.data import DataLoader

batch_size = 32
all_preds = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    preds = outputs.logits.argmax(dim=1).cpu().numpy()
    all_preds.extend(preds)


import collections

counter = collections.Counter(all_preds)
print(counter)

total = len(all_preds)

real_pct = counter.get(0, 0) / total * 100
fake_pct = counter.get(1, 0) / total * 100

print(f"Predicted REAL: {real_pct:.2f}%")
print(f"Predicted FAKE: {fake_pct:.2f}%")
# label_map = {0: "Real", 1: "Fake"}

# text = "Breaking News: Scientists confirm drinking salt water cures all diseases overnight."

# inputs = tokenizer(
#     text,
#     return_tensors="pt",
#     truncation=True,
#     padding=True,
#     max_length=256
# )

# with torch.no_grad():          # disables training behavior
#     outputs = model(**inputs)

# prediction = outputs.logits.argmax(dim=1).item()

# print("Prediction:", label_map[prediction])