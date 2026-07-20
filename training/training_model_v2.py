import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)

SEED = 42
set_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() - 1))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

base_dir = "data"
train_file = os.path.join(base_dir, "youtube_comments_v2_train.csv")
val_file   = os.path.join(base_dir, "youtube_comments_v2_val.csv")
test_file  = os.path.join(base_dir, "youtube_comments_v2_test.csv")

train_df = pd.read_csv(train_file)
val_df   = pd.read_csv(val_file)
test_df  = pd.read_csv(test_file)

le = LabelEncoder()
train_labels = le.fit_transform(train_df["sentiment"].astype(str))
val_labels   = le.transform(val_df["sentiment"].astype(str))
test_labels  = le.transform(test_df["sentiment"].astype(str))

num_labels = len(le.classes_)
print("Classes:", list(le.classes_))

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

MAX_LEN = 64

class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False
        )
        enc["labels"] = int(self.labels[idx])
        return enc

train_dataset = CommentDataset(train_df["comment"], train_labels, tokenizer, max_len=MAX_LEN)
val_dataset   = CommentDataset(val_df["comment"],   val_labels,   tokenizer, max_len=MAX_LEN)
test_dataset  = CommentDataset(test_df["comment"],  test_labels,  tokenizer, max_len=MAX_LEN)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

training_args = TrainingArguments(
    output_dir="./results_v2_cpu_fast",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=200,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    eval_strategy="epoch",
    save_strategy="epoch",
)

early_stop = EarlyStoppingCallback(
    early_stopping_patience=1,
    early_stopping_threshold=0.0005
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stop],
)

trainer.train()

device = torch.device("cpu")
model.to(device)
model.eval()

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator
)

all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report on Test Set (V2 CPU fast):")
print(classification_report(all_labels, all_preds, target_names=le.classes_))
