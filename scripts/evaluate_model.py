import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def preprocess_function(examples):
    return tokenizer(examples['question1'], examples['question2'], truncation=True, padding=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Load dataset
dataset = load_dataset("toughdata/quora-question-answer-dataset")

tokenizer = AutoTokenizer.from_pretrained("models/bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("models/bert-base-uncased")

tokenized_datasets = dataset.map(preprocess_function, batched=True)
val_dataset = tokenized_datasets["validation"]

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

eval_result = trainer.evaluate(val_dataset)
print(eval_result)

# Save evaluation results
with open('results/metrics/evaluation_results.txt', 'w') as f:
    f.write(str(eval_result))
