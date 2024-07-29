import sys
import os
import torch
import transformers
import datasets
import sklearn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

print("\nModule versions:")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")

print(f"\nCUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples['question'], truncation=True, padding=True)
    tokenized_inputs['labels'] = examples['encoded_answer']  # Use the pre-encoded labels
    return tokenized_inputs

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Load dataset
dataset = load_dataset("toughdata/quora-question-answer-dataset")
print("Dataset loaded successfully")

# Convert the dataset to a Pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset['train'])

# Sample 20% of the data
sample_size = int(len(df) * 0.20)
df_sampled = df.sample(n=sample_size, random_state=42)

print(f"Original dataset size: {len(df)}")
print(f"Sampled dataset size: {len(df_sampled)} (20% of original)")

# Inspect the unique values in the 'answer' column of the sampled dataset
print("Unique values in 'answer' column of sampled data:")
print(df_sampled['answer'].unique())
print(f"Number of unique answers in sampled data: {df_sampled['answer'].nunique()}")

# Initialize the LabelEncoder and fit it to the 'answer' column of the sampled dataset
label_encoder = LabelEncoder()
df_sampled['encoded_answer'] = label_encoder.fit_transform(df_sampled['answer'])

# Convert the sampled DataFrame back to a HuggingFace Dataset
dataset_sampled = datasets.Dataset.from_pandas(df_sampled)

# Splitting the sampled dataset into training and validation sets
dataset_sampled = dataset_sampled.shuffle(seed=42)
train_testvalid = dataset_sampled.train_test_split(test_size=0.2)
train_dataset = train_testvalid['train']
test_valid_dataset = train_testvalid['test']
test_valid_dataset = test_valid_dataset.train_test_split(test_size=0.5)
val_dataset = test_valid_dataset['train']
test_dataset = test_valid_dataset['test']

print("Sampled dataset split into training, validation, and test sets")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("Tokenizer loaded successfully")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
print("Model loaded successfully")

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
print("Datasets preprocessed successfully")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()
print("Training completed")

print("Evaluating on test set...")
test_results = trainer.evaluate(tokenized_test_dataset)
print(f"Test results: {test_results}")

print("Saving model...")
trainer.save_model("models/bert-base-uncased-20percent")
print("Model saved successfully")

print("\nScript execution completed.")