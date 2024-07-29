import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples['question'], truncation=True, padding=True)
    tokenized_inputs['labels'] = examples['encoded_answer']
    return tokenized_inputs

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Load the full dataset
dataset = load_dataset("toughdata/quora-question-answer-dataset")
print("Dataset loaded successfully")

# Convert the dataset to a Pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset['train'])

# Sample 20% of the data
sample_size = int(len(df) * 0.20)
df_sampled = df.sample(n=sample_size, random_state=42)

print(f"Original dataset size: {len(df)}")
print(f"Sampled dataset size: {len(df_sampled)} (20% of original)")

# Initialize the LabelEncoder and fit it to the 'answer' column of the sampled dataset
label_encoder = LabelEncoder()
df_sampled['encoded_answer'] = label_encoder.fit_transform(df_sampled['answer'])

# Convert the sampled DataFrame back to a HuggingFace Dataset
dataset_sampled = Dataset.from_pandas(df_sampled)

# Splitting the sampled dataset into training and test sets (80-20 split)
train_test_split = dataset_sampled.train_test_split(test_size=0.2, seed=42)
test_dataset = train_test_split['test']

print(f"Test set size: {len(test_dataset)}")

# Load the tokenizer and model
model_path = "models/bert-base-uncased-20percent"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print("Model and tokenizer loaded successfully from saved files")
except OSError:
    print(f"Couldn't load model or tokenizer from {model_path}. Falling back to original BERT model.")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))
    
    # Load the state dict
    try:
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        print("Model state loaded successfully")
    except FileNotFoundError:
        print(f"Couldn't find saved model state at {model_path}. Using original BERT weights.")

# Tokenize the test dataset
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate the model
print("Evaluating the model...")
eval_result = trainer.evaluate(tokenized_test_dataset)
print(eval_result)

# Save evaluation results
os.makedirs('results/metrics', exist_ok=True)
with open('results/metrics/evaluation_results_20percent.txt', 'w') as f:
    f.write(str(eval_result))

print("Evaluation results saved to results/metrics/evaluation_results_20percent.txt")