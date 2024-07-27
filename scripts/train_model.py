import sys
import os

def check_import(module_name):
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

required_modules = ['torch', 'transformers', 'datasets', 'sklearn', 'pandas']

print("Checking required modules:")
for module in required_modules:
    print(f"{module}: {'Installed' if check_import(module) else 'Not Installed'}")

try:
    import torch
    import transformers
    import datasets
    import sklearn
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
        return tokenizer(examples['question1'], examples['question2'], truncation=True, padding=True)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # Load dataset
    dataset = load_dataset("toughdata/quora-question-answer-dataset")
    print("Dataset loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("Tokenizer loaded successfully")

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    print("Model loaded successfully")

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    print("Datasets preprocessed successfully")

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
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    print("Training completed")

    print("Saving model...")
    trainer.save_model("models/bert-base-uncased")
    print("Model saved successfully")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("\nDetailed error information:")
    import traceback
    traceback.print_exc()

print("\nScript execution completed.")
