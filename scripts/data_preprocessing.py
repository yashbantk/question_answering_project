import pandas as pd
import re
import nltk
from datasets import load_dataset
import os

# Download necessary NLTK data
nltk.download('punkt')

# Define normalization function
def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # Remove standalone numbers
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = text.strip()
    return text

# Load dataset from the provided URL
dataset = load_dataset("toughdata/quora-question-answer-dataset")

# Convert the dataset to a pandas DataFrame
train_df = pd.DataFrame(dataset['train'])

# Debug: Print columns and first few rows
print("Columns in train_df:", train_df.columns)
print("First few rows of train_df:")
print(train_df.head())

# Ensure the columns are correctly named
if 'question' not in train_df.columns or 'answer' not in train_df.columns:
    raise KeyError("Expected columns 'question' and 'answer' not found in the dataset")

# Normalize text
train_df['question'] = train_df['question'].apply(normalize_text)
train_df['answer'] = train_df['answer'].apply(normalize_text)

# Set the directory to save the preprocessed data
current_dir = os.path.dirname(os.path.abspath(__file__))
preprocessed_data_dir = current_dir  # Save in the same directory as the script

# You can uncomment and modify the following line to set a custom save directory
# preprocessed_data_dir = r"C:\path\to\your\preferred\directory"

print(f"Will attempt to save file in: {preprocessed_data_dir}")

# Save the preprocessed data
preprocessed_data_path = os.path.join(preprocessed_data_dir, "train_preprocessed.csv")
print(f"Attempting to save file: {preprocessed_data_path}")

try:
    train_df.to_csv(preprocessed_data_path, index=False)
    print(f"Preprocessed data saved to {preprocessed_data_path}")
except Exception as e:
    print(f"Error saving file: {e}")