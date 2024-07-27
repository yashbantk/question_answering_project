# visualize_results.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datasets import load_dataset

# Load dataset
dataset = load_dataset("toughdata/quora-question-answer-dataset")
train_df = dataset['train'].to_pandas()

# Length distribution of questions
train_df['question1_length'] = train_df['question1'].apply(len)
train_df['question2_length'] = train_df['question2'].apply(len)

plt.figure(figsize=(12, 6))
sns.histplot(train_df['question1_length'], kde=True, label='Question 1')
sns.histplot(train_df['question2_length'], kde=True, label='Question 2', color='orange')
plt.title('Length Distribution of Questions')
plt.xlabel('Length of Questions')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('results/plots/length_distribution.png')
plt.show()

# Example history object (replace with actual history from training)
history = {
    'epoch': [1, 2, 3],
    'train_loss': [0.6, 0.4, 0.3],
    'val_loss': [0.5, 0.4, 0.35],
    'train_accuracy': [0.75, 0.85, 0.9],
    'val_accuracy': [0.78, 0.82, 0.88]
}

history_df = pd.DataFrame(history)

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss')
plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('results/plots/loss_plot.png')
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history_df['epoch'], history_df['train_accuracy'], label='Training Accuracy')
plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('results/plots/accuracy_plot.png')
plt.show()
