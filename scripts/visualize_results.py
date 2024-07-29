import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datasets import load_dataset
import os
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Create directory for saving plots
os.makedirs('results/plots', exist_ok=True)

# Load dataset
dataset = load_dataset("toughdata/quora-question-answer-dataset")
train_df = dataset['train'].to_pandas()

print("Columns in the dataset:", train_df.columns)

# Length distribution of questions and answers
train_df['question_length'] = train_df['question'].apply(len)
train_df['answer_length'] = train_df['answer'].apply(len)

plt.figure(figsize=(12, 6))
sns.histplot(train_df['question_length'], kde=True, label='Question')
sns.histplot(train_df['answer_length'], kde=True, label='Answer', color='orange')
plt.title('Length Distribution of Questions and Answers')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('results/plots/length_distribution.png')
plt.close()

# Box plot of question and answer lengths
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df[['question_length', 'answer_length']])
plt.title('Box Plot of Question and Answer Lengths')
plt.savefig('results/plots/length_boxplot.png')
plt.close()

# Scatter plot of question length vs answer length
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_df, x='question_length', y='answer_length', alpha=0.5)
plt.title('Question Length vs Answer Length')
plt.xlabel('Question Length')
plt.ylabel('Answer Length')
plt.savefig('results/plots/length_scatter.png')
plt.close()

# Word cloud of questions
text = ' '.join(train_df['question'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Questions')
plt.savefig('results/plots/question_wordcloud.png')
plt.close()

# Word cloud of answers
text = ' '.join(train_df['answer'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Answers')
plt.savefig('results/plots/answer_wordcloud.png')
plt.close()

# Top 20 most common words in questions
vectorizer = CountVectorizer(stop_words='english')
question_bow = vectorizer.fit_transform(train_df['question'])
word_freq = dict(zip(vectorizer.get_feature_names_out(), question_bow.sum(axis=0).A1))
top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

plt.figure(figsize=(12, 6))
sns.barplot(x=[word for word, freq in top_words], y=[freq for word, freq in top_words])
plt.title('Top 20 Most Common Words in Questions')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/plots/top_words_questions.png')
plt.close()

# Correlation heatmap
correlation_matrix = train_df[['question_length', 'answer_length']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('results/plots/correlation_heatmap.png')
plt.close()

# Example history object (replace with actual history from training)
history = {
    'epoch': [1, 2, 3, 4, 5],
    'train_loss': [0.6, 0.4, 0.3, 0.25, 0.2],
    'val_loss': [0.5, 0.4, 0.35, 0.32, 0.3],
    'train_accuracy': [0.75, 0.85, 0.9, 0.92, 0.94],
    'val_accuracy': [0.78, 0.82, 0.88, 0.89, 0.91]
}

history_df = pd.DataFrame(history)

# Interactive line plot for training and validation metrics
fig = go.Figure()
fig.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['train_loss'], mode='lines+markers', name='Training Loss'))
fig.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['val_loss'], mode='lines+markers', name='Validation Loss'))
fig.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['train_accuracy'], mode='lines+markers', name='Training Accuracy'))
fig.add_trace(go.Scatter(x=history_df['epoch'], y=history_df['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
fig.update_layout(title='Training and Validation Metrics', xaxis_title='Epoch', yaxis_title='Value')
fig.write_html('results/plots/training_metrics_interactive.html')

# Save as static image using matplotlib
plt.figure(figsize=(12, 6))
plt.plot(history_df['epoch'], history_df['train_loss'], 'b-', label='Training Loss')
plt.plot(history_df['epoch'], history_df['val_loss'], 'r-', label='Validation Loss')
plt.plot(history_df['epoch'], history_df['train_accuracy'], 'g-', label='Training Accuracy')
plt.plot(history_df['epoch'], history_df['val_accuracy'], 'y-', label='Validation Accuracy')
plt.title('Training and Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.savefig('results/plots/training_metrics.png')
plt.close()

# Confusion matrix (example data, replace with actual results)
confusion_matrix = np.array([[80, 20], [10, 90]])
labels = ['Negative', 'Positive']

fig = px.imshow(confusion_matrix,
                x=labels,
                y=labels,
                color_continuous_scale='Viridis',
                labels=dict(x="Predicted", y="Actual", color="Count"))
fig.update_layout(title='Confusion Matrix')
fig.write_html('results/plots/confusion_matrix.html')

# Save as static image using matplotlib
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('results/plots/confusion_matrix.png')
plt.close()

# ROC curve (example data, replace with actual results)
fpr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
tpr = np.array([0, 0.4, 0.5, 0.7, 0.8, 0.8, 0.85, 0.9, 0.95, 0.98, 1])

fig = px.line(x=fpr, y=tpr, labels={'x':'False Positive Rate', 'y':'True Positive Rate'})
fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve')
fig.write_html('results/plots/roc_curve.html')

# Save as static image using matplotlib
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', label='ROC curve')
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig('results/plots/roc_curve.png')
plt.close()

print("All plots have been saved in the 'results/plots' directory.")