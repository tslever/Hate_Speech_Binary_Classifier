# Load And Preprocess Data
import pandas as pd
df = pd.read_csv('HateSpeechDataset.csv')
print(f'Shape Of Hate Speech Data Set: {df.shape}')
print(f'Head Of Hate Speech Data Set:\n{df.head(3)}')
df = df[df['Label'].apply(lambda x: x.isnumeric())]
df['Label'] = df['Label'].astype(int)
df = df.drop('Content_int', axis=1)
df = df.rename(columns={'Label':'labels'})
df.reset_index(inplace=True, drop=False)
df.rename(columns={'index': 'id'}, inplace=True)
print(f'Shape Of Hate Speech Data Set: {df.shape}')
print(f'Head Of Hate Speech Data Set:\n{df.head(3)}')

# Tokenize And Prepare Data
checkpoint = 'bert-base-uncased'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(examples):
    return tokenizer(examples['Content'], padding="max_length", truncation=True)
from datasets import Dataset
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
from datasets import DatasetDict
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

# Model Training Setup
import numpy as np
from sklearn.metrics import accuracy_score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
from transformers import TrainingArguments
training_arguments = TrainingArguments(
    output_dir="./training_output",
    logging_dir = None,  # Disables TensorBoard logs
    save_strategy = "no",  # Disables saving model checkpoints
    evaluation_strategy = "epoch", #"no",  # Disables evaluation during training
    logging_strategy = "no",  # Disables logging
    load_best_model_at_end = False,  # Avoids loading the best model at the end, which can save disk space
)

from transformers import Trainer
trainer = Trainer(
    model = model,
    args = training_arguments,
    train_dataset = dataset_dict['train'],
    eval_dataset = dataset_dict['validation'],
    compute_metrics = compute_metrics
)

# Train
trainer.train()
