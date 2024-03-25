import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('HateSpeechDataset.csv')
print(f'Shape Of Hate Speech Data Set: {df.shape}')
print(f'Head Of Hate Speech Data Set:\n{df.head(3)}')

#df = df.dropna(subset = ['Label'])
#df = df.astype({'Label': str})
df = df[df['Label'].apply(lambda x: x.isnumeric())]
df['Label'] = df['Label'].astype(int)
df = df.drop('Content_int', axis=1)
df = df.rename(columns={'Label':'labels'})
df.reset_index(inplace=True, drop=False)
df.rename(columns={'index': 'id'}, inplace=True)
print(f'Shape Of Hate Speech Data Set: {df.shape}')
print(f'Head Of Hate Speech Data Set:\n{df.head(3)}')

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(examples):
    return tokenizer(examples['Content'], padding="max_length", truncation=True)
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
shuffled_dataset = tokenized_dataset.shuffle(seed=42)
sampled_dataset = shuffled_dataset.select(range(20000))

train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})

from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2) # Adjust `num_labels` as necessary

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy='epoch',
)

training_arguments = TrainingArguments(
    output_dir="./training_output",
    logging_dir=None,  # Disables TensorBoard logs
    save_strategy="no",  # Disables saving model checkpoints
    evaluation_strategy="no",  # Disables evaluation during training
    logging_strategy="no",  # Disables logging
    load_best_model_at_end=False,  # Avoids loading the best model at the end, which can save disk space
)

trainer.train()
