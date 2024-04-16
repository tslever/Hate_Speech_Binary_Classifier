from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import numpy as np
import pandas as pd

checkpoint_path = './training_output/checkpoint-14000'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
data_frame = pd.read_feather('df_2020C_6_11_Clean.feather')[['body']].rename(columns = {'body': 'text'})
data_frame['prediction'] = np.nan
number_of_texts = 1_000
for i in range(0, len(data_frame), number_of_texts):
    the_slice = data_frame.iloc[i : i + number_of_texts]
    list_of_texts = the_slice['text'].to_list()
    dictionary_of_encodings = tokenizer(list_of_texts, padding = "max_length", truncation = True)
    data_set = Dataset.from_dict({key: np.array(dictionary_of_encodings[key]) for key in dictionary_of_encodings})
    trainer = Trainer(model = model)
    numpy_array_of_predictions = trainer.predict(data_set).predictions.argmax(-1)
    data_frame.loc[the_slice.index, 'prediction'] = numpy_array_of_predictions
    the_slice = data_frame.iloc[i : i + number_of_texts]
    the_slice.to_csv(path_or_buf = 'Data_Frame_Of_Texts_And_Predictions.csv', mode = 'a', index = False, header = False)