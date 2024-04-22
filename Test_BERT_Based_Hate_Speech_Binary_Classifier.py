from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import numpy as np
import pandas as pd
import torch  # Import torch to use softmax

checkpoint_path = './training_output/checkpoint-14000'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
input_data_frame = pd.read_feather('df_2020C_07_01_to_02_testdata.feather')[['body']].rename(columns={'body': 'text'})
output_filename = 'Data_Frame_Of_Texts_And_Predictions--2020-07-01--2020-07-02.csv'
output_data_frame = pd.read_csv(output_filename, index_col = False, header = 0)
input_data_frame_minus_output_data_frame = input_data_frame[~input_data_frame['text'].isin(output_data_frame['text'])]
input_data_frame_minus_output_data_frame['prediction'] = np.nan
print(input_data_frame_minus_output_data_frame.shape)
number_of_texts = 1_000
threshold = 0.33
#data_frame.iloc[0 : 0].to_csv(path_or_buf = output_filename, mode = 'w', index = False)
for i in range(0, len(input_data_frame_minus_output_data_frame), number_of_texts):
    the_slice = input_data_frame_minus_output_data_frame.iloc[i : i + number_of_texts]
    list_of_texts = the_slice['text'].to_list()
    dictionary_of_encodings = tokenizer(list_of_texts, padding="max_length", truncation=True)
    data_set = Dataset.from_dict({key: np.array(dictionary_of_encodings[key]) for key in dictionary_of_encodings})
    trainer = Trainer(model = model)
    outputs = trainer.predict(data_set)
    softmax_scores = torch.nn.functional.softmax(torch.tensor(outputs.predictions), dim = -1).numpy()
    predicted_labels = (softmax_scores[:, 1] >= threshold).astype(int)  # Assume class 1 is the "positive" class.
    input_data_frame_minus_output_data_frame.loc[the_slice.index, 'prediction'] = predicted_labels
    the_slice = input_data_frame_minus_output_data_frame.iloc[i: i + number_of_texts]
    the_slice.to_csv(path_or_buf = output_filename, mode = 'a', index = False, header = False)