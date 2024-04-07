# Load And Preprocess Data
import pandas as pd
data_frame = pd.read_csv('HateSpeechDataset.csv', dtype = str)
data_frame = data_frame[['Content', 'Label']]
data_frame = data_frame[data_frame['Label'].apply(lambda x: x.isnumeric())]
data_frame['Label'] = data_frame['Label'].astype(int)
data_frame = data_frame.rename(columns = {'Label': 'labels'})


# Tokenize And Prepare Data
# Instantiate one of the tokenizer classes of the library from a pretrained model vocabulary.
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize(data_set):
    return tokenizer(data_set['Content'], padding = "max_length", truncation = True)

from datasets import Dataset
data_set = Dataset.from_pandas(data_frame)
tokenized_dataset = data_set.map(tokenize, batched = True)
dictionary_of_training_and_testing_data_sets = tokenized_dataset.train_test_split(test_size = 0.1)

from datasets import DatasetDict
dictionary_of_training_and_validation_data_sets = DatasetDict({
    'training': dictionary_of_training_and_testing_data_sets['train'],
    'validation': dictionary_of_training_and_testing_data_sets['test']
})


# Model Training Setup
global_step = 0

import numpy as np
def softmax(x, axis = None):
    e_x = np.exp(x - np.max(x, axis = axis, keepdims = True))
    return e_x / e_x.sum(axis = axis, keepdims = True)

path_to_training_output = "./training_output_2"

import os
def compute_metrics(evalPrediction):
    if not os.path.exists(path_to_training_output):
        os.makedirs(path_to_training_output)
    numpy_array_of_logits, numpy_array_of_labels = evalPrediction # 1000 x 2, 1000 x 1
    numpy_array_of_probabilities = softmax(numpy_array_of_logits, axis = 1)[:, 1]
    numpy_array_of_thresholds = np.linspace(0, 1, 101)
    list_of_accuracies = []
    list_of_TPRs = []
    list_of_FPRs = []
    list_of_PPVs = []
    list_of_F1_measures = []
    maximum_F1_measure = -1.0
    index_of_maximum_F1_measure = -1
    for i in range(0, len(numpy_array_of_thresholds)):
        numpy_array_of_predictions_at_threshold = (numpy_array_of_probabilities >= numpy_array_of_thresholds[i]).astype(int)
        accuracy = (numpy_array_of_predictions_at_threshold == numpy_array_of_labels).sum() / len(numpy_array_of_labels)
        FN = ((numpy_array_of_predictions_at_threshold == 0) & (numpy_array_of_labels == 1)).sum()
        FP = ((numpy_array_of_predictions_at_threshold == 1) & (numpy_array_of_labels == 0)).sum()
        TN = ((numpy_array_of_predictions_at_threshold == 0) & (numpy_array_of_labels == 0)).sum()
        TP = ((numpy_array_of_predictions_at_threshold == 1) & (numpy_array_of_labels == 1)).sum()
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
        F1_measure = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) > 0 else 0
        list_of_accuracies.append(accuracy)
        list_of_TPRs.append(TPR)
        list_of_FPRs.append(FPR)
        list_of_PPVs.append(PPV)
        list_of_F1_measures.append(F1_measure)
        if F1_measure > maximum_F1_measure:
            maximum_F1_measure = F1_measure
            index_of_maximum_F1_measure = i
    data_frame_of_performance_metrics = pd.DataFrame({
        'threshold': numpy_array_of_thresholds,
        'F1_measure': list_of_F1_measures,
        'accuracy': list_of_accuracies,
        'True Positive Rate / Recall': list_of_TPRs, 
        'False Positive Rate': list_of_FPRs,
        'Positive Predictive Value / Precision': list_of_PPVs
    }).sort_values(
        by = ['F1_measure', 'accuracy', 'True Positive Rate / Recall', 'False Positive Rate', 'Positive Predictive Value / Precision'],
        ascending = [False, False, False, False, False]
    )
    global global_step
    global_step += 500
    data_frame_of_performance_metrics.to_csv(
        path_or_buf = f"{path_to_training_output}/Data_Frame_Of_Performance_Metrics_After_{str(global_step).zfill(7)}_Steps.csv",
        index = False
    )
    return {
        'maximum_F1_measure': maximum_F1_measure,
        'corresponding_threshold': numpy_array_of_thresholds[index_of_maximum_F1_measure],
        'corresponding_accuracy': list_of_accuracies[index_of_maximum_F1_measure],
        'corresponding_TPR': list_of_TPRs[index_of_maximum_F1_measure],
        'corresponding_FPR': list_of_FPRs[index_of_maximum_F1_measure],
        'corresponding_PPV': list_of_PPVs[index_of_maximum_F1_measure]
    }

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)

from transformers import TrainingArguments
# For descriptions of all parameters to constructor TrainingArguments, see https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py .
training_arguments = TrainingArguments(
    output_dir = path_to_training_output,
    overwrite_output_dir = True,
    do_train = True,
    do_eval = True,
    do_predict = True,
    evaluation_strategy = "steps",
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps = 1,
    #eval_accumulation_steps
    #eval_delay
    learning_rate = 3e-5,
    weight_decay = 0,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_epsilon = 1e-8,
    max_grad_norm = 1.0,
    num_train_epochs = 3.0,
    max_steps = -1,
    lr_scheduler_type = "linear",
    lr_scheduler_kwargs = {},
    warmup_ratio = 0.0,
    warmup_steps = 0,
    log_level = "passive",
    log_level_replica = "passive",
    log_on_each_node = True,
    #logging_dir
    logging_strategy = "steps",
    logging_first_step = False,
    logging_steps = 500,
    logging_nan_inf_filter = True,
    save_strategy = "steps",
    save_steps = 500,
    save_total_limit = 1,
    save_safetensors = True,
    save_on_each_node = False,
    save_only_model = False,
    use_cpu = False,
    seed = 42,
    #data_seed
    jit_mode_eval = False,
    use_ipex = False,
    bf16 = False,
    fp16 = False,
    fp16_opt_level = "O1",
    #fp16_backend = "auto",
    half_precision_backend = "auto",
    bf16_full_eval = False,
    fp16_full_eval = False,
    #tf32
    local_rank = -1,
    #ddp_backend
    #tpu_num_cores
    dataloader_drop_last = False,
    #eval_steps
    dataloader_num_workers = 0,
    past_index = -1,
    #run_name
    #disable_tqdm
    remove_unused_columns = True,
    #label_names
    load_best_model_at_end = True,
    metric_for_best_model = "maximum_F1_measure",
    greater_is_better = True,
    ignore_data_skip = False,
    fsdp = '',
    #fsdp_config
    #deepspeed
    #accelerator_config
    label_smoothing_factor = 0.0,
    debug = "",
    optim = "adamw_torch",
    #optim_args
    group_by_length = False,
    length_column_name = "length",
    report_to = "all",
    #ddp_find_unused_parameters
    #ddp_bucket_cap_mb
    #ddp_broadcast_buffers
    dataloader_pin_memory = True,
    dataloader_persistent_workers = False,
    #dataloader_prefetch_factor
    skip_memory_metrics = True,
    push_to_hub = False,
    #resume_from_checkpoint
    #hub_model_id
    hub_strategy = "every_save",
    #hub_token
    hub_private_repo = False,
    hub_always_push = False,
    gradient_checkpointing = False,
    gradient_checkpointing_kwargs = None,
    include_inputs_for_metrics = False,
    auto_find_batch_size = False,
    full_determinism = True,
    #torchdynamo
    ray_scope = "last",
    ddp_timeout = 1800,
    use_mps_device = False,
    torch_compile = False,
    #torch_compile_backend
    #torch_compile_mode
    #split_batches
    #include_tokens_per_second
    #include_num_input_tokens_seen
    #neftune_noise_alpha
    #optim_target_modules
)

from transformers import Trainer, EarlyStoppingCallback
trainer = Trainer(
    model = model,
    args = training_arguments,
    train_dataset = dictionary_of_training_and_validation_data_sets['training'],
    eval_dataset = dictionary_of_training_and_validation_data_sets['validation'],
    compute_metrics = compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)]
)


# Train
trainer.train()