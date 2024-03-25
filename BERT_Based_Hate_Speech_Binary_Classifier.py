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
# For descriptions of all parameters to constructor TrainingArguments, see https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py .
training_arguments = TrainingArguments(
    output_dir = "./training_output",
    overwrite_output_dir = True,
    do_train = True,
    do_eval = True,
    do_predict = True,
    evaluation_strategy = "steps",
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps = 1,
    #eval_accumulation_steps
    #eval_delay
    learning_rate = 5e-5,
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
    metric_for_best_model = "accuracy",
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
