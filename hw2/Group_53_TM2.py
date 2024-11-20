import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict,Sequence,ClassLabel
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np
from transformers import AutoModelForTokenClassification
from huggingface_hub import notebook_login
from transformers import TrainingArguments
from transformers import Trainer
import os
from huggingface_hub import login
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
import optuna
from transformers import BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from evaluate import load
from sklearn.metrics import precision_recall_fscore_support



metric = evaluate.load("seqeval")

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# #extracting tokens and labels

def extract_tokens_and_labels(file_path):
    sentences_items = []
    current_sentence = {"tokens":[],"POS_TAGS":[],"NER_TAGS":[]}
    unique_labels= set()

    with open(file_path, 'r') as file:
        lines=file.readlines() # read all lines in the file

    for line in lines: # for each line of file
        line=line.lstrip() # stripping white space at start of line
        line=line.rstrip()
        line_parts = line.split(' ') # splitting the line into parts like ['modern' 'JJ' 'B-PER']
    
        try:
            if len(line_parts) == 3: # line should have 3 parts to indicate token, POS, Label
                token = line_parts[0]
                pos = line_parts[1]
                NER = line_parts[2]

                current_sentence["tokens"].append(token)
                current_sentence["POS_TAGS"].append(pos)
                current_sentence["NER_TAGS"].append(NER)
                unique_labels.add(NER)

        except ValueError:
            continue

        if line == '': # if line is empty
            if current_sentence["tokens"]:
                sentences_items.append(current_sentence) # set sentences_items with word as key and label as value])
                current_sentence = {"tokens":[],"POS_TAGS": [], "NER_TAGS": []}

    if current_sentence["tokens"]:
            sentences_items.append(current_sentence)

    return sentences_items, unique_labels

converted_train,unique_labels_train = extract_tokens_and_labels('train.txt')
converted_valid,unique_labels_val  = extract_tokens_and_labels('val.txt')
converted_test,unique_labels_test = extract_tokens_and_labels('test.txt')
#print(converted_train[:5])

train_dataset = Dataset.from_list(converted_train)
validation_dataset = Dataset.from_list(converted_valid)
test_dataset = Dataset.from_list(converted_test)


model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Assuming you have your dataset prepared as dataset_dict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

labels_list_train = sorted(unique_labels_train)
label_map = {label: idx for idx, label in enumerate(labels_list_train)}

def adjusted_labels(NER_TAGS,word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else label_map[NER_TAGS[word_id]]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = label_map[NER_TAGS[word_id]]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels    


def tokenize_and_align_labels(sentences):
    tokenized_inputs = tokenizer(
        sentences["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = sentences["NER_TAGS"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(adjusted_labels(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

tokenized_datasets = dataset_dict.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset_dict["train"].column_names
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
print(batch["labels"])

ner_feature = dataset_dict["train"].features["NER_TAGS"]
label_names = sorted(unique_labels_train)

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100 and l < len(label_names)] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100 and p < len(label_names)]
        for prediction, label in zip(predictions, labels)
    ] # just converting the model prediction of numbers (2,3).. to corresponding name labels like (B-PER,I-PER)
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#login()

args = TrainingArguments(
    "bert-finetuned-ner",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    #push_to_hub=True,
)

#will handle training and evaluation of the model
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()

test_results = trainer.evaluate(tokenized_datasets["test"])
print("Test result is:", test_results)
#trainer.push_to_hub(commit_message="Training complete")

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False,collate_fn=data_collator, batch_size=8
)

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

# test_dataloader = DataLoader(
#     tokenized_datasets["test"], collate_fn=data_collator, batch_size=8
# )

optimizer = AdamW(model.parameters(), lr=2e-5)

accelerator = Accelerator()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

#model, optimizer= accelerator.prepare(model, optimizer)

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],  # Use validation dataset here
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=tokenizer,
#     optimizers=(optimizer,None) # use constant learning rate 
# )

# trainer.train()

# test_results_optimizer = trainer.evaluate(tokenized_datasets["test"])
# print("Test result using optimizer is:", test_results_optimizer)

def hyperparameter_optimization(trial):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-4)
    batch_size = trial.suggest_int('batch_size', 16, 32, step=16)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2)

    # Prepare the training arguments with the hyperparameters
    args = TrainingArguments(
        "bert-finetuned-ner",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=3, 
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size
    )

    # Define the model
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    # Prepare the Trainer with the datasets, optimizer, and evaluation metric
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Evaluate the model on the validation set
    eval_results = trainer.evaluate()

    # Use F1 score as the objective to minimize (since it's commonly used in NER tasks)
    return eval_results["eval_f1"]

# Create the study and start the optimization
study = optuna.create_study(direction="maximize")  # Maximize F1 score
study.optimize(hyperparameter_optimization, n_trials=5)  # Run 5 trials

# Get the best hyperparameters
best_trial = study.best_trial
print(f"Best trial: {best_trial.values}")

best_learning_rate = best_trial.params['learning_rate']
best_batch_size = best_trial.params['batch_size']
best_weight_decay = best_trial.params['weight_decay']

best_args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_learning_rate,
    num_train_epochs=3, 
    weight_decay=best_weight_decay,
    per_device_train_batch_size=best_batch_size,
    per_device_eval_batch_size=best_batch_size,
)

best_model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

best_params_trainer = Trainer(
    model=best_model,
    args=best_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

best_params_trainer.train()

test_results = best_params_trainer.evaluate(tokenized_datasets["test"]) 
print("Test result after optimization parameters are incorporated:", test_results)

ner_feature = dataset_dict["train"].features["NER_TAGS"]
label_names = sorted(unique_labels_train)

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


ner_tags = [
    ner_tag 
    for div in dataset_dict  
    for sentence in dataset_dict[div]  
    for ner_tag in sentence['NER_TAGS'] 
]

entity_types = [ner_tag.split('-')[-1] for ner_tag in ner_tags]
entity_types = list(set(entity_types))

def group_metrics_calculation(true_labels, true_predictions, entity_type):
    flat_true_labels= [label for sublist in true_labels for label in sublist]
    flat_true_predictions = [prediction for sublist in true_predictions for prediction in sublist]

    true_labels_b = [label for label in flat_true_labels if label.startswith(f"B-{entity_type}")]
    true_predictions_b = [prediction for prediction in flat_true_predictions if prediction.startswith(f"B-{entity_type}")]
    
    true_labels_i = [label for label in flat_true_labels if label.startswith(f"I-{entity_type}")]
    true_predictions_i = [prediction for prediction in flat_true_predictions if prediction.startswith(f"I-{entity_type}")]

    pad = 'N' # padding is different from O so that system does not get confused 

    max_length_b = max(len(true_labels_b), len(true_predictions_b))
    true_labels_b.extend([pad] * (max_length_b - len(true_labels_b)))
    true_predictions_b.extend([pad] * (max_length_b - len(true_predictions_b)))

    max_length_for_i = max(len(true_labels_i), len(true_predictions_i))
    true_labels_i.extend([pad]* (max_length_for_i - len(true_labels_i)))
    true_predictions_i.extend([pad] * (max_length_for_i - len(true_predictions_i)))
    
    precision_b_label, recall_b_label, f1_b_label, _ = precision_recall_fscore_support(
        true_labels_b, 
        true_predictions_b, 
        #average='None', 
        labels=[f'B-{entity_type}'],
        zero_division=0
    )

    precision_i_label, recall_i_label, f1_i_label, _ = precision_recall_fscore_support(
        true_labels_i, 
        true_predictions_i, 
        #average='None', 
        labels=[f'I-{entity_type}'],
        zero_division=0
    )

    return precision_b_label, recall_b_label, f1_b_label, precision_i_label, recall_i_label, f1_i_label

def entity_metrics_calculation(true_labels, true_predictions, entity_type):
    flat_true_labels= [label for sublist in true_labels for label in sublist]
    flat_true_predictions = [prediction for sublist in true_predictions for prediction in sublist]

    true_labels_entity = [label for label in flat_true_labels if label.startswith(f"{entity_type}-") and label != 'N']
    true_predictions_entity = [prediction for prediction in flat_true_predictions if prediction.startswith(f"{entity_type}-") and prediction != 'N']
    
    if not true_labels_entity or not true_predictions_entity:
        return [0.0], [0.0], [0.0]

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels_entity, 
        true_predictions_entity, 
        #average='None', 
        labels=[f'B-{entity_type}', f'I-{entity_type}'],
        zero_division=0
    )
    
    return precision, recall, f1
    

def updated_compute_metrics(eval_preds):
    logits, labels = eval_preds

    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    predictions = np.argmax(logits.detach().cpu().numpy(), axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100 and l < len(label_names)] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100 and p < len(label_names)]
        for prediction, label in zip(predictions, labels)
    ] # just converting the model prediction of numbers (2,3).. to corresponding name labels like (B-PER,I-PER)

    group_metrics = {"for_B": {}, "for_I": {}, "for_I_and_B": {}}
    entity_metrics = {}

    all_losses = []

    for entity in entity_types:
        precision_b_entity, recall_b_entity, f1_b_entity, precision_i_entity, recall_i_entity, f1_i_entity = group_metrics_calculation(true_labels,true_predictions,entity)
        precision_full_entity, recall_full_entity, f1_full_entity = entity_metrics_calculation(true_labels, true_predictions, entity)

        group_metrics["for_B"][entity] = {"precision": precision_b_entity, "recall": recall_b_entity, "f1": f1_b_entity}
        group_metrics["for_I"][entity] = {"precision": precision_i_entity, "recall": recall_i_entity, "f1": f1_i_entity}
        group_metrics["for_I_and_B"][entity] = {"precision": (precision_b_entity + precision_i_entity) / 2, "recall": (recall_b_entity + recall_i_entity) / 2, "f1": (f1_b_entity + f1_i_entity) / 2}
        
        entity_metrics[entity] = {"precision": precision_full_entity, "recall": recall_full_entity, "f1": f1_full_entity}

    if hasattr(logits, 'loss'):
        all_losses.append(logits.loss.item())

    if all_losses:
        all_losses_tensor = torch.tensor(all_losses)
        loss = all_losses_tensor.mean().item()  # Calculate the mean if there are losses
    else:
        loss = 0.0  # Default value if no losses were recorded


    group_metrics = {k: {entity: {key: value.tolist() if isinstance(value, np.ndarray) else value
                                  for key, value in v.items()} for entity, v in group_items.items()}
                     for k, group_items in group_metrics.items()}

    entity_metrics = {entity: {key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in metrics.items()} 
                     for entity, metrics in entity_metrics.items()}


    return {
        "group_metrics": group_metrics,
        "entity_metrics": entity_metrics,
        "loss": loss
    }

test_results2 = best_params_trainer.predict(tokenized_datasets["test"])
logits = test_results2.predictions
labels = test_results2.label_ids
group_metric_result = updated_compute_metrics((logits, labels))
print("Test results with group-level performance metric:")
print(group_metric_result)


def compute_macro_micro_f1(eval_preds):
    logits, labels = eval_preds
    
    predictions = np.argmax(logits, axis=-1)

    flat_true_labels = [
        label_names[l] for inner_list in labels for l in inner_list if l != -100 and l < len(label_names)
    ]
    flat_true_predictions = [
        label_names[p] for inner_list_pred, inner_list_label in zip(predictions, labels) 
        for p, l in zip(inner_list_pred, inner_list_label) if l != -100 and p < len(label_names)
    ]

    filter_labels = [label for label in label_names if label != "O"]

    _, _, f1_scores, _ = precision_recall_fscore_support(
        flat_true_labels, 
        flat_true_predictions, 
        labels=filter_labels, 
        average=None
    )
    macro_f1 = np.mean(f1_scores)

    _, _, f1_micro, _ = precision_recall_fscore_support(
        flat_true_labels, 
        flat_true_predictions, 
        labels=filter_labels, 
        average="micro"
    )

    print(f"Macro-average for F1 Score: {macro_f1:.4f}")
    print(f"Micro-average for F1 Score: {f1_micro:.4f}")

    return {"macro_f1": macro_f1, "micro_f1": f1_micro}

test_results2 = best_params_trainer.predict(tokenized_datasets["test"])
logits = test_results2.predictions
labels = test_results2.label_ids
micro_macro_result= compute_macro_micro_f1((logits, labels))
print("Test results with micro and macro:")
print(micro_macro_result)
