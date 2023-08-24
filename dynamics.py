import os
import re
from typing import List, Optional

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)


def log_training_dynamics(output_dir: os.path,
                          epoch: int,
                          train_ids: List[int],
                          train_logits: List[List[float]],
                          train_golds: List[int]):
  """
  Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
  """
  td_df = pd.DataFrame({"guid": train_ids,
                        f"logits_epoch_{epoch}": train_logits,
                        "gold": train_golds})

  logging_dir = os.path.join(output_dir, f"training_dynamics")
  # Create directory for logging training dynamics, if it doesn't already exist.
  if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
  epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
  td_df.to_json(epoch_file_name, lines=True, orient="records")

def evaluate_model(model, train_dataloader, output_dir, device, epoch: int):
    input_ids = torch.empty(0).to(device)
    gold_labels = torch.empty(0).to(device)
    logits = torch.empty(0).to(device)

    #set model into evaluation mode
    model.eval()
    for batch in tqdm(train_dataloader):
        #transfer values to device
        batch = {k: v.to(device) for k, v in batch.items()}

        tmp_input_ids = batch["id"]
        tmp_gold_labels = batch["labels"]


        with torch.no_grad():
            outputs = model(labels = batch["labels"], input_ids=batch["input_ids"], attention_mask = batch["attention_mask"])
                
        tmp_logits = outputs.logits
        
        assert(len(tmp_input_ids) == len(tmp_gold_labels) == len(tmp_logits))
        
        input_ids = torch.cat((input_ids, tmp_input_ids), dim=0)
        gold_labels = torch.cat((gold_labels, tmp_gold_labels), dim=0)
        logits = torch.cat((logits, tmp_logits), dim=0)


    input_ids.type(torch.int32)
    gold_labels.type(torch.uint8)
        
    try:
        log_training_dynamics(output_dir=output_dir,
                                epoch=epoch,
                                train_ids=input_ids.detach().cpu().tolist(),
                                train_logits=logits.detach().cpu().tolist(),
                                train_golds=gold_labels.detach().cpu().tolist())
    except Exception as e:
        print(f"error in log_training dynamics: {e}")

def compute_dynamics(model_directory: str, dataset: str, dataset_subset: Optional[str]=None):
    checkpoint_dirs = sorted([os.path.join(model_directory, x) for x in filter(lambda x: x.startswith("checkpoint-"), os.listdir(model_directory))], key=lambda x: int(re.search("[0-9]+", x)[0]))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dirs[0])
    #tokenization function
    def tokenize_function(example):
        return tokenizer(example["claim"], example["evidence"], truncation=True)

    #tokenize dataset
    if dataset_subset is not None:
        dataset = load_dataset(dataset, dataset_subset)
    else:
        dataset = load_dataset(dataset)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns = ["claim", "evidence"], num_proc=4)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=False, batch_size=32, collate_fn=data_collator)

    # check gpu availability
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for epoch, checkpoint_dir in enumerate(checkpoint_dirs):
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, num_labels=3)
        model.to(device)
        evaluate_model(model=model, train_dataloader=train_dataloader, output_dir=model_directory, device=device, epoch=epoch)

MODEL_DIR = "/home/mlynatom/models/xlm-roberta-large-squad2-csfever_v2-original_nli-batchsize-9-wr0.4-steps3000"

compute_dynamics(model_directory=MODEL_DIR, dataset="ctu-aic/csfever_v2", dataset_subset="original_nli")