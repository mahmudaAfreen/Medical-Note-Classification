import json
import random
from typing import Optional
import csv
#import lightning.pytorch as pl
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertTokenizerFast
import numpy as np
import ast
import pandas as pd
import pickle

class ClassifcationCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        task: str,
        max_seq_len: int = 512,
        all_examples_with_null: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.all_examples_with_null = all_examples_with_null
        self.task = task

    def __call__(self, data):
        admission_notes = [x["text"] for x in data]

        labels = torch.stack([x["labels"] for x in data])


        tokenized = self.tokenizer(
            admission_notes,
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
        )
        input_ids = [torch.tensor(x) for x in tokenized["input_ids"]]
        attention_masks = [
            torch.tensor(x, dtype=torch.bool) for x in tokenized["attention_mask"]
        ]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
       
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "note": admission_notes, 
        }


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, label_lookup, sampling_strategy: str = "random"):
        # tokenize admission notes
        #self.task = task
        self.examples = examples
        self.label_lookup = label_lookup
        self.inverse_label_lookup = {v: k for k, v in label_lookup.items()}
        self.sampling_strategy = sampling_strategy
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        print(f"Loading item {idx}")
        example = self.examples.iloc[idx]
        note = example["note"]
        labels = example["labels"]
   
        label_arr = torch.zeros(len(self.label_lookup), dtype=torch.float32)
        label_arr[self.label_lookup[labels]] = 1
        return {
            "text": note,
            "labels": label_arr,
        }


class MIMICClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        use_code_descriptions: bool = False,
        #data_dir: str = "D:/Work/MedSum/classesclassification",
        #task: str = "dia",
        batch_size: int = 32,
        eval_batch_size: int = 16,
        tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        num_workers: int = 4,
        sampling_strategy: str = "random",
        val_sampling_strategy: str = "random",
        max_seq_len: int = 512,
        test_file=None,
        train_file=None,
        val_file=None,
        all_labels_path=None,
    ):
    
        super().__init__()
        
        """"
        test_data = pd.read_csv(data_dir + "/test.csv").rename(
            columns={"classes": "labels"}
        )
        train_data = pd.read_csv(data_dir  + "/train.csv").rename(
            columns={"classes": "labels"}
        )
        val_data = pd.read_csv(data_dir  + "/valid.csv").rename(
            columns={"classes": "labels"}
        )
        """
        
        test_data = pd.read_csv("/pvc/sentences_test.csv").rename(
            columns={"classes": "labels"}
        )
        train_data = pd.read_csv("/pvc/sentences_train.csv").rename(
            columns={"classes": "labels"}
        )
        val_data = pd.read_csv("/pvc/sentences_val.csv").rename(
            columns={"classes": "labels"}
        )
        
        train_counts = train_data['labels'].value_counts()
        val_counts = val_data['labels'].value_counts()
        test_counts = test_data['labels'].value_counts()

        #print("Training set label distribution:\n", train_counts)
        #print("Validation set label distribution:\n", val_counts)
        #print("Test set label distribution:\n", test_counts)

        self.training_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        #all_labels = list(train_data.labels.unique())
        all_labels = list(set(train_data["labels"].unique()).union(val_data["labels"].unique()).union(test_data["labels"].unique()))
        all_labels.sort()
        label_idx = {v: k for k, v in enumerate(all_labels)}
        
        #NEW
                # Save index -> class name mapping so the model can print human-readable names
        import json
        index_to_label = {idx: label for label, idx in label_idx.items()}
        with open("/pvc/label_mapping.json", "w") as f:
            json.dump(index_to_label, f)

        
        # build label index
        self.label_idx = label_idx
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        #self.collator = ClassifcationCollator(self.tokenizer, max_seq_len)
        self.collator = ClassifcationCollator(
            tokenizer=self.tokenizer,
            task="classification",
            max_seq_len=max_seq_len,
                            )

        self.num_workers = num_workers
        #self.task = task
        self.sampling_strategy = sampling_strategy
        self.val_sampling_strategy = val_sampling_strategy

    def setup(self, stage: Optional[str] = None):
        mimic_train = ClassificationDataset(
            self.training_data,
            label_lookup=self.label_idx,
            sampling_strategy=self.sampling_strategy,
            #task=self.task,
        )
        
        mimic_val = ClassificationDataset(
            self.val_data,
            label_lookup=self.label_idx,
            sampling_strategy=self.val_sampling_strategy,
            #task=self.task,
        )

        mimic_test = ClassificationDataset(
            self.test_data,
            label_lookup=self.label_idx,
            sampling_strategy=self.val_sampling_strategy,
            #task=self.task,
        )
        self.mimic_train = mimic_train
        self.mimic_val = mimic_val
        self.mimic_test = mimic_test
        print("Val length: ", len(self.mimic_val))
        print("Train Length: ", len(self.mimic_train))

    def train_dataloader(self):
        return DataLoader(
            self.mimic_train,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mimic_val,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mimic_test,
            batch_size=self.eval_batch_size,
            collate_fn=self.collator,
            #pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
