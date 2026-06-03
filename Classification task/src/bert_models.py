from typing import Optional, Dict, Any, List, Union
import os
import csv

#from custom_metrics import PR_AUC
#import lightning.pytorch as pl
import pytorch_lightning as pl
import torch
import torchmetrics
from torchmetrics.functional.classification.auroc import _multilabel_auroc_compute
from torchmetrics.functional.classification import auroc as auroc_f
from torchmetrics.functional.classification import average_precision, accuracy, f1_score, precision, recall
import transformers
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, BertModel, AutoModel
from torchmetrics.functional.retrieval import retrieval_recall, retrieval_precision
import torch.nn.functional as F
import json
from torchmetrics.functional import confusion_matrix
import pandas as pd
#from sentence_transformers import SentenceTransformer
class BertClassificationModel(pl.LightningModule):
    def __init__(self,
                 num_classes: int = 19,
                 encoder_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 warmup_steps: int = 0,
                 decay_steps: int = 50_000,
                 num_training_steps: int = 50_000,
                 weight_decay: float = 0.01,
                 lr: float = 2e-5,
                 optimizer_name="adam",
                 #task: str = "dia",
                 ):
        
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        #self.encoder.pooler = None
        if hasattr(self.encoder, "pooler"):
            self.encoder.pooler = None

        #self.task = task
        self.num_classes = num_classes
        #NEW
     # Will be loaded lazily from /pvc/label_mapping.json
        self.class_names = None

        #self.classification_layer = torch.nn.Linear(768, num_classes)
        hidden_size = self.encoder.config.hidden_size
        self.classification_layer = torch.nn.Linear(hidden_size, num_classes)
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.num_training_steps = num_training_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.auroc = None
        self.mean_precision = None
        self.val_output_list = []
        self.test_output_list = []
        self.train_output_list = []
        self.running_allocated_memory = 0
        self.running_reserved_memory = 0
       
        
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)

        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

    def forward(self,
                input_ids,
                attention_mask):
        encoded = self.encode(input_ids, attention_mask)
        #return self.classification_layer(encoded)

        logits = self.classification_layer(encoded)
        return logits
    def training_step(self, batch, batch_idx):

        logits = self(batch['input_ids'], batch['attention_mask'])
        _, labels = torch.max(batch['labels'], dim=1)
        loss = F.cross_entropy(logits, labels)
        self.log("Train/Loss", loss)
        return loss

#    def on_train_epoch_end(self) -> None:
#        torch.cuda.empty_cache()


    def test_step(self, batch, batch_idx, **kwargs):

        logits = self(batch['input_ids'], batch['attention_mask'])


        self.test_output_list.append(
            {"logits": logits, "labels": batch["labels"]})

        return {"logits": logits,
                "labels": batch["labels"], }

    def on_test_epoch_end(self) -> None:
        logits = torch.cat([x["logits"] for x in self.test_output_list])
        labels = torch.cat([x["labels"] for x in self.test_output_list]).int()
        _, labels = torch.max(labels, dim=1)

        # predicted class indices
        preds = torch.argmax(logits, dim=1)

        # -------- Load class names once --------
        if self.class_names is None:
            with open("/pvc/label_mapping.json", "r") as f:
                index_to_label = json.load(f)
            self.class_names = [
                index_to_label[str(i)] if str(i) in index_to_label else f"Class_{i}"
                for i in range(self.num_classes)
            ]

        # -------- Per-class metrics --------
        per_class_precision = precision(
            preds,
            labels,
            task="multiclass",
            num_classes=self.num_classes,
            average=None
        )

        per_class_recall = recall(
            preds,
            labels,
            task="multiclass",
            num_classes=self.num_classes,
            average=None
        )

        per_class_f1 = f1_score(
            preds,
            labels,
            task="multiclass",
            num_classes=self.num_classes,
            average=None
        )

        # Support = number of true samples per class
        per_class_support = torch.bincount(
            labels,
            minlength=self.num_classes
        )

        # -------- Save to CSV (PVC-safe) --------
        out_dir = self.trainer.default_root_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "per_class_metrics_test.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "class_id",
                "class_name",
                "precision",
                "recall",
                "f1_score",
                "support"
            ])
            for i, name in enumerate(self.class_names):
                writer.writerow([
                    i,
                    name,
                    float(per_class_precision[i]),
                    float(per_class_recall[i]),
                    float(per_class_f1[i]),
                    int(per_class_support[i])
                ])
            print(f"\nSaved per-class metrics to:\n{csv_path}\n")

# ================= CONFUSION MATRIX (TEST) =================

        cm = confusion_matrix(
            preds,
            labels,
            task="multiclass",
            num_classes=self.num_classes
        )

        # Load class names (already available above, reuse)
        cm_df = pd.DataFrame(
            cm.cpu().numpy(),
            index=[f"True_{c}" for c in self.class_names],
            columns=[f"Pred_{c}" for c in self.class_names],
        )

        # Save confusion matrix to CSV
        cm_path = os.path.join(out_dir, "confusion_matrix_test.csv")
        cm_df.to_csv(cm_path)

        print("\nConfusion matrix saved to:")
        print(cm_path)


        auroc_ = auroc_f(logits, labels, num_classes=self.num_classes, 
                                    task="multiclass", average='macro')
       
        mean_precision = average_precision(logits, labels, num_classes=self.num_classes, 
                                    task="multiclass", average='macro')
        accuracy_ = accuracy(logits, labels, task='multiclass', num_classes=self.num_classes,
                                average='macro') 
        f1_score_ = f1_score(logits, labels, task='multiclass', num_classes=self.num_classes, average='macro')
        
        precision_ = precision(logits, labels, task='multiclass', num_classes=self.num_classes, average='macro')
        
        recall_ = recall(logits, labels, task='multiclass', num_classes=self.num_classes, average='macro')

        loss = F.cross_entropy(logits, labels)


        self.log("test/loss", loss)
        self.log("test/AUROC", auroc_.mean())
        self.log("test/A_PREC", mean_precision.mean())
        self.log("test/BalancedAcc", accuracy_)
        self.log("test/F1_Score", f1_score_)
        self.log("test/Precision", precision_)
        self.log("test/Recall", recall_)
        self.test_output_list.clear()

    def validation_step(self, batch, batch_idx, **kwargs):
        logits = self(batch['input_ids'], batch['attention_mask'])

        self.val_output_list.append(
            {"logits": logits, "labels": batch["labels"]})

        return {"logits": logits,
                "labels": batch["labels"], }
    
    def on_validation_epoch_end(self) -> None:
        logits = torch.cat([x["logits"] for x in self.val_output_list])
        labels = torch.cat([x["labels"] for x in self.val_output_list]).int()

        
        _, labels = torch.max(labels, dim=1)
        auroc_ = auroc_f(logits, labels, num_classes=self.num_classes, 
                                    task="multiclass", average='macro')
       
        mean_precision = average_precision(logits, labels, num_classes=self.num_classes, 
                                    task="multiclass", average='macro')
        accuracy_ = accuracy(logits, labels, task='multiclass', num_classes=self.num_classes,
                                average='macro') 
        f1_score_ = f1_score(logits, labels, task='multiclass', num_classes=self.num_classes, average='macro')
        
        precision_ = precision(logits, labels, task='multiclass', num_classes=self.num_classes, average='macro')
        
        recall_ = recall(logits, labels, task='multiclass', num_classes=self.num_classes, average='macro')

        loss = F.cross_entropy(logits, labels)
        
    
        print(f"F1 Score: {f1_score_}, Precision: {precision_}, Recall: {recall_}, Loss: {loss}")
        self.log("Val/loss", loss)
        self.log("Val/AUROC", auroc_.mean())
        self.log("Val/A_PREC", mean_precision)
        self.log("Val/BalancedAcc", accuracy_)
        self.log("Val/F1_Score", f1_score_, prog_bar=True)
        self.log("Val/Precision", precision_, prog_bar=True)
        self.log("Val/Recall", recall_, prog_bar=True)
        self.val_output_list.clear()
        """
        current_allocated = torch.cuda.memory_allocated(self.device)
        current_reserved = torch.cuda.memory_reserved(self.device)
        if self.current_epoch % 4 == 0 and self.current_epoch > 3:
            assert (current_allocated == self.running_allocated_memory)
            assert (current_reserved == self.running_reserved_memory)
        self.running_allocated_memory = current_allocated
        self.running_reserved_memory = current_reserved
        """
    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        weight_decay = 0.01
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                weight_decay
        }, {
            'params':
                [p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr)

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, self.warmup_steps, num_training_steps=self.num_training_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
        }

        return [optimizer], [scheduler]
