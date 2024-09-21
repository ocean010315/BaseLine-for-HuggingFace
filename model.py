import numpy as np
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollator
from torch import optim
import torch
from pandas import read_csv

from dataset import preprocess


class Model():
    def __init__(self, model_name, epoch, train_data, valid_data, batch_size, lr, weight_decay):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
        self.training_arguments = TrainingArguments(
            output_dir='./results/',
            overwrite_output_dir=True,
            num_train_epochs=epoch,
            learning_rate=lr,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            lr_scheduler_type='linear', # linear, cosine, constant, etc.
        )

        # optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=train_data,
            eval_dataset=valid_data,
            compute_metrics=self.compute_metrics,
            # optimizers=(optimizer, scheduler), # default = AdamW
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        
        predictions = logits.squeeze()
        labels = np.array(labels)

        pearson_corr, _ = pearsonr(predictions, labels)

        return {"pearson_corr": pearson_corr}

    def train(self):
        self.trainer.train()

        test_data = preprocess(task="test", data_path='../../test.csv', model_name='klue/roberta-small')
        
        predictions = self.trainer.predict(test_data)
        print(predictions)
        # logits = [round(float(i), 1) for i in torch.cat(logits)]

        # output_csv = read_csv('../../sample_submission.csv')
        # output_csv['target'] = logits
        # output_csv.to_csv('sample_output_v1.csv', index=False)
