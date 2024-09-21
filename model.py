import numpy as np
from scipy.stats import pearsonr
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


class Model():
    def __init__(self, model_name, epoch, train_data, valid_data, batch_size, lr, weight_decay):
        self.model_name = model_name

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=1)
    
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
