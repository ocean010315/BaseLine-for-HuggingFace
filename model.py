import numpy as np
from scipy.stats import pearsonr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from torch import optim


class Model():
    def __init__(self, model_name, epoch, train_data, valid_data, lr, weight_decay):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        self.training_arguments = TrainingArguments(
            output_dir='./results/',
            overwrite_output_dir=True,
            num_train_epochs=epoch,
            learning_rate=lr,
            weight_decay=weight_decay,
            eval_strategy="epoch",
        )

        # optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # schedular = optim.lr_schedular.StepLR(optimizer, step_size=1, gamma=0.1)

        # data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=train_data,
            eval_dataset=valid_data,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            # data_collator=data_collator
            # optimizers=(optimizer, schedular), # default = AdamW
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        
        predictions = np.argmax(logits, axis=-1)
        labels = np.array(labels)

        pearson_corr, _ = pearsonr(predictions, labels)

        return {"pearson_corr": pearson_corr}

    def train(self):
        self.trainer.train()