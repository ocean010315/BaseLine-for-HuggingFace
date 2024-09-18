import argparse
from configurer import configurer
from torch.utils.data import DataLoader

from dataset import preprocess, Dataset
from model import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config.yaml", help="path of the yaml file")
    arg = parser.parse_args()

    config = configurer(arg.config)

    train_dataset = preprocess(config.train_path)
    valid_dataset = preprocess(config.valid_path)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)

    model = Model(model_name=config.model_name,
                  epoch=config.epoch,
                  train_data=train_dataloader,
                  valid_data=valid_dataloader,
                  lr=config.lr,
                  weight_decay=config.weight_decay)
    model.train()