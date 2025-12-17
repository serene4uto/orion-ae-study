
import torch


from pickletools import optimize


class Trainer:
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = config["device"]
        self.optimizer_cfg = config["optimizer"]


    # def train(self):

    #     for batch in self.train_loader:

    #         inputs =  batch['preprocessed'].to(self.device)
    #         labels = batch['label'].to(self.device)

            
            
        

    def val(self):
        pass
    

