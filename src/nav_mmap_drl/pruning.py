import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class ModelPruner:
    def __init__(self, model):
             
            
        self.model = model

    def prune_model(self, amount=0.2):
        
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=0)
                print(f"Pruned {amount*100:.1f}% of {layer} weights.")
        return self.model

    def remove_pruning(self):
        
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                prune.remove(layer, 'weight')
                print(f"Removed pruning from {layer}.")

    def print_pruning_summary(self):
        
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                pruning_amount = 1.0 - torch.count_nonzero(layer.weight) / layer.weight.numel()
                print(f"{layer}: {pruning_amount*100:.1f}% pruned.")

 
   
