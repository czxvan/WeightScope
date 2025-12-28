import torch.nn as nn

class ModelWalker:
    def __init__(self, model):
        self.model = model

    def walk(self, types=None):
        """
        Yields (name, module) for layers matching the specified types.
        Default types include Linear, Conv2d, Embedding, and Conv1D (GPT-2 style) layers.
        """
        if types is None:
            # Include common weight-bearing layers
            types = (nn.Linear, nn.Conv2d, nn.Embedding)
            
            # Also check for GPT-2 style Conv1D
            try:
                from transformers.pytorch_utils import Conv1D
                types = types + (Conv1D,)
            except:
                pass
        
        for name, module in self.model.named_modules():
            if isinstance(module, types):
                yield name, module
    
    def get_all_modules(self):
        """
        Returns a list of all module names and types for debugging.
        """
        return [(name, type(module).__name__) for name, module in self.model.named_modules()]
