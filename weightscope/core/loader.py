import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoModel
import os

class ModelLoader:
    def __init__(self, model_name_or_path, device="cpu", model_type="auto"):
        self.model_name = model_name_or_path
        self.device = device
        self.model_type = model_type
        self.model = self._load_model()

    def _load_model(self):
        print(f"Loading model from {self.model_name} via ModelScope...")
        try:
            # Download model from ModelScope if needed
            if not os.path.exists(self.model_name):
                model_dir = snapshot_download(self.model_name)
            else:
                model_dir = self.model_name
            
            # Load the full causal LM model to get all layers
            print(f"Loading model from local path: {model_dir}")
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=self.device
            )
            return model
        except Exception as e:
            print(f"Error loading model as CausalLM, trying AutoModel: {e}")
            try:
                model = AutoModel.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    device_map=self.device
                )
                return model
            except Exception as e2:
                print(f"Error loading model via AutoModel: {e2}")
                raise e2
