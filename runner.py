/ src/inference/runner.py
import torch
from pathlib import Path
from src.model.infer import InferenceModel

class InferenceRunner:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
        self.model = InferenceModel.load_from_path(model_path, device=device)

    def run(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            return self.model.infer(input_tensor)

if __name__ == "__main__":
    import numpy as np
    runner = InferenceRunner("/path/to/model.pt", device="cuda:0")
    dummy = torch.from_numpy(np.random.randn(1, 3, 224, 224).astype("float32"))
    out = runner.run(dummy)
    print(out.shape)
