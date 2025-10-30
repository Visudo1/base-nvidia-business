/src/model/infer.py
import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.seq(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class InferenceModel:
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

    @staticmethod
    def load_from_path(path: str, device: str = "cuda"):
        # load state_dict jika ada, contoh pakai SimpleModel
        m = SimpleModel()
        # misal: m.load_state_dict(torch.load(path, map_location=device))
        return InferenceModel(m, device)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
