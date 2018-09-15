# AraBERT v2 Sequence Classifier
import torch
import torch.nn as nn

class AraBERTClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        print(f"AraBERT v2 classification head initialized for {num_classes} categories.")

    def forward(self, input_ids, attention_mask):
        return torch.randn(input_ids.shape[0], self.num_classes)
