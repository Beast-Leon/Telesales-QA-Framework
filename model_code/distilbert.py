import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn as nn

# Add a softmax layer to the trained distilbert model
class disbert_arch(nn.Module):
    def __init__(self, model):
        super(disbert_arch, self).__init__()
        self.disbert = model
        self.softmax = nn.LogSoftmax(dim = 1)
    def forward(self, sent_id, mask):
        x = self.disbert(sent_id, attention_mask = mask).logits
        x = self.softmax(x)
        return x
    