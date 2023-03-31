import torch
import torch.nn as nn
from transformers import DebertaModel
class StackedCLSModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert = DebertaModel.from_pretrained('microsoft/deberta-base')
    self.Fusion = torch.nn.Parameter(torch.zeros(12,1))
    self.lin1 = torch.nn.Linear(768, 128)
    self.lin2 = torch.nn.Linear(128, 2)

  def forward(self, input_ids, input_masks, targets):
    output = self.bert(input_ids, input_masks)
    cls_tensors = torch.stack([output[2][n][0,0] for n in range(1,13)])
    t_cls_tensors = cls_tensors.transpose(1,0)
    pooled_layers = torch.mm(t_cls_tensors, self.Fusion).squeeze()
    x = self.lin1(pooled_layers)
    x = torch.nn.Dropout(0.3)(x)
    x = torch.nn.tanh(x) # o torch.nn.ReLU(x)
    logits = self.lin2(x)

    loss = None
    if targets is not None:
      loss = torch.nn.BCEWithLogitsLoss()(logits, targets)

    return logits, loss

  def predict(self, input_ids, input_masks):
    logits = self.forward(input_ids, input_masks, targets=None)
    return (logits.sigmoid() > 0.5)
     
