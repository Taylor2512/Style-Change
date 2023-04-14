import torch
import torch.nn as nn
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer
 
class StackedCLSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DebertaModel.from_pretrained('microsoft/deberta-base')
        self.Fusion = torch.nn.Parameter(torch.zeros(12, 1))
        self.lin1 = torch.nn.Linear(768, 128)
        self.lin2 = torch.nn.Linear(128, 2)

    def forward(self, input_ids, input_masks, targets):
        output = self.bert(input_ids, input_masks)
        cls_tensors = torch.stack([output[2][n][0, 0] for n in range(1, 13)])
        t_cls_tensors = cls_tensors.transpose(1, 0)
        pooled_layers = torch.mm(t_cls_tensors, self.Fusion).squeeze()
        x = self.lin1(pooled_layers)
        x = torch.nn.Dropout(0.3)(x)
        x = torch.nn.tanh(x)  # o torch.nn.ReLU(x)
        logits = self.lin2(x)

        loss = None
        if targets is not None:
            loss = torch.nn.BCEWithLogitsLoss()(logits, targets)

        return logits, loss

    def predict(self, input_ids, input_masks):
        logits = self.forward(input_ids, input_masks, targets=None)
        return (logits.sigmoid() > 0.5)
    
    def TokenizarParrafo(self,sequence):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        model = BertModel.from_pretrained('bert-base-uncased', config=config)
        tokenized_sequence = tokenizer.tokenize(sequence)
        indexed_tokens = tokenizer.encode(tokenized_sequence, largPading=largePading)
        outputs = model(indexed_tokens)
        print(len(outputs))  # 4 Bert da una tupla de 4 elementos
        print(outputs[0].shape)  # 1, 16, 768 son los embedding de los últimos tokens
        print(outputs[1].shape)  # 1, 768 del token CLS
        print(len(outputs[2]))  # 13  = input embedding (index 0) + 12 hidden layers (indices 1 to 12)
        print(outputs[2][0].shape)  # for each of these 13: 1,16,768 = input sequence, index of each input id in sequence, size of hidden layer
        print(len(outputs[3]))  # 12 (=attention for each layer)
        print(outputs[3][0].shape)  # 0 index = first layer, 1,12,16,16 = , layer, index of each input id in sequence, index of each input id in sequence
        outsuma=   outputs[1].shape
        print(outsuma) # for each of these
        outsuma= outputs[1].sum()
        print(outsuma) # for each of these
        outsuma= outputs[0][:, 0].shape
        print(outsuma) # for each of these
        outsuma= outputs[0][:, 0].sum()
        print(outsuma) # for each of these
        outsuma= outputs[2][5][:, 0]
        print(outsuma) # for each of these
  
        cls_tensors = []
        for i in range(1,13):
               cls_tensors.append(outputs[2][1][0,0])
        
        imrpimir= cls_stack = torch.stack(cls_tensors)
        print(imrpimir) # for each of these
        imrpimir= cls_stack.shape
        print(imrpimir) # for each of these
        imrpimir= t_cls_stack = cls_stack.transpose(1,0)
        print(imrpimir) # for each of these
        imrpimir= F = torch.zeros(12,1)
        print(imrpimir) # for each of these
        imrpimir= t_cls_stack.shape
        print(imrpimir) # for each of these
        imrpimir= F.shape
        print(imrpimir) # for each of these
        imrpimir= torch.mm(t_cls_stack, F).squeeze()
        print(imrpimir) # for each of these
            
             
            
    def _recorrer_parrafos(self):
            print("La cantidad de párrafos es:", len(self.parrafos))
            for p in self.parrafos:
                self.TokenizarParrafo(p)
