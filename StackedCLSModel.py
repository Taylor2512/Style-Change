import re
import torch
import torch.nn as nn
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer
 
class StackedCLSModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.bert = DebertaModel.from_pretrained('microsoft/deberta-base')
        # self.Fusion = torch.nn.Parameter(torch.zeros(12, 1))
        # self.lin1 = torch.nn.Linear(768, 128)
        # self.lin2 = torch.nn.Linear(128, 2)

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
    
    def TokenizarParrafo(self,sequence1:str,secuence2:str,max_length):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        model = BertModel.from_pretrained('bert-base-uncased', config=config)
        # # tokenized_sequence = tokenizer.tokenize(sequence)
        # sequence1 = re.sub(r"[^a-zA-Záéíóú.,!?;:<>()$€\[\]]+", r" ", sequence1)   # reemplaza todas las coincidencias del patrón con un espacio
        # sequence2 = re.sub(r"[^a-zA-Záéíóú.,!?;:<>()$€\[\]]+", r" ", sequence2)   # reemplaza todas las coincidencias del patrón con un espacio

        indexed_tokens = tokenizer.encode(sequence1,secuence2,
          add_special_tokens=True,      # especifica si se agregan tokens especiales al principio y al final de la secuencia de tokens
      max_length=max_length,        # especifica la longitud máxima de la secuencia de tokens resultante
      padding='longest',            # especifica cómo rellenar secuencias más cortas a la misma longitud que la secuencia más larga.
      truncation=True,              # especifica si se truncan las secuencias que son más largas que "max_length"
      return_tensors='np'           # especifica que la salida debe devolverse como una matriz numpy (o pt?).
        )
        return indexed_tokens[0]
        
            
             
            
    def _recorrer_parrafos(self):
            print("La cantidad de párrafos es:", len(self.parrafos))
            for p in self.parrafos:
                self.TokenizarParrafo(p)
