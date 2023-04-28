import ast
from pickletools import optimize
import re
from sklearn import datasets
from sklearn.preprocessing import binarize
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer,DebertaTokenizer,DebertaConfig
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
 
class StackedCLSModel(nn.Module):
    def __init__(self, model=None, model_type=None,tokenizer=None):
        super(StackedCLSModel, self).__init__()
        self.model=model
        self.tokenizer=tokenizer
        self.model_type=model_type
        self.Fusion = nn.Parameter(torch.zeros(12,1))
        self.lin1 = nn.Linear(768, 128)
        self.lin2 = nn.Linear(128, 2)
        model = nn.Linear(10, 2) 
    
    def forward(self, input_ids, input_masks,targets):
    #output = self.bert(input_ids, input_masks)
     outputs = self.model(input_ids, input_masks)
     if self.model_type == "bert":
         cls_tensors = torch.stack([outputs[2][n][0,0] for n in range(1,13)]) #ojo cómo leo la columna outputs
     elif self.model_type == "deberta":
      cls_tensors = torch.stack([outputs[1][n][0,0] for n in range(1,13)]) #ojo cómo leo la columna outputs
    
     t_cls_tensors = cls_tensors.transpose(1,0)
     pooled_layers = torch.mm(t_cls_tensors, self.Fusion).squeeze()
     #pooled_layers = nn.functional.linear(t_cls_tensors, self.Fusion).squeeze()
     x = self.lin1(pooled_layers)
      #x = torch.nn.Dropout(0.3)
      #x = torch.nn.tanh(x) # o torch.nn.ReLU(x)
     x = nn.Dropout(0.3)(x)
     x = nn.Tanh()(x) # o torch.nn.ReLU(x)
     logits = self.lin2(x)
     loss = None
     #if targets:
          #loss = torch.nn.BCEWithLogitsLoss(logits, targets)
     loss_fn = nn.CrossEntropyLoss()
     loss = loss_fn(logits, targets.float())
#ojo xq los labels los hace multiclase, ya no suso el BCE
     print("loss:", loss)
     # Realizar un paso de entrenamiento
     perdida=loss.item()
     print('Pérdida:', loss.item())

     return logits, loss

 
    
    def TokenizarParrafo(self,sequence1:str,sequence2:str,max_length):
        sequence1 = re.sub(r"[^a-zA-Záéíóú.,!?;:<>()$€\[\]]+", r" ", sequence1)   # reemplaza todas las coincidencias del patrón con un espacio
        sequence2 = re.sub(r"[^a-zA-Záéíóú.,!?;:<>()$€\[\]]+", r" ", sequence2)   # reemplaza todas las coincidencias del patrón con un espacio

        if self.model_type=='bert':
            
            encoded_dict = self.tokenizer(sequence1, sequence2, add_special_tokens = True,  # Agregar los tokens especiales
                        max_length = 512,           # Longitud máxima de la secuencia de entrada
                        padding = 'max_length',     # Rellenar la secuencia de entrada hasta la longitud máxima
                        return_attention_mask = True,# Obtener la máscara de atención
                        return_tensors = 'pt'       # Devolver tensores de PyTorch
                   )
            return encoded_dict
        elif self.model_type=='deberta':
            encoded_dict = self.tokenizer(sequence1, sequence2, 
                         padding= 'max_length', 
                         truncation=True, 
                         max_length=512, 
                         return_tensors='pt')
 

            
             
            
    def _recorrer_parrafos(self):
            print("La cantidad de párrafos es:", len(self.parrafos))
            for p in self.parrafos:
                self.TokenizarParrafo(p)

        
    def predict(self, input_ids, input_masks,targets=None):
          logits = self.forward(input_ids, input_masks, targets)
          #return (logits.sigmoid() > 0.5) * 1
          return (logits.argmax() > 0.5) * 1
    def binarize(y ,threshold=0.5, triple_valued=False):
        y = np.array(y)
        y = np.ma.fix_invalid(y, fill_value=threshold)
        if triple_valued:
            y[y > threshold] = 1
        else:
            y[y >= threshold] = 1
            y[y < threshold] = 0
        
        return y
    
            
    def __len__(self):    
        return self.len   # devuelve la longitud del conjunto de datos personalizado
    
     
    def f1(true_y, pred_y):
        true_y_filtered, pred_y_filtered = [], []
        for true, pred in zip(true_y, pred_y):
            if pred != 0.5:
                true_y_filtered.append(true)
                pred_y_filtered.append(pred)
        pred_y_filtered = binarize(pred_y_filtered)
        return f1_score(true_y_filtered, pred_y_filtered)


        
    
