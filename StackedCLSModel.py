import ast
import re
from sklearn import datasets
from sklearn.preprocessing import binarize
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer,DebertaTokenizer,DebertaConfig
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
config = DebertaConfig.from_pretrained("microsoft/deberta-base", output_hidden_states=True, output_attentions=True)
MODEL_TYPE = 'bert'
MODEL = DebertaModel.from_pretrained("microsoft/deberta-base", config=config)
class StackedCLSModel(nn.Module):
    def __init__(self, model= MODEL, model_type=MODEL_TYPE):
        super(StackedCLSModel, self).__init__()
        self.model=model
        self.model_type=model_type
        self.Fusion = nn.Parameter(torch.zeros(12,1))
        self.lin1 = nn.Linear(768, 128)
        self.lin2 = nn.Linear(128, 2)
    
    def forward(self, input_ids, input_masks, targets):
    #output = self.bert(input_ids, input_masks)
     outputs = self.model(input_ids, input_masks,targets)
     if self.model_type == "bert":
         cls_tensors = torch.stack([outputs[2][n][0,0] for n in range(1,13)]) #ojo cómo leo la columna outputs
     elif self.model_type == "deberta":
      cls_tensors = torch.stack([outputs[1][n][0,0] for n in range(1,13)]) #ojo cómo leo la columna outputs
      t_cls_tensors = cls_tensors.transpose(1,0)
    #pooled_layers = torch.nn(t_cls_tensors, self.Fusion).squeeze()
      pooled_layers = nn(t_cls_tensors, self.Fusion).squeeze()
      x = self.lin1(pooled_layers)
    #x = torch.nn.Dropout(0.3)
    #x = torch.nn.tanh(x) # o torch.nn.ReLU(x)
      x = nn.Dropout(0.3)
      x = nn.tanh(x) # o torch.nn.ReLU(x)
      logits = self.lin2(x)
      loss = None
      if targets:
      #loss = torch.nn.BCEWithLogitsLoss(logits, targets)
       loss = nn.CrossEntropyLoss(logits, targets)  #ojo xq los labels los hace multiclase, ya no suso el BCE
      print("loss:", loss)
      return logits, loss

 
    
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

        
    def predict(self, input_ids, input_masks,targets):
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
    
    
    def forward(self, input_ids, input_masks, targets):
    #output = self.bert(input_ids, input_masks)
     outputs = self.model(input_ids, input_masks,targets)
     if self.model_type == "bert":
         cls_tensors = torch.stack([outputs[2][n][0,0] for n in range(1,13)]) #ojo cómo leo la columna outputs
     elif self.model_type == "deberta":
      cls_tensors = torch.stack([outputs[1][n][0,0] for n in range(1,13)]) #ojo cómo leo la columna outputs
      t_cls_tensors = cls_tensors.transpose(1,0)
    #pooled_layers = torch.nn(t_cls_tensors, self.Fusion).squeeze()
      pooled_layers = nn(t_cls_tensors, self.Fusion).squeeze()
      x = self.lin1(pooled_layers)
    #x = torch.nn.Dropout(0.3)
    #x = torch.nn.tanh(x) # o torch.nn.ReLU(x)
      x = nn.Dropout(0.3)
      x = nn.tanh(x) # o torch.nn.ReLU(x)
      logits = self.lin2(x)
      loss = None
      if targets:
      #loss = torch.nn.BCEWithLogitsLoss(logits, targets)
       loss = nn.CrossEntropyLoss(logits, targets)  #ojo xq los labels los hace multiclase, ya no suso el BCE
      print("loss:", loss)
      return logits, loss
  
  
    def f1(true_y, pred_y):
        true_y_filtered, pred_y_filtered = [], []
        for true, pred in zip(true_y, pred_y):
            if pred != 0.5:
                true_y_filtered.append(true)
                pred_y_filtered.append(pred)
        pred_y_filtered = binarize(pred_y_filtered)
        return f1_score(true_y_filtered, pred_y_filtered)



class MyDataset(Dataset):             # define una nueva clase MyDataset que hereda de Dataset 
    def __init__(self, dataframe):    # define el constructor  "__init__"  que toma un solo argumento dataframe
        #print(dataframe)
        self.len = len(dataframe)   # calcula la longitud de la entrada dataframe usando la funcion "len" y la almacena como una variable de instancia "self.len"
        self.data = dataframe       # se asigna la entrada dataframe a una variable de instancia "self.data"
        
    def __getitem__(self, index):   # define el método "__getitem__" que toma un solo argumento index
        ''' el metodo __getitem__ devuelve un diccionario que contiene cuatro claves: 'input_ids', 'attention_mask', 'labels'y 'added_features' '''
        # verificar si los datos son arrrglo y luego recorrer para optener los imputs de cada instancia del dataframe
        
        arregloImput=self.data.text_vec.iloc[index]
        imputarray=[]
        attention_maskarray=[]
        targetsarray=[]
        for val in arregloImput:
            input_ids = torch.tensor(val).cpu() # almacena las características de los datos de "text_vec" ​​que se han convertido en un vector de longitud fija.
            imputarray.append(input_ids)
            input_masks = torch.ones([input_ids.size(0)]).cpu()  # attention_mask almacena los elementos de entrada que se debe prestar atención y cuáles se deben ignorar
            attention_maskarray.append(input_masks)
            labels=ast.literal_eval(self.data.same.iloc[index]) 
            for label in labels:
                targets = torch.tensor([1 - label, label])  #ojo probar
                targetsarray.append(targets) 
                modelo = StackedCLSModel()
                prueba= modelo.predict(input_ids,input_masks,targets)
                verifica=prueba
           
          
        return {
        'input_ids': imputarray,               # devuelve las características de entrada para el punto de datos
            'attention_mask': attention_maskarray,     # devuelve la máscara de atención para el punto de datos
            'labels': targetsarray                    # devuelve un valor escalar que representa la puntuación de complejidad
        }
        
    
