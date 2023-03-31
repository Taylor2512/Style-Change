from sklearn.metrics import accuracy_score
import torch
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer
import os
from glob import glob
class TextoConParrafos:
    def __init__(self, texto):
        self.texto_original = texto
        self.parrafos = texto.split('\n')
    @property
    def texto(self):
        return self.texto_original    
    def TokenizarParrafo(self,sequence):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        model = BertModel.from_pretrained('bert-base-uncased', config=config)
        tokenized_sequence = tokenizer.tokenize(sequence)
        indexed_tokens = tokenizer.encode(tokenized_sequence, return_tensors='pt')
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
