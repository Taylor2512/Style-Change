from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import os
import numpy as np
import torch
from transformers import BertModel, BertConfig, BertTokenizer
 
class TextoConParrafos:
    def __init__(self, texto):
        self.texto_original = texto
        self.parrafos = texto.split('\n')
        recorrer_parrafos(self)

def TokenizarParrafo(sequence):
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
    outputs[1].shape
    outputs[1].sum()
    outputs[0][:, 0].shape
    outputs[0][:, 0].sum()
    outputs[2][5][:, 0]
    cls_tensors = []
    for i in range(1, 13):
        cls_tensors.append(outputs[2][i][0, 0])
    cls_stack = torch.stack(cls_tensors)
    cls_stack.shape
    t_cls_stack = cls_stack.transpose(1, 0)
    F = torch.zeros(12, 1)
    t_cls_stack.shape
    F.shape
    torch.mm(t_cls_stack, F).squeeze()

def recorrer_parrafos(self):
    print("La cantidad de párrafos es:", len(self.parrafos))
    for p in self.parrafos:
        TokenizarParrafo(p)

 



# Directorio donde se encuentran los archivos de entrenamiento
corpus_dir = "C:\\DataSets\\release\\Prueba\dataset1\\problem-4200.txt"
with open(corpus_dir, 'r') as file:
    texto_con_parrafos = TextoConParrafos(file.read())

# print("La cantidad de párrafos es:", len(texto_con_parrafos.parrafos))

# for parrafo in texto_con_parrafos.parrafos:
#     print(parrafo)
