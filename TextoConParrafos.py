from sklearn.metrics import accuracy_score
import torch
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer
import os
from glob import glob
class TextoConParrafos:
    def __init__(self, id=None, texto=None, authors=None, changes=None):
        self._id = id
        self._texto = texto
        self._authors = authors
        self._changes = changes
        if texto is not None:
            self._parrafos = texto.split('\n')
        else:
            self._parrafos = []
        
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def texto(self):
        return self._texto  
    
    @texto.setter
    def texto(self, value):
         self._texto = value
      
    
    @property
    def authors(self):
        return self._authors
    
    @authors.setter
    def authors(self, value):
        self._authors = value
    
    @property
    def changes(self):
        return self._changes
    
    @changes.setter
    def changes(self, value):
        self._changes = value
    
    @property
    def parrafos(self):
        return self._parrafos
    
    @parrafos.setter
    def parrafos(self, value):
        self._parrafos = value
    
    def GetListParrafos(self):
        self.parrafos = self.texto.split('r/')
    
