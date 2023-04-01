from sklearn.metrics import accuracy_score
import torch
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer
import os
import re
from glob import glob
class TextoConParrafos:
    def __init__(self, id=None, texto=None, authors=None, changes=None,sepsParrafo=None):
        self._id = id
        self._texto = texto
        self._authors = authors
        self._changes = changes
        self._sepsParrafo=sepsParrafo
        if texto is not None:
            self._parrafos = self.GetListParrafos()
            self._totalParrafos= len(self._parrafos)    
        else:
            self._parrafos = []
            self._totalParrafos=0
        
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
        
    @property
    def totalParrafos(self):
        return self._totalParrafos
    
    @totalParrafos.setter
    def totalParrafos(self, value):
        self._totalParrafos = value
        
    @property
    def sepsParrafo(self):
        return self._sepsParrafo
    
    @sepsParrafo.setter
    def sepsParrafo(self, value):
        self._sepsParrafo = value
    
    def GetListParrafos(self):
        self.parrafos = re.split(r'\n|\n\n|\r\n\r\n|\n \n', self.texto)
        self.totalParrafos= len(self.parrafos)
        self._UnirSeparaciones()
        
    def _UnirSeparaciones(self):
        self.sepsParrafo= 'SEP'.join(self.parrafos)
