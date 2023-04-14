from sklearn.metrics import accuracy_score
import torch
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer
import os
import re
from glob import glob
class TextoConParrafos:
    def __init__(self, id=None, texto=None, authors=None, changes=None,sepsParrafo=None,nuevoparrafos=None):
        self._id = id
        self._texto = texto
        self._authors = authors
        self._changes = changes
        self._sepsParrafo=sepsParrafo
        self._nuevoparrafos=nuevoparrafos
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
    def nuevoparrafos(self):
        return self._nuevoparrafos
    
    @nuevoparrafos.setter
    def nuevoparrafos(self, value):
        self._nuevoparrafos = value
        
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
        self.sepsParrafo= '[SEP]'.join(self.parrafos)
    
    
    def _GenerarAgrupaciones(self):
        grupos = []
        self._nuevoparrafos = []

        for i in range(len(self.parrafos)-1):
            grupo = [self.parrafos[i], self.parrafos[i+1]]
            grupos.append(grupo)
        
      
        for grupo in grupos:
            nuevo_grupo = []
            nuevo_grupo = '[SEP]'.join(grupo)
            self._nuevoparrafos.append(nuevo_grupo)
        
        #NO MOVER REFICAR MAS TARDE
        # for agregarcls in self._nuevoparrafos:
        #     agregarcls.append('CLS' + agregarcls)