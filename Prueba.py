from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer,DebertaTokenizer,DebertaConfig
import os
from glob import glob
import argparse
import json
import warnings
from enum import Enum
import pandas as pd
import ast
from typing import List
from TextoConParrafos import TextoConParrafos
from TextDataset import TextDataset
import pandas as pd
import numpy as np
import datetime
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import re
import re,os
import unicodedata
import matplotlib.pyplot as plt
import itertools
import sklearn.metrics as metrics
from sklearn import metrics, feature_selection
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss,accuracy_score,classification_report,accuracy_score,confusion_matrix,recall_score,precision_score,roc_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, brier_score_loss,accuracy_score,classification_report,accuracy_score,confusion_matrix,recall_score,precision_score,roc_curve
from transformers import Trainer, TrainingArguments, EvalPrediction,DataCollatorWithPadding, Trainer, TrainingArguments, AutoModelForSequenceClassification
from transformers import BertModel, BertConfig, BertTokenizer, DebertaConfig, DebertaModel, DebertaTokenizer,DebertaV2Model, DebertaV2Config,DebertaV2Tokenizer,AutoTokenizer,AutoModel,AutoConfig
import optuna.visualization as optuna_visualization
import plotly
import optuna
import numpy as np
 
import random
import pandas as pd
import numpy as np
import datetime
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import re,os
import unicodedata
import matplotlib.pyplot as plt
import itertools
import sklearn.metrics as metrics
from sklearn import metrics, feature_selection
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, brier_score_loss,accuracy_score,classification_report,accuracy_score,confusion_matrix,recall_score,precision_score,roc_curve
from transformers import Trainer, TrainingArguments, EvalPrediction,DataCollatorWithPadding, Trainer, TrainingArguments, AutoModelForSequenceClassification
from transformers import BertModel, BertConfig, BertTokenizer, DebertaConfig, DebertaModel, DebertaTokenizer,DebertaV2Model, DebertaV2Config,DebertaV2Tokenizer,AutoTokenizer,AutoModel,AutoConfig
import optuna.visualization as optuna_visualization
import plotly
import optuna
import numpy as np

from transformers.modeling_outputs import SequenceClassifierOutput
from bs4 import BeautifulSoup
from TextoConParrafos import TextoConParrafos
# Definir las variables globales MODEL y MODEL_TYPE
MODEL = None
MODEL_TYPE = None

def main():
    model_types = ['mdeberta', 'deberta'] 
    parser = argparse.ArgumentParser(description="PAN23 Style Change Detection Task: Output Verifier")
    parser.add_argument("--output",type=str,help="folder containing output/solution files (json)",required=True,)
    parser.add_argument("--input",type=str,help="folder containing input files for task (txt)",required=True,)
    parser.add_argument("--modelType",type=str,default="mdeberta",help="type model to use",required=False,choices=model_types)
    args = parser.parse_args()
    global tokenizer, config, MODEL,MODEL_TYPE,date_string
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    if args.modelType=='mdeberta': 
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
        config = AutoConfig.from_pretrained("microsoft/mdeberta-v3-base",output_hidden_states=True, output_attentions=True)
        MODEL = AutoModel.from_pretrained("microsoft/mdeberta-v3-base", config=config)
        MODEL_TYPE=args.modelType
    elif args.modelType=='deberta':
        tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
        config = DebertaConfig.from_pretrained("microsoft/deberta-base", output_hidden_states=True, output_attentions=True)
        MODEL = DebertaModel.from_pretrained("microsoft/deberta-base", config=config) 
        MODEL_TYPE=args.modelType
    for i in range(1, 4):
        carpeta = 'pan23-multi-author-analysis-dataset' + str(i)
        SaveDataSet(args, carpeta)

    for i in range(1, 4):
        carpeta = 'pan23-multi-author-analysis-dataset' + str(i)
        GenerarModelo(args, carpeta)
    for i in range(1, 4):
        carpeta = 'pan23-multi-author-analysis-dataset' + str(i)
        GenerarSolucion(args, carpeta)

def GenerarSolucion(argss, carpeta):
    datatest=None
    train= os.path.join(argss.input, carpeta) 
    if argss.modelType=='mdeberta': 
        datatest = pd.read_json(os.path.join(train,carpeta+'-test','mdebertaTokenizer.json'))
    elif argss.modelType=='deberta':
        datatest = pd.read_json(os.path.join(train,carpeta+'-test','ebertaTokenizer.json'))
    
    direccion=GenerarDirectorio('best_model')
    rutamodel=os.path.join(direccion,'best_model.pth')
    modelo_cargado = torch.load(rutamodel)

def SaveDataSet(args, carpeta):
    folder= args.input
    folderComplete = os.path.join(folder, carpeta, carpeta+'-train')
    if os.path.exists(folderComplete):
        SaveValidationOrTrain(folderComplete,args)
    folderComplete = os.path.join(folder, carpeta, carpeta+'-validation')
    if os.path.exists(folderComplete):
        SaveValidationOrTrain(folderComplete,args)

def GenerarModelo(argss, carpeta):
    dataTrainer=None
    dataEvaluation=None
    train= os.path.join(argss.input, carpeta) 
    if argss.modelType=='mdeberta': 
        dataTrainer = pd.read_json(os.path.join(train,carpeta+'-train','mdebertaTokenizer.json'))
        dataEvaluation = pd.read_json(os.path.join(train,carpeta+'-validation','mdebertaTokenizer.json'))
    elif argss.modelType=='deberta':
        dataTrainer = pd.read_json(os.path.join(train,carpeta+'-train','ebertaTokenizer.json'))
        dataEvaluation = pd.read_json(os.path.join(train,carpeta+'-validation','ebertaTokenizer.json'))
    dataTrainer=   dataTrainer.iloc[:5,:]
    dataEvaluation=   dataEvaluation.iloc[:5,:]
    train_set, eval_dataset = MyDataset(dataTrainer), MyDataset(dataEvaluation)        
    model = StackedCLSModel(MODEL, argss.modelType)
    trainer = Trainer(model=model,
                      args=arguments,
                      train_dataset=train_set,
                      eval_dataset=eval_dataset,
                      compute_metrics=compute_metrics)
    def objective(trial):
        # Definir los hiperparámetros a sintonizar con Optuna
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        num_layers = trial.suggest_int('num_layers', 1, 4)
        # Entrenar el modelo
        trainer.train()
    
        # Devolver la métrica que deseas optimizar (por ejemplo, precisión)
        return trainer.evaluate();
    study = optuna.create_study(direction='maximize')  # Cambia a 'minimize' si deseas minimizar la métrica
    study.optimize(objective, n_trials=2)  # Ajusta el número de ensayos según tus necesidades
    if len(study.best_trials) != 0:
        best_trial = study.best_trials[0]  # Obtén el primer mejor resultado
        learning_rate = best_trial.params['learning_rate']
        arguments.learning_rate=learning_rate

        num_layers = best_trial.params['num_layers']
        arguments.num_train_epochs=num_layers
     
    trainer = Trainer(model=model,
                      args=arguments,
                      train_dataset=train_set,
                      eval_dataset=eval_dataset,
                      compute_metrics=compute_metrics)
    result=trainer.train()
    metrics=trainer.state.log_history
    metrics=normalizar_propiedades(metrics)
    generarGrafico(metrics)
    direccion=GenerarDirectorio('best_model')
    rutamodel=os.path.join(direccion,'best_model.pth')
    torch.save(trainer.model, rutamodel)
    
def GenerarDirectorio(name):
    rutabase = os.getcwd()  # Obtiene la ruta base del proyecto actual
    directorio = os.path.join(rutabase, name)
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    return directorio
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text
def vectorize_text(s0, s1, max_length):
    # Unicode normalization
    s0=remove_html_tags(s0)
    s1=remove_html_tags(s1)
    s0 = ''.join(c for c in unicodedata.normalize('NFD', s0) if unicodedata.category(c) != 'Mn')  # elimina cualquier diacrítico o acento de la cadena s
    s1 = ''.join(c for c in unicodedata.normalize('NFD', s1) if unicodedata.category(c) != 'Mn')  # elimina cualquier diacrítico o acento de la cadena s
    # Unicode normalization
    #s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')  # elimina cualquier diacrítico o acento de la cadena s
    s0 = re.sub(r"[^a-zA-Záéíóú.,!?;:<>()$€\[\]]+", r" ", s0)   # reemplaza todas las coincidencias del patrón con un espacio
    s1 = re.sub(r"[^a-zA-Záéíóú.,!?;:<>()$€\[\]]+", r" ", s1)   # reemplaza todas las coincidencias del patrón con un espacio

    #'''convierte la entrada de texto sin formato en un formato numérico que se puede introducir en un modelo de aprendizaje automático'''
    input_ids = tokenizer.encode(   # utiliza el tokenizador previamente entrenado 
      ''.join(s0), 
      ''.join(s1),                       # para codificar una cadena "s0" en sus identificadores de token correspondientes
      add_special_tokens=True,      # especifica si se agregan tokens especiales al principio y al final de la secuencia de tokens
      max_length=max_length,        # especifica la longitud máxima de la secuencia de tokens resultante
      #padding='longest',            # especifica cómo rellenar secuencias más cortas a la misma longitud que la secuencia más larga.
      padding='max_length',
      truncation=True,              # especifica si se truncan las secuencias que son más largas que "max_length"
      return_tensors='np'           # especifica que la salida debe devolverse como una matriz numpy (o pt?).
    )
    return input_ids[0]            # devuelve solo el primer elemento de la matriz como una matriz numpy
def get_problem_ids(input_folder: str) -> list:
    """
    gathers all problem-ids of input files as list
    :param input_folder: folder holding input files (txt)
    :return: sorted list of problem-ids
    """
    problem_ids = []
    for file in glob(os.path.join(input_folder, "*.txt")):
        problem_ids.append(os.path.basename(file)[8:-4])
    print(f"Read {len(problem_ids)} problem ids from {input_folder}.")
    return sorted(problem_ids)
def GetProblemsFileTxtAndJson(input_folder: str, problem_id: str) -> TextoConParrafos:
    file_pathJson = os.path.join(
        input_folder, "truth-problem-" + problem_id + ".json")
    file_pathtxt = os.path.join(input_folder, "problem-" + problem_id + ".txt")
    with open(file_pathtxt, 'r', encoding='utf-8') as filetxt:
        with open(file_pathJson, 'r') as filejson:
            data = json.load(filejson)
        texto_con_parrafos = TextoConParrafos()
        texto_con_parrafos.texto = filetxt.read()
        texto_con_parrafos.id = problem_id
        texto_con_parrafos.authors=data["authors"]
        texto_con_parrafos.changes=data["changes"]
        texto_con_parrafos.GetListParrafos()
        texto_con_parrafos._GenerarAgrupaciones()
        #  texto_con_parrafos._recorrer_parrafos()
        if isinstance(texto_con_parrafos, TextoConParrafos):
            return texto_con_parrafos
        else:
            return None
def SaveValidationOrTrain(folder,args):
    problem_ids = get_problem_ids(folder)
    Lista: List[TextoConParrafos] = []
    for problem_id in problem_ids:
        textos = TextoConParrafos()
        textos = GetProblemsFileTxtAndJson(folder, problem_id)
        Lista.append(textos)
    # Obtener el número de elementos a tomar
    num_instances = int(len(Lista) * 0.8)
    # Copiar la lista original para preservar su orden
    copia_lista = Lista.copy()
    # Tomar el 80% de los elementos en una nueva lista
    lista_80porciento = random.sample(copia_lista, num_instances)
    # Eliminar los elementos seleccionados de la copia de la lista original
    for elemento in lista_80porciento:
        copia_lista.remove(elemento)
    # Los elementos restantes corresponden al 20% restante
    lista_20porciento = copia_lista
    if "train" in os.path.basename(folder):    
        SaveDatasetComplete(folder, args, lista_80porciento)
        
        SaveDatasetComplete(folder.replace('train','test'), args, lista_20porciento)

def SaveDatasetComplete(folder, args, Lista):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if args.modelType=='mdeberta': 
       data = [{ 'id': o.id,'pair': pair, 'same': same,'text_vec':vectorize_text(pair[0],pair[1],512)} for o in Lista for pair, same ,Id in zip(o.nuevoparrafos, o.changes)]
       datatrain = pd.DataFrame(data)
       texts = pd.DataFrame([{'id': o.id, 'textos': o.texto,'same':o.changes,'authors':o.authors,'totalParrafo':o.totalParrafos, 'parrafos':o.parrafos,'nuevoParrafo':o.nuevoparrafos} for o in Lista])
       datatrain.to_json(os.path.join(folder, 'mdebertaTokenizer.json'), orient='records')
       texts.to_json(os.path.join(folder, 'textosproblem.json'), orient='records')
    elif args.modelType=='deberta':
       data = [{'id': o.id,'pair': pair, 'same': same,'text_vec':vectorize_text(pair[0],pair[1],512)} for o in Lista for pair, same in zip(o.nuevoparrafos, o.changes)]
       datatrain = pd.DataFrame(data)
       datatrain.to_json(os.path.join(folder, 'ebertaTokenizer.json'), orient='records')
       texts = pd.DataFrame([{'id': o.id, 'textos': o.texto,'same':o.changes,'authors':o.authors,'totalParrafo':o.totalParrafos, 'parrafos':o.parrafos,'nuevoParrafo':o.nuevoparrafos} for o in Lista])
       texts.to_json(os.path.join(folder, 'textosproblem.json'), orient='records')
# Definir bien los argumentos
arguments = TrainingArguments(
    output_dir='output',  # Ruta del directorio de salida donde se guardarán los resultados del entrenamiento
    evaluation_strategy='epoch',  # Evaluación del modelo al final de cada época
    num_train_epochs=1,  # Número total de épocas de entrenamiento
    per_device_train_batch_size=16,  # Tamaño del lote de entrenamiento por dispositivo. Ajustar según la memoria GPU disponible
    per_device_eval_batch_size=16,  # Tamaño del lote de evaluación por dispositivo. Ajustar según la memoria GPU disponible
    learning_rate=5e-5,  # Tasa de aprendizaje utilizada en el entrenamiento
    overwrite_output_dir=True,  # Sobrescribir el directorio de salida si ya existe
    remove_unused_columns=False,  # No eliminar columnas no utilizadas del conjunto de datos
    logging_dir='logs',  # Ruta del directorio donde se guardarán los archivos de registro del entrenamiento
    logging_steps=10,  # Número de pasos después de los cuales se realizará el registro
    save_strategy='epoch',  # Estrategia de guardado del modelo: al final de cada época
    save_total_limit=10,  # Límite total de modelos guardados
    load_best_model_at_end=True,  # Cargar el mejor modelo al final del entrenamiento
    warmup_steps=10,  # Número de pasos de calentamiento antes de ajustar la tasa de aprendizaje
    weight_decay=0.03,  # Factor de decaimiento de peso para la regularización L2
    adam_epsilon=1e-8,  # Epsilon para el optimizador Adam, utilizado para la estabilidad numérica
    adam_beta1=0.5,  # Coeficiente beta1 para el optimizador Adam
    adam_beta2=0.5,  # Coeficiente beta2 para el optimizador Adam
    lr_scheduler_type='cosine',  # Tipo de programador de tasa de aprendizaje: programador coseno curva de aprendizaje  entre el eje x
    gradient_accumulation_steps=1,  # Número de pasos de acumulación de gradiente antes de realizar una actualización de parámetros
    max_grad_norm=5.0,  # Valor máximo de la norma del gradiente para evitar explosiones de gradiente
    save_steps=10  # Número de pasos después de los cuales se guarda el modelo
    )
def group_by_property(metrics):
    properties = set()
    for metric in metrics:
        for property in metric:
            properties.add(property)
    grouped_metrics = {}
    for property in properties:
        grouped_metrics[property] = [metric for metric in metrics if property in metric]
    return grouped_metrics
def agrupar_propiedades(lista):
    resultado = {}
    for diccionario in lista:
        for propiedad, valor in diccionario.items():
            # Reemplazar espacios y caracteres especiales en la propiedad con guiones bajos
            propiedad = propiedad.replace(' ', '_').replace('-', '_')
            if propiedad not in resultado:
                resultado[propiedad] = []
            resultado[propiedad].append(valor)
    return resultado

def _getvalores(data,key):{
  [metric[key] for metric in data if key in data]  
}

def generarGrafico(metrics,numero=2):
    metricas2 = agrupar_propiedades(metrics)
    fecha_hora = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Configurar el gráfico
    plt.figure(figsize=(10, 8))
    plt.title('Métricas de entrenamiento por época')
    plt.xlabel('Época')
    plt.ylabel('Valor de la métrica')
    plt.ylim(0, 1.5)

    # Graficar cada métrica con longitud distinta a 7
    for name, valores in metricas2.items():
        if len(valores) == numero:  # Verificar si la longitud es igual a 7
            plt.plot(range(len(valores)), valores, label=name,marker='o')

    # Agregar leyenda y mostrar gráfico
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Establecer valores y etiquetas en el eje y con todos los decimales
    plt.yticks([i / 10 for i in range(15)])

    # Agregar fecha y hora al pie de la gráfica
    plt.figtext(0.99, 0.01, fecha_hora, ha='right', va='bottom')
    ruta =GenerarDirectorio('Graficos')
       # Ajustar el tamaño de la gráfica dentro de la ventana
     # Guardar el gráfico en la ubicación especificada con mayor resolución
    plt.tight_layout()

    plt.savefig(os.path.join(ruta, f'{date_string}_grafico.png'), dpi=1800,bbox_inches='tight')

def GenerarMatrizConfuncion(matriz_confusion):
    # Configurar el gráfico
    plt.imshow(matriz_confusion, cmap='Blues', interpolation='nearest')
    plt.title('Matriz de Confusión')
    plt.colorbar()

    # Etiquetas de los ejes x e y
    tick_marks = np.arange(len(matriz_confusion))
    plt.xticks(tick_marks, ['Negativo', 'Positivo'])
    plt.yticks(tick_marks, ['Negativo', 'Positivo'])

    # Mostrar los valores de la matriz de confusión en cada celda
    thresh = matriz_confusion.max() / 2.
    for i, j in np.ndindex(matriz_confusion.shape):
        plt.text(j, i, format(matriz_confusion[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if matriz_confusion[i, j] > thresh else "black")

    plt.figtext(0.99, 0.01, date_string, ha='right', va='bottom')
    ruta =GenerarDirectorio('Graficos')
    # Ajustar los márgenes del gráfico
    plt.tight_layout()

    plt.savefig(os.path.join(ruta, f'{date_string}_matrizConfusion.png'), dpi=1800,bbox_inches='tight')

class MyDataset(Dataset):             # define una nueva clase MyDataset que hereda de Dataset
          def __init__(self, dataframe):    # define el constructor  "__init__"  que toma un solo argumento dataframe
              #print(dataframe)
              self.len = len(dataframe)   # calcula la longitud de la entrada dataframe usando la funcion "len" y la almacena como una variable de instancia "self.len"
              self.data = dataframe       # se asigna la entrada dataframe a una variable de instancia "self.data"


          def __getitem__(self, index):   # define el método "__getitem__" que toma un solo argumento index
              ''' el metodo __getitem__ devuelve un diccionario que contiene cuatro claves: 'input_ids', 'attention_mask', 'labels'y 'added_features' '''

              input_ids = torch.tensor(self.data.text_vec.iloc[index]).cpu() # almacena las características de los datos de "text_vec" ​​que se han convertido en un vector de longitud fija.
              #attention_mask = torch.ones([input_ids.size(0)]).cpu()  # attention_mask almacena los elementos de entrada que se debe prestar atención y cuáles se deben ignorar
              #
              mask = torch.ones(input_ids.shape,dtype=int)#Crear un tensor con el mismo tamaño que input_ids lleno de unos:
              pad_positions = (input_ids == 0)#Identificar las posiciones en input_ids que contienen el token especial [PAD]
              mask[pad_positions] = 0 #Actualizar las posiciones correspondientes en mask a cero:
              attention_mask = mask #Actualizar las posiciones correspondientes en mask a cero:
              #
              label = self.data.same.iloc[index] # almacena un valor escalar que representa la etiqueta de salida para la puntuación de complejidad
              targets = torch.tensor([1 - label, label])  #ojo probar ESTO ES NUEVO
              return {
                  'input_ids': input_ids,               # devuelve las características de entrada para el punto de datos
                  'attention_mask': attention_mask,     # devuelve la máscara de atención para el punto de datos
                  'labels': targets                    # devuelve un valor escalar que representa la puntuación de complejidad
              }

          def __len__(self):
              return self.len   # devuelve la longitud del conjunto de datos personalizado
class StackedCLSModel(nn.Module):
          def __init__(self, model=MODEL, model_type=MODEL_TYPE):
              super(StackedCLSModel, self).__init__()
              self.model = MODEL
              self.model_type = MODEL_TYPE
              self.Fusion = nn.Parameter(torch.zeros(12, 1))
              self.dropout = nn.Dropout(0.2)
              self.funActivacion = nn.ReLU()
              self.lin1 = nn.Linear(768, 128)
              self.lin2 = nn.Linear(128, 2)
              self.loss_func = nn.CrossEntropyLoss()
              self.init = 0
          def forward(self, input_ids, attention_mask, labels=None):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            if self.model_type == "mdeberta":
              cls_tensors = torch.stack([outputs[1][n][:, 0, :] for n in range(1, 13)])
            elif self.model_type == "deberta":
              cls_tensors = torch.stack([outputs[1][n][:, 0, :] for n in range(1, 13)])
            t_cls_tensors = cls_tensors.transpose(1, 0)
            t_cls_tensors_mean = torch.mean(t_cls_tensors, dim=1)  # Reducción de la dimensión 12 a 2
            x = self.lin1(t_cls_tensors_mean)
            x = self.dropout(x)
            x = self.funActivacion(x)
            logit = self.lin2(x)
            loss = None
            if labels is not None:
              loss = self.loss_func(logit, labels.float())
            return SequenceClassifierOutput(loss=loss, logits=logit)
        
          def predict(self, input_ids, attention_mask):
              logits = self.forward(input_ids, attention_mask, labels=None)
              predicciones = logits.logits.argmax(dim=1)
              predicciones =np.argmax(predicciones.tolist(),axis=-1)
              return predicciones.tolist()        
def normalizar_propiedades(lista):
    for diccionario in lista:
        for propiedad in list(diccionario.keys()):
            propiedad_normalizada = propiedad.replace(" ", "_").replace("-", "_").replace(".", "_").replace(":", "_").replace("/", "_")
            if propiedad != propiedad_normalizada:
                diccionario[propiedad_normalizada] = diccionario.pop(propiedad)
    return lista
def c_at_1(train_data, test_data, threshold=0.5):
      n = float(len(test_data))
      nc, nu = 0.0, 0.0

      for gt_score, pred_score in zip(train_data, test_data):
        if pred_score == 0.5:
          nu += 1
        elif (pred_score > 0.5) == (gt_score > 0.5):
          nc += 1.0
    
      return (1 / n) * (nc + (nu * nc / n))
def binarize(y, threshold=0.5, triple_valued=False):
    y = np.array(y)
    y = np.ma.fix_invalid(y, fill_value=threshold)
    if triple_valued:
        y[y > threshold] = 1
    else:
        y[y >= threshold] = 1
    y[y < threshold] = 0
    return y
def f_05_u_score(train_data, test_data, pos_label=1, threshold=0.5):
      test_data = binarize(test_data)

      n_tp = 0
      n_fn = 0
      n_fp = 0
      n_u = 0

      for i, pred in enumerate(test_data):
        if pred == threshold:
          n_u += 1
        elif pred == pos_label and pred == train_data[i]:
          n_tp += 1
        elif pred == pos_label and pred != train_data[i]:
          n_fp += 1
        elif train_data[i] == pos_label and pred != train_data[i]:
          n_fn += 1

      return (1.25 * n_tp) / (1.25 * n_tp + 0.25 * (n_fn + n_u) + n_fp)
def brier_score(train_data, test_data):
      try:
        return 1 - brier_score_loss(train_data, test_data)
      except ValueError:
        return 0.0 
def auc_score(train_data, test_data):
    try:
        return roc_auc_score(train_data, test_data)
    except ValueError:
        return 0.0
def compute_metrics(p: EvalPrediction): # calcula diversas métricas de evaluación
  preds = np.argmax(p.predictions,axis=-1)
  preds = np.squeeze(preds)
  labels = np.argmax(p.label_ids,axis=-1)
  labels = np.squeeze(labels)
  precision = precision_score(labels, preds, average='micro')
  precision_macro = precision_score(labels, preds, average='macro')
  auc = auc_score(labels, preds)
  #mse = metrics.mean_squared_error(labels, preds)# Se toma el primer indice del label
  #print("SIZES::::",labels.shape, preds.shape)
  return {
          'auc': auc,
          'c@1': c_at_1(labels, preds),
          'f_05_u': f_05_u_score(labels, preds),
          'F1': f1_score(labels, preds, average = 'micro'),
          'brier': brier_score(labels, preds),
          'precision micro': precision,
          'precision macro': precision_macro
          } 
if __name__ == "__main__":
    main() 
    
