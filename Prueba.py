from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import DebertaModel, BertModel, BertConfig, BertTokenizer
import os
from glob import glob
import argparse
import json
import warnings
from enum import Enum
import pandas as pd
from typing import List


from TextoConParrafos import TextoConParrafos
from StackedCLSModel import StackedCLSModel


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


def SaveDataSet(folder, carpeta):
    folderComplete = os.path.join(folder, carpeta, carpeta+'-train')
    if os.path.exists(folderComplete):
        SaveValidationOrTrain(folderComplete)

    folderComplete = os.path.join(folder, carpeta, carpeta+'-validation')
    if os.path.exists(folderComplete):
        SaveValidationOrTrain(folderComplete)


def SaveValidationOrTrain(folder):
    problem_ids = get_problem_ids(folder)
    Lista: List[TextoConParrafos] = []
    for problem_id in problem_ids:
        textos = TextoConParrafos()
        textos = GetProblemsFileTxtAndJson(folder, problem_id)
        Lista.append(textos)
    texts = pd.DataFrame([{'id': o.id, 'textos': o.texto,'same':o.changes,'authors':o.authors,
                           'totalParrafo':o.totalParrafos, 'parrafos':o.parrafos,'nuevoParrafo':o.nuevoparrafos} for o in Lista])
    # thruhs = pd.DataFrame([{'id': o.id, 'autores': o.authors,
    #                       'cambios': o.changes, 'textos': o.texto} for o in Lista])
    # Exportar el DataFrame a un archivo CSV
    texts.to_csv(os.path.join(folder, 'textosproblem.csv'),
                 encoding='utf-8-sig', index=False)
    # thruhs.to_csv(os.path.join(folder, 'jsotrutn.csv'),
    #               encoding='utf-8-sig', index=False)
    
    # Exportar el DataFrame a un archivo XLSX file
    texts.to_excel(os.path.join(folder, 'textosproblem.xlsx'),
                 encoding='utf-8-sig', index=False)
    # thruhs.to_excel(os.path.join(folder, 'jsotrutn.xlsx'),
    #               encoding='utf-8-sig', index=False)
    


def main():
    parser = argparse.ArgumentParser(
        description="PAN23 Style Change Detection Task: Output Verifier"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="folder containing output/solution files (json)",
        required=True,
    )
    parser.add_argument(
        "--input",
        type=str,
        help="folder containing input files for task (txt)",
        required=True,
    )

    args = parser.parse_args()
    for i in range(1, 4):
        carpeta = 'pan23-multi-author-analysis-dataset' + str(i)
        SaveDataSet(args.input, carpeta)

    # for i in range(1, 4):
    #     carpeta = 'pan23-multi-author-analysis-dataset' + str(i)
    #     datframe = pd.read_csv(os.path.join(carpeta,carpeta+'-train','textosproblem.csv'))
    #     datframe = pd.read_csv(os.path.join(carpeta,carpeta+'-validation','textosproblem.csv'))


if __name__ == "__main__":
    main()
# print("La cantidad de párrafos es:", len(texto_con_parrafos.parrafos))

# for parrafo in texto_con_parrafos.parrafos:
#     print(parrafo)
