import chromadb
from chromadb.utils import embedding_functions

import pandas as pd
import json


class RAG():
    def __init__(self, file):
        self.collection = self.get_collection(file)

    def get_collection(self, file):
        lista = ler_arquivo_para_lista(file)

        my_data = pd.Series(lista).to_frame().rename(columns={0:"texto"})

        docs=my_data["texto"].tolist() 
        ids= [str(x) for x in my_data.index.tolist()]
        client = chromadb.Client()
        collection = client.get_or_create_collection("bula")

        collection.add(
            documents=docs,
            ids=ids
        )

        return collection
    
    def get_doc(self, query):
        results=self.collection.query(
            
            query_texts=query,
            n_results=2,
            include=["documents"]
        )
        return results['documents'][0][0]
    
def ler_arquivo_para_lista(nome_arquivo):
    minha_lista = []
    with open(nome_arquivo, 'r') as file:
        for line in file:
            minha_lista.append(line.strip())
    return minha_lista