

import json
import tqdm
import pandas as pd
from bullama import web_scrapping

from enum import Enum
 
class BularioType(Enum):
    Questions = 1
    PlainText = 2

class Bulario():
    def __init__(self, type_bulario : BularioType):

        self.type_bulario = type_bulario

        url_base = 'https://www.bulario.com'
        
        all_urls = []

        
        for letter in tqdm.tqdm(list(map(chr, range(97, 123))), desc="Getting the URLs for each letter"):
            url_letter = f"https://www.bulario.com/alfa/{letter}/"

            try:

              all_urls.append(url_letter)
              page_letter = web_scrapping.get_page(url_letter, True)
              all_urls.extend(get_pagination(url_base, page_letter))

            except:
              print(f"Erro em {url_letter}")
        all_drug_urls = []

        for url in tqdm.tqdm(all_urls, desc="Getting the URLs for each medication"):
            page_letter = web_scrapping.get_page(url, True)
            all_drug_urls.extend(get_drug_list(url_base, page_letter))
        self.all_drug_urls = all_drug_urls
                      
    def create_data(self):
        self.df_bulas = pd.DataFrame()

        if self.type_bulario is BularioType.Questions:
            for url in tqdm.tqdm(self.all_drug_urls, desc="Getting all questions from each medication"):
                self.df_bulas = get_df_row_bula(url, self.df_bulas)

          #  self.list_dict = cria_lista_perguntas(self.df_bulas)            

        else:
            for url in tqdm.tqdm(self.all_drug_urls, desc="Getting all raw text from each medication"):
                page = web_scrapping.get_page(url, True)
                if page:
                    bula_page = get_bula_page(page)
                    title = page.title.get_text().split('-')[0].strip()
                    supa_dict = {"Nome":title}
                    dict = { "texto" : bula_page }
                    supa_dict = merge_dicts(supa_dict, dict)

                    self.df_bulas = pd.concat([self.df_bulas, pd.DataFrame.from_records([supa_dict])])

                #    row = pd.DataFrame(pd.Series(bula_page).to_frame().rename(columns={0:"texto"}))
                #    self.df_bulas = pd.concat([self.df_bulas, row], ignore_index=True)
                else:
                   print(f"Erro: {url}")
            self.list_dict = self.df_bulas.to_dict()

    def save_txt(self, name_file):
        with open(name_file, 'w', encoding='utf8') as json_file:
            json.dump(self.list_dict, json_file, ensure_ascii=False)

    def save_csv(self, name_file):
       self.df_bulas.to_csv(name_file, sep=';', index=False)      

def get_bula_page(soup):
  bula = soup.find(id="bulaBody")
  content_except_laboratorio = []
 
  # Iterar sobre os filhos da div "bula"
  for child in bula.children:
      # Verificar se o filho é uma tag h2 com o texto "Laboratório"
      if 'Laboratório' in child.get_text():
          break  # Parar a iteração ao encontrar a tag h2 "Laboratório"
      if 'Informações Legais' in child.get_text():
          break  # Parar a iteração ao encontrar a tag h2 "Laboratório"   
      # Adicionar o conteúdo à lista
      if child.name != 'h2' and child.name != 'h3' and child.name != 'h4': 
        content_except_laboratorio.append(child)
  bula_text = ""
  for ctx in content_except_laboratorio:
     bula_text += " " + ctx.get_text().strip().replace("_","")

  return bula_text

def get_questions_bula(url_bula):
  bula_aux = web_scrapping.get_page(url_bula, True)
  bula = bula_aux.find(id="bulaBody")
  title = bula_aux.title.get_text().split('-')[0].strip()
  elements = bula.find_all("h3")
  list_dict = []
  for el in elements:

    resposta = ''
    next_sibling = el.find_next_sibling()
    while next_sibling and (next_sibling.name != 'h2' and next_sibling.name != 'h4'):
            resposta += " " + next_sibling.get_text().strip().replace("\n","").replace("\t", " ")
            next_sibling = next_sibling.find_next_sibling()
    
    dict = {
        "pergunta":el.get_text().replace("?", f" {title}?"),
        "resposta":resposta
    }

    if(dict["pergunta"] != "Laboratório"):
      list_dict.append(dict)
  return list_dict

def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_df_row_bula(url_bula, df):
  bula_aux = web_scrapping.get_page(url_bula, True)
  if bula_aux:
    bula = bula_aux.find(id="bulaBody")
    title = bula_aux.title.get_text().split('-')[0].strip()
    elements = bula.find_all("h3")

    supa_dict = {"Nome":title}
    for el in elements:
      dict = {
          el.get_text():el.find_next_siblings()[0].get_text().strip().replace("\n","")
      }

      supa_dict = merge_dicts(supa_dict, dict)

    df = pd.concat([df, pd.DataFrame.from_records([supa_dict])])

  return df

def criar_novas_perguntas(list_perguntas, nome_remedio, resposta):
  list_aux = []

  for pergunta in list_perguntas:
    dict = {
        "pergunta":pergunta.format(nome_remedio),
        "resposta": resposta
    }
    list_aux.append(dict)
  return list_aux

def get_pagination(url_base, soup):
  list_pages = []
  if soup:
    pagination = soup.find(attrs={"class": "pagination"})
    if pagination:
      pagination = pagination("a")
      for el in pagination:
        url_pagina = url_base + el['href']
        list_pages.append(url_pagina)
  return list(dict.fromkeys(list_pages))

def get_drug_list(url_base, soup):
  drug_list = []
  if soup:
    lista = soup.find(attrs={"class": "lists_list"})("a")
    for el in lista:
      url_remedio = url_base + el['href']
      drug_list.append(url_remedio)
  return list(dict.fromkeys(drug_list))
