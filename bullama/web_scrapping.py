import socket
import urllib.error
import urllib.request
from bs4 import BeautifulSoup
import json

def get_page(url, utf):
    try:
        req = urllib.request.Request(url, headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
        with urllib.request.urlopen(req, timeout = 5) as f:
            content = f.read()
            if utf:
                content = content.decode('utf-8')
            return BeautifulSoup(content, features="html5lib")
    except socket.timeout as e1:
        print(f'\tTime out - {url}')
    except urllib.error.URLError as e2:
        print('\tURL error')
    except urllib.error.HTTPError as e3:
        print('\tHTTP error')
    return None

def get_bula_questions_dict(link):
    page = get_page(link, False)
    if(page):
        bula = page.find(attrs={"itemprop": "articleBody"})
        for sup in bula("sup"):
            sup.decompose()
        for br in bula("br"):
            br.decompose()
        # Encontrar todas as tags "h3"
        h3_tags = bula.find_all('h3')
        dict = {}
        # Iterar sobre as tags "h3" para encontrar as tags "p" posteriores até a próxima "h3"
        for h3_tag in h3_tags:
            next_element = h3_tag.find_next()
            resposta = ""
            while next_element is not None and next_element.name != "h3":
                resposta += next_element.get_text().replace('\n', '')
                next_element = next_element.find_next_sibling()
            dict[h3_tag.text] = resposta
        return dict

