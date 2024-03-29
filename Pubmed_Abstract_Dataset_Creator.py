import pandas as pd
import requests
from text_fix import *

def format_query(search_query):
    if ' ' not in search_query:
        query = search_query
    else: 
        query = '+'.join(search_query.split())
    return query
def Dataset_creater(list,category,nb_pages):
    url_base='https://pubmed.ncbi.nlm.nih.gov/?term='
    data = pd.DataFrame()
    query_nb=1
    for term in list:
        print('query nb',query_nb)
        page_nb = 1
        pg = nb_pages
        while pg>0:
            try:
                # if query_nb==29 and page_nb==4:
                #     break
                a = len(data)
                query = format_query(term)
                search_terms = query
                page = '&page='
                filter_year='&filter=years.2015-2024'
                format_pubmed='&format=pubmed'
                size ='&size=200'
                url = url_base+search_terms+'&filter=simsearch1.fha'+page+str(page_nb)+filter_year+format_pubmed+size
                r = requests.get(url)
                with open('C:\Dataset/pubmed_data.txt', 'w',encoding="utf8") as file:
                    file.write(r.text)
                Abstract_extract('pubmed_data.txt','pubmed_test.txt')
                Categorization('pubmed_test.txt','pubmed_out.txt',category)
                data_temp = pd.read_csv('pubmed_out.txt')
                data = pd.concat([data,data_temp],axis=0)
                pg-=1
                page_nb += 1
                
                data = data.reset_index(drop=True)
                b = len(data)
                print(len(data))
                if b-a<190:
                    break
            except:0
        query_nb += 1
    return data