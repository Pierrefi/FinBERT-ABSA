# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import pandas as pd  
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import numpy as np

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from transformers import BertTokenizer

torch.cuda.is_available()
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

### config ###

pd.set_option('display.max_columns', None)

proxy = "127.0.0.1:9000"
cacert_file = "C:/Users/fihey_p/Downloads/ZscalerRootCertificate-2048-SHA256.crt" 


os.environ["HTTP_PROXY"] = proxy
os.environ["HTTPS_PROXY"] = proxy
os.environ["REQUESTS_CA_BUNDLE"] = cacert_file

###### Modèles Bert ####

config = BertConfig.from_pretrained('ProsusAI/finbert')
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

tokenizer_ner = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model_ner = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
entity_recognition = pipeline('ner', model=model_ner, tokenizer=tokenizer_ner, aggregation_strategy="simple", device = 'cuda:0')

### Recup données ###

headlines = pd.read_csv('C:/Users/fihey_p/Desktop/ESG/ESG - trainingset.csv', encoding = 'latin1', delimiter = ';')

######### NER sur données #######

from tqdm import tqdm 
for index in tqdm(range(len(headlines))):
    sequence = headlines.loc[index, 'Headline']
    print(sequence)
    decoded_string = []
    for index_2 in range(len(entity_recognition(sequence))):
        if entity_recognition(sequence)[index_2]['entity_group'] == 'ORG' :
            company = [entity_recognition(sequence)[index_2]['word']]
            decoded_string.append(company[0])
    print(decoded_string)
    if decoded_string == []:
        decoded_string = ['no entity']
    for i in range(len(decoded_string)):
      headlines.loc[index, f'ner{i}'] = decoded_string[i]
     

######## Nettoyage tableau NER avec seulement les headlines où la bonne entité est trouvée ####
headlines_mod = headlines

for index in range(len(headlines_mod)): ##enleve espaces et majuscules
  headlines_mod.loc[index, 'Entité'] = str(headlines_mod.loc[index, 'Entité']).replace(" ", "")
  headlines_mod.loc[index, 'Entité'] = str(headlines_mod.loc[index, 'Entité']).lower()
  headlines_mod.loc[index, 'entity_ner0'] = str(headlines_mod.loc[index, 'ner0']).replace(" ", "")
  headlines_mod.loc[index, 'entity_ner0'] = str(headlines_mod.loc[index, 'entity_ner0']).lower()
  headlines_mod.loc[index, 'entity_ner1'] = str(headlines_mod.loc[index, 'ner1']).replace(" ", "")
  headlines_mod.loc[index, 'entity_ner1'] = str(headlines_mod.loc[index, 'entity_ner1']).lower()

for index in range(len(headlines_mod)):   ##garde que les bonnes
  if  headlines_mod.loc[index, 'entity_ner0'] in headlines_mod.loc[index, 'Entité'] :
    headlines_mod.loc[index, 'good'] = 1
  elif headlines_mod.loc[index, 'entity_ner1'] in headlines_mod.loc[index, 'Entité']: 
    headlines_mod.loc[index, 'entity_ner0'] = headlines_mod.loc[index, 'entity_ner1']
    headlines_mod.loc[index, 'ner0'] = headlines_mod.loc[index, 'ner1']
    headlines_mod.loc[index, 'good'] = 1
  else : 
    headlines_mod.loc[index, 'good'] = 0
  
headlines_mod = headlines_mod.drop(['ner1','ner2', 'ner3', 'ner4','entity_ner1'], axis = 1)
print('sum good_ner : ', headlines_mod['good'].sum())

headlines_good_ner = headlines_mod.loc[headlines_mod['good'] == 1.0]
headlines_good_ner = headlines_good_ner[['Headline', 'Sentiment', 'Entité', 'ner0', 'entity_ner0']]
headlines_good_ner = headlines_good_ner.reset_index(drop=True)
headlines_good_ner

headlines_good_ner.to_csv('C:/Users/fihey_p/Desktop/headlines_esg_train.csv')


