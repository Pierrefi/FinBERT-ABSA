# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:06:36 2023

@author: fihey_p
"""


import pandas as pd  
from transformers import  AutoModelForTokenClassification
import numpy as np
from transformers import AutoTokenizer
from class_and_preprocess import BertSentimentAnalysis, preprocessing_for_model_use

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re 

import os
from tqdm import tqdm
import torch
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import pipeline

from datasets import Dataset
from torch.utils.data import DataLoader

############################## cuda - certif ################################################################################################################

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

proxy = "127.0.0.1:9000"    
cacert_file = "C:/Users/fihey_p/Downloads/ZscalerRootCertificate-2048-SHA256.crt" 


os.environ["HTTP_PROXY"] = proxy
os.environ["HTTPS_PROXY"] = proxy
os.environ["REQUESTS_CA_BUNDLE"] = cacert_file

############################## Data #########################################################################################################################

headlines = pd.read_csv('C:/Users/fihey_p/Desktop/ESG/esg_june_text.csv', encoding = 'latin1', delimiter = ',')
#date = headlines['captureTime'][1]
#############################################################################################################################################################
########################################### Modèles Bert ####################################################################################################
#############################################################################################################################################################

############################################ Modèle NER #####################################################################################################

config = BertConfig.from_pretrained('ProsusAI/finbert')
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

tokenizer_ner = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
model_ner = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
entity_recognition = pipeline('ner', model=model_ner, tokenizer=tokenizer_ner, aggregation_strategy="simple", device = 'cuda:0')


############################################# Modèle Finbert ABSA ###########################################################################################

model = BertSentimentAnalysis(config)#.to('cuda')
model.load_state_dict(torch.load('best_model_esg_sentiment.pth'))


##############################################################################################################################################################
####################################### NER SUR DATA #########################################################################################################
##############################################################################################################################################################

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
    headlines.loc[index, 'NER'] = decoded_string[0]
    if len(decoded_string) > 1 : 
        headlines.loc[index, 'NER_2'] = decoded_string[1]


#################################### Extraction seconde entité ###############################################################################################

headlines_second_entity = headlines[headlines['NER_2'].notna()][['CaptureTime', 'Headline', 'NER_2', 'news_id']]
headlines_second_entity = headlines_second_entity.reset_index(drop = True)
headlines_second_entity.columns = ['CaptureTime', 'Headline',  'NER', 'news_id']

headlines_first_entity = headlines[['CaptureTime', 'Headline', 'NER', 'news_id']]

headlines_new = pd.concat([headlines_first_entity, headlines_second_entity], ignore_index=True)
headlines_new.columns

headlines = headlines_new[['CaptureTime', 'Headline', 'NER', 'news_id']]
####################### Premier préprocessing pour éviter problèmes de tokenization ###########################################################################


def prepross_entity(df): 
  for index in range(len(df)):
      df.loc[index, 'NER'] = str(df.loc[index,'NER'].lower())
      if df.loc[index, 'NER'][0] == " ":
         df.loc[index, 'NER'] = df.loc[index, 'NER'][1:]
      #print(df.loc[index, 'ner0'])
  return(df)

headlines_prep = prepross_entity(headlines)

headlines_with_entities = headlines.loc[headlines['NER'] != 'no entity']
headlines_with_entities = headlines_with_entities.reset_index(drop=True)

headlines_with_entities.to_csv('C:/Users/fihey_p/Desktop/headlines_for_mod_ESG_2.csv')

headlines_with_entities = pd.read_csv('C:/Users/fihey_p/Desktop/headlines_for_mod_ESG_2.csv', encoding = 'latin1')

date = []
for ele in headlines_with_entities['CaptureTime'] :
    date_obj = str(ele)[0:10]
    date_obj = date_obj.replace('-', '')
    date_obj = int(date_obj)
    date.append(date_obj)

headlines_with_entities['CaptureTime'] = date  

###############################################################################################################################################################
############################# Pré-processing des données post-NER pour utilisation modèle #####################################################################
###############################################################################################################################################################

dataset_for_model = Dataset.from_pandas(headlines_with_entities)
dataset_for_model = dataset_for_model.map(
    lambda e: {
        'input_ids': tokenizer(e['Headline'], truncation=True, padding='max_length', max_length=64)['input_ids'],
        'ner_input_ids': tokenizer(e['NER'], truncation=True, padding='max_length', max_length=64)['input_ids'],
        }, batched=True)
    
dataset_for_model.set_format(type='torch', columns=['input_ids', 'ner_input_ids'])

###############################################################################################################################################################
############################# Ajout des informations sur phrases et entités ###################################################################################
###############################################################################################################################################################

capture_time = []
phrases = []
entities = []
length_entities = []
length_phrases = []
news_id = []
start_token = 101
end_token = 102
for index in tqdm(range(len(dataset_for_model['ner_input_ids']))):
    phrases.append(dataset_for_model['input_ids'][index])
    start_index_phrases = (dataset_for_model['input_ids'][index] == 101).nonzero()[0]
    end_index_phrases = (dataset_for_model['input_ids'][index] == 102).nonzero()[0]
    length_phrases.append(torch.tensor(len(dataset_for_model['input_ids'][index][start_index_phrases+1:end_index_phrases])))
    start_index_ent = (dataset_for_model['ner_input_ids'][index] == 101).nonzero()[0]
    end_index_ent = (dataset_for_model['ner_input_ids'][index] == 102).nonzero()[0]
    entities.append(dataset_for_model['ner_input_ids'][index][1])
    length_entities.append(torch.tensor(len(dataset_for_model['ner_input_ids'][index][start_index_ent+1:end_index_ent])))
    news_id.append(torch.tensor(dataset_for_model['news_id'][index]))
    capture_time.append(torch.tensor(dataset_for_model['CaptureTime'][index]))
    
data_with_caracteristics = {
    'capture_time' : capture_time,
    'news_id' : news_id,
    'Texte': phrases,
    'entities': entities,
    'length_entities': length_entities,
    'length_phrases': length_phrases,

}

data_with_caracteristics = Dataset.from_dict(data_with_caracteristics)

def custom_collate_fn(samples):
    phrases = [torch.tensor(sample['Texte']) for sample in samples]
    entities = [torch.tensor(sample['entities']) for sample in samples]
    length_entities = [torch.tensor(sample['length_entities']) for sample in samples]
    length_phrases = [torch.tensor(sample['length_phrases']) for sample in samples]
    news_id = [torch.tensor(sample['news_id']) for sample in samples]
    capture_time = [torch.tensor(sample['capture_time']) for sample in samples]

    return {'Texte': torch.stack(phrases), 'entities': torch.stack(entities), 'length_entities': torch.stack(length_entities), 'length_phrases' : torch.stack(length_phrases), 'news_id' : torch.stack(news_id), 'date' : torch.stack(capture_time)}


data_loader = DataLoader(data_with_caracteristics, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

######################################################################################################################################################################
############################# Utilisation de la fonction préprocessing pour modèle ###################################################################################
#############################                et utlisation modèle                #####################################################################################
######################################################################################################################################################################


text_inp = []
ner_inp = []
lab_out = []
len_ent = []
news_ids = []
date = []
for batch in data_loader:
    if len(batch['Texte']) < data_loader.batch_size:  
        continue  
    else:
        input_phrases_preprocess, input_entities_preprocess, input_length_preprocess, input_length_phrases_preprocess, input_news_id, input_date = preprocessing_for_model_use(batch['Texte'], batch['entities'], batch['length_entities'], batch['length_phrases'], batch['news_id'], batch['date'])
        input_phrases_preprocess = input_phrases_preprocess.to('cpu')
        input_entities_preprocess = input_entities_preprocess.to('cpu')
        input_length_preprocess = input_length_preprocess.to('cpu')
        input_length_phrases_preprocess = input_length_phrases_preprocess.to('cpu')
        outputs = model(input_phrases_preprocess, input_entities_preprocess, input_length_preprocess, input_length_phrases_preprocess).float()
        text_inp.append(input_phrases_preprocess.to('cpu'))
        ner_inp.append(input_entities_preprocess.to('cpu'))
        len_ent.append(input_length_preprocess.to('cpu'))
        lab_out.append(outputs.to('cpu'))
        news_ids.append(input_news_id.to('cpu'))
        date.append(input_date.to('cpu'))

text_inp = torch.cat(text_inp, dim=0)
ner_inp = torch.cat(ner_inp, dim=0)
lab_out = torch.cat(lab_out, dim=0)
len_ent = torch.cat(len_ent, dim=0)
news_ids = torch.cat(news_ids, dim = 0)
date_inp = torch.cat(date, dim = 0)

######################################################################################################################################################################
######################################################################################################################################################################
###############################################        Post-processing       ###############################################
######################################################################################################################################################################
######################################################################################################################################################################


######################################################################################################################################################################
############################# On remet tout sous forme de liste pour créer un dataframe ###################################################################################
######################################################################################################################################################################

text = []
entities = []
for index in tqdm(range(len(ner_inp))):
    start_index = ner_inp[index]
    end_index =  ner_inp[index]+len_ent[index]
    
    entity = text_inp[index][start_index:end_index]
    entities.append(tokenizer.decode(entity))
    
    text_wpad = tokenizer.decode(text_inp[index])
    text_wpad = ' '.join(token for token in text_wpad.split() if token not in ['[PAD]', '[CLS]', '[SEP]'])    
    text.append(text_wpad)
    
    
label = lab_out.detach().numpy()
label = label.tolist()

news_ident = news_ids.numpy()
news_ident = news_ident.tolist()

dates = date_inp.numpy()
dates = date_inp.tolist()

neg, neut, pos = zip(*label)
negative = list(neg)
neutral = list(neut)
positive = list(pos)

#dates = [date]*len(negative)

new_dataframe = pd.DataFrame({
    'date' : dates,
    'entity': entities,
    'negative': negative,
    'neutral': neutral,
    'positive': positive,
    'news_id' : news_ident
})


nb_entities = len(new_dataframe['entity'].unique())

############################################################################################################################################
############################################ Similarités de mots ###########################################################################
############################################################################################################################################

############################################ Matrice de similarités d'embeddings ###########################################################
model_similarity = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


entities_for_sim = new_dataframe['entity'].tolist()
entities_embeddings = model_similarity.encode(entities_for_sim)
embeddings_array = np.array(entities_embeddings)

similarity_matrix = cosine_similarity(embeddings_array)
most_similar_indices = np.argsort(similarity_matrix, axis=1)

nb_entities = len(new_dataframe['entity'].unique())
############################################ On parcourt pour remplacer les mots très similaires ############################################
for i, indices in enumerate(most_similar_indices):
    print(entities_for_sim[i])
    for j in range(1, 100):  
        similar_index = indices[-j]
        similarity_score = similarity_matrix[i, similar_index]
        if similarity_score > 0.8 : 
          similar_word = entities_for_sim[similar_index]
          entities_for_sim[similar_index] = entities_for_sim[i]
          print(f"    - Mot : {similar_word}, Similarité : {similarity_score}, Nouveau mot : {entities_for_sim[similar_index]}")
    print()

new_dataframe['entity'] =  entities_for_sim
nb_entities_bis = len(new_dataframe['entity'].unique())
nb_change = nb_entities_bis - nb_entities

############################################ Post-processing entities ############################################

new_dataframe['entity'] = new_dataframe['entity'].apply(lambda entity: re.sub('[^A-Za-z0-9 ]+', '', entity))

new_dataframe.columns = ['date', 'entity', 'negative', 'neutral', 'positive',  'news_id']

############################################ Déconcaténation dataframes (par dates) ############################################

new_dataframe['date'] = pd.to_datetime(new_dataframe['date'], format = '%Y%m%d')
new_dataframe['date'] = new_dataframe['date'].apply(lambda date : str(date)[0:10])

date_groups = new_dataframe.groupby('date')
date_dataframes = []
for date, group in date_groups:
    date_dataframes.append(group)
    print(date)
   # date = date.replace("-", "")
    group.to_csv(f'C:/Users/fihey_p/Desktop/Sentiment_analysis ESG/Sentiment_esg_{date}.csv', encoding = 'latin1',  errors='ignore',  index=False, sep=';', decimal='.')



new_dataframe.to_csv('C:/Users/fihey_p/Desktop/Sentiment_analysis/Sentiment_june_esg.csv', encoding = 'latin1',  errors='ignore',  index=False, sep=';', decimal='.')

