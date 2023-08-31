# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:37:05 2023

@author: fihey_p
"""

from transformers import pipeline, AutoTokenizer
import pandas as pd 
from tqdm import tqdm

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")



headlines = pd.read_csv('C:/Users/fihey_p/Desktop/csv/20230601_bloombergStoryBodies.csv')
scores_esg = pd.read_csv('C:/Users/fihey_p/Desktop/Sentiment_analysis ESG/Sentiment_esg_2023-06-01.csv', delimiter = ';')


############################################ Extraction de scores_esg avec Headlines jour/jour ###########################################

for i in range(1,31):
    i_str = str(i).zfill(2)
    print(i_str)
    try : 
        headlines = pd.read_csv(f'C:/Users/fihey_p/Desktop/csv/202306{i_str}_bloombergStoryBodies.csv')
        scores_esg = pd.read_csv(f'C:/Users/fihey_p/Desktop/Sentiment_analysis ESG/Sentiment_esg_2023-06-{i_str}.csv', delimiter = ';')    
    except Exception :
        continue
    
    hed = []
    for index in range(len(scores_esg)):  
        new = scores_esg.loc[index]
        new_index = new["news_id"]
        new_headline = headlines.loc[new_index]
        new_headline = new_headline.Headline
        hed.append(new_headline)
    scores_esg['Headline'] = hed
    
    scores_esg.to_csv(f'C:/Users/fihey_p/Desktop/Sentiment_with_headlines/2023_06_{i_str}.csv', encoding = 'latin1',  errors='ignore',  index=False, sep=';', decimal='.')

############################################ Extraction de scores_esg avec headlines tout le mois de juin ########################################## 

score_june = []
for i in range(1,31):
    i_str = str(i).zfill(2)
  print(i_str)
    try : 
        headlines = pd.read_csv(f'C:/Users/fihey_p/Desktop/csv/202306{i_str}_bloombergStoryBodies.csv')
        scores_esg = pd.read_csv(f'C:/Users/fihey_p/Desktop/Sentiment_analysis ESG/Sentiment_esg_2023-06-{i_str}.csv', delimiter = ';')    
    except Exception :
        continue
    
    hed = []
    for index in range(len(scores_esg)):  
        new = scores_esg.loc[index]
        new_index = new["news_id"]
        new_headline = headlines.loc[new_index]
        new_headline = new_headline.Headline
        hed.append(new_headline)
    scores_esg['Headline'] = hed
    score_june.append(scores_esg)

score_june_headline = pd.concat(score_june, ignore_index=True)

score_june_headline.to_csv('C:/Users/fihey_p/Desktop/Sentiment_analysis/Sentiment_june_esg_with_headlines.csv', encoding = 'latin1',  errors='ignore',  index=False, sep=';', decimal='.')

############################################ Extraction de summary mois de juin ESG  ###########################################

for i in tqdm(range(1,31)):
    i_str = str(i).zfill(2)
    try :  #Au cas ou y a pas la date 
        headlines = pd.read_csv(f'C:/Users/fihey_p/Desktop/csv/202306{i_str}_bloombergStoryBodies.csv')
        scores_esg = pd.read_csv(f'C:/Users/fihey_p/Desktop/Sentiment_analysis ESG/Sentiment_esg_2023-06-{i_str}.csv', delimiter = ';') 
    except Exception :
        continue
    summarize = []
    news_identification = []
    for index in tqdm(range(len(scores_esg))):  
        new = scores_esg.loc[index]
        new_index = new.news_id
        new_headline_story = headlines.loc[new_index].Body
    #    new_headline_ID = new_headline.SUID
   #     new_headline_story = headlines[(headlines['SUID'] == new_headline_ID) & (headlines['Event'] == 'ADD_STORY')].Body.values
        try :
            tokens = tokenizer.tokenize(new_headline_story)#Au cas ou y a un body vide
        except Exception :
            summarize.append('no summary')
            news_identification.append(new_index)
            continue
        
        if len(tokens) > 900 : 
            c = tokenizer.convert_tokens_to_string(tokens[0:900])
        else : 
            c = tokenizer.convert_tokens_to_string(tokens)
        try : 
            summary_text = summarizer(c, max_length=130, min_length=30, do_sample=True)
        except Exception : 
            
            summarize.append('no summary')
            news_identification.append(new_index)
            continue
        summary_text = summary_text[0]['summary_text']
        summarize.append(summary_text)
        news_identification.append(new_index)

    summary = pd.DataFrame({'news_id' : news_identification, 'summary' : summarize })    
    summary.to_csv(f'C:/Users/fihey_p/Desktop/Summary_esg/2023-06-{i_str}.csv', encoding = 'latin1',  errors='ignore',  index=False, sep=';', decimal='.')

############################################ Extraction de traduction des résumés ###########################################

translator = pipeline("translation", model="t5-base")

objectif = 'translate English to French:'
for i in tqdm(range(1,31)):
    i_str = str(i).zfill(2)
    try : 
        summaries = pd.read_csv(f'C:/Users/fihey_p/Desktop/Summary_esg/2023-06-{i_str}.csv', delimiter = ';', encoding='latin1')
    except Exception : 
        continue
    translation = []
    for index in tqdm(range(len(summaries))) : 
        news = summaries.summary[index]
        preproc_news = objectif + " " + news
        translated_news = translator(preproc_news)[0]['translation_text']
        translation.append(translated_news)

    summaries['Summary_translation'] =  translation
    summaries.to_csv(f'C:/Users/fihey_p/Desktop/Summary + Translation ESG/Summary_translation_2023-06-{i_str}.csv', encoding = 'latin1',  errors='ignore',  index=False, sep=';', decimal='.')

############################################ Extraction de body jour/jour dans le dossier Summary_translation ###########################################


for i in range(1,31):
    i_str = str(i).zfill(2)
    print(i_str)
    try : 
        headlines = pd.read_csv(f'C:/Users/fihey_p/Desktop/csv/202306{i_str}_bloombergStoryBodies.csv')
        transl_summary = pd.read_csv(f'C:/Users/fihey_p/Desktop/Summary + Translation ESG/Summary_translation_2023-06-{i_str}.csv', delimiter = ';', encoding='latin1')    
    except Exception :
        continue
    
    headlines['news_id'] = headlines.reset_index().index
    transl_summary = transl_summary.merge(headlines[['SUID', 'news_id']], on = 'news_id', how = 'left')
    transl_summary = transl_summary.merge(headlines.query("Event == 'ADD_STORY'")[['Body', 'SUID']], on = 'SUID', how = 'left' )
    
    transl_summary.to_csv(f'C:/Users/fihey_p/Desktop/Summary + Translation + Body ESG/with_body_2023_06_{i_str}.csv', encoding = 'latin1',  errors='ignore',  index=False, sep=';', decimal='.')






