# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:50:26 2023

@author: fihey_p
"""

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoModelForTokenClassification
import numpy as np

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from transformers import BertTokenizer

from datasets import Dataset
from class_for_train import preprocessing_for_model, BertSentimentAnalysisTrain
import matplotlib.pyplot as plt

############################## cuda - certif #########################################################################################

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

####### Prepross headlines #####

headlines_good_ner = pd.read_csv(
    'C:/Users/fihey_p/Desktop/headlines_good_ner_ESG_2.csv', encoding='latin1', delimiter=',')
headlines_good_ner = headlines_good_ner.dropna()
headlines_good_ner = headlines_good_ner.reset_index(drop=True)


def prepross_entity(df):  # Enleve majuscules et espaces
    for index in range(len(df)):
        df.loc[index, 'ner0'] = str(df.loc[index, 'ner0'].lower())
        if df.loc[index, 'ner0'][0] == " ":
            df.loc[index, 'ner0'] = df.loc[index, 'ner0'][1:]
        #print(df.loc[index, 'ner0'])
    return(df)


############################## Adding tokens #########################################################################################
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
config = BertConfig.from_pretrained('ProsusAI/finbert')

pretrained_model = AutoModelForTokenClassification.from_pretrained(
    'ProsusAI/finbert', config=config).to('cuda')
model = BertSentimentAnalysisTrain(config).to('cuda')

########################################################################################################################
####################################### Dataset huggingface-mode #####################################################


new_set = headlines_good_ner[['Headline', 'ner0', 'Score']]
new_set = new_set.rename(columns={'Score': 'label', 'Headline': 'Headline'})


df_train, df_test, = train_test_split(
    new_set, stratify=new_set['label'], test_size=0.2, random_state=42)
df_train, df_val = train_test_split(
    df_train, stratify=df_train['label'], test_size=0.2, random_state=42)


dataset_train = Dataset.from_pandas(df_train)
dataset_val = Dataset.from_pandas(df_val)
dataset_test = Dataset.from_pandas(df_test)
dataset_total = Dataset.from_pandas(new_set)

dataset_total2 = dataset_total.map(
    lambda e: {
        'input_ids': tokenizer(e['Headline'], truncation=True, padding='max_length', max_length=64)['input_ids'],
        'ner0_input_ids': tokenizer(e['ner0'], truncation=True, padding='max_length', max_length=64)['input_ids'], }, batched=True)


dataset_train2 = dataset_train.map(
    lambda e: {
        'input_ids': tokenizer(e['Headline'], truncation=True, padding='max_length', max_length=64)['input_ids'],
        'ner0_input_ids': tokenizer(e['ner0'], truncation=True, padding='max_length', max_length=64)['input_ids'], }, batched=True)

dataset_val2 = dataset_val.map(lambda e: {
    'input_ids': tokenizer(e['Headline'], truncation=True, padding='max_length', max_length=64)['input_ids'],
    'ner0_input_ids': tokenizer(e['ner0'], truncation=True, padding='max_length', max_length=64)['input_ids'], }, batched=True)

dataset_test2 = dataset_test.map(lambda e: {
    'input_ids': tokenizer(e['Headline'], truncation=True, padding='max_length', max_length=64)['input_ids'],
    'ner0_input_ids': tokenizer(e['ner0'], truncation=True, padding='max_length', max_length=64)['input_ids'], }, batched=True)

dataset_train2.set_format(type='torch', columns=[
                          'input_ids', 'label', 'ner0_input_ids'])
dataset_val2.set_format(type='torch', columns=[
                        'input_ids', 'label', 'ner0_input_ids'])
dataset_test2.set_format(type='torch', columns=[
                         'input_ids', 'label', 'ner0_input_ids'])
dataset_total2.set_format(type='torch', columns=[
                          'input_ids', 'label', 'ner0_input_ids'])


########################################################################################################################
##################### On refait un dataset train pour rentre dans mon modèle (avec juste entités, phrase, length_entité, label) ###############
phrases = []
entities = []
length_entities = []
length_phrases = []
labels = []
dates = []
start_token = 101
end_token = 102
for index in range(len(dataset_train2['ner0_input_ids'])):
    phrases.append(dataset_train2['input_ids'][index])
    start_index_phrases = (
        dataset_train2['input_ids'][index] == 101).nonzero()[0]
    end_index_phrases = (
        dataset_train2['input_ids'][index] == 102).nonzero()[0]
    length_phrases.append(torch.tensor(
        len(dataset_train2['input_ids'][index][start_index_phrases+1:end_index_phrases])))
    start_index_ent = (
        dataset_train2['ner0_input_ids'][index] == 101).nonzero()[0]
    end_index_ent = (
        dataset_train2['ner0_input_ids'][index] == 102).nonzero()[0]
    entities.append(dataset_train2['ner0_input_ids'][index][1])
    length_entities.append(torch.tensor(
        len(dataset_train2['ner0_input_ids'][index][start_index_ent+1:end_index_ent])))
    labels.append(dataset_train2['label'][index])
    labels[index] += 1

data_train_dict = {
    'Headline': phrases,
    'entities': entities,
    'length_entities': length_entities,
    'label': labels,
    'length_phrases': length_phrases
}

dataset_train3 = Dataset.from_dict(data_train_dict)

phrases_val = []
entities_val = []
length_entities_val = []
length_phrases_val = []
labels_val = []
start_token = 101
end_token = 102
for index in range(len(dataset_val2['ner0_input_ids'])):
    phrases_val.append(dataset_val2['input_ids'][index])
    start_index_phrases = (
        dataset_val2['input_ids'][index] == 101).nonzero()[0]
    end_index_phrases = (dataset_val2['input_ids'][index] == 102).nonzero()[0]
    length_phrases_val.append(torch.tensor(
        len(dataset_val2['input_ids'][index][start_index_phrases+1:end_index_phrases])))
    start_index_ent = (
        dataset_val2['ner0_input_ids'][index] == 101).nonzero()[0]
    end_index_ent = (dataset_val2['ner0_input_ids'][index] == 102).nonzero()[0]
    entities_val.append(dataset_val2['ner0_input_ids'][index][1])
    length_entities_val.append(torch.tensor(
        len(dataset_train2['ner0_input_ids'][index][start_index_ent+1:end_index_ent])))
    labels_val.append(dataset_val2['label'][index])
    labels_val[index] += 1

data_val_dict = {
    'Headline': phrases_val,
    'entities': entities_val,
    'length_entities': length_entities_val,
    'label': labels_val,
    'length_phrases': length_phrases_val
}


dataset_val3 = Dataset.from_dict(data_val_dict)



phrases_test = []
entities_test = []
length_entities_test = []
length_phrases_test = []
labels_test = []
start_token = 101
end_token = 102
for index in range(len(dataset_test2['ner0_input_ids'])):
    phrases_test.append(dataset_test2['input_ids'][index])
    start_index_phrases = (
        dataset_test2['input_ids'][index] == 101).nonzero()[0]
    end_index_phrases = (dataset_test2['input_ids'][index] == 102).nonzero()[0]
    length_phrases_test.append(torch.tensor(
        len(dataset_test2['input_ids'][index][start_index_phrases+1:end_index_phrases])))
    start_index_ent = (
        dataset_test2['ner0_input_ids'][index] == 101).nonzero()[0]
    end_index_ent = (dataset_test2['ner0_input_ids'][index] == 102).nonzero()[0]
    entities_test.append(dataset_test2['ner0_input_ids'][index][1])
    length_entities_test.append(torch.tensor(
        len(dataset_train2['ner0_input_ids'][index][start_index_ent+1:end_index_ent])))
    labels_test.append(dataset_test2['label'][index])
    labels_test[index] += 1

data_test_dict = {
    'Headline': phrases_test,
    'entities': entities_test,
    'length_entities': length_entities_test,
    'label': labels_test,
    'length_phrases': length_phrases_test
}


dataset_test3 = Dataset.from_dict(data_test_dict)
########################################################################################################################################################
############# On freeze les poids de Bert pour entrainer uniquement les couches que je viens de rajouter ###########
print("Paramètres entraînables :")
for name, param in model.named_parameters():
    if any(torch.equal(param, pretrained_param) for pretrained_param in pretrained_model.parameters()):
        param.requires_grad = False
    if param.requires_grad:
        print(name)

################################################ On va entrainer #############################################

############## Hyperparamètres ############
learning_rate = 0.0005
batch_size = 32
epochs = 30
early_stopping_threshold = 5


linear_params = list(model.linear_layer.parameters()) + list(model.linear_layer_bis.parameters())+ list(model.linear_layer_bis.parameters())
# Créer un optimiseur pour les paramètres des couches linéaires uniquement
optimizer = torch.optim.Adam(linear_params, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


def custom_collate_fn(samples):
    phrases = [torch.tensor(sample['Headline']) for sample in samples]
    entities = [torch.tensor(sample['entities']) for sample in samples]
    length_entities = [torch.tensor(sample['length_entities'])
                       for sample in samples]
    length_phrases = [torch.tensor(sample['length_phrases'])
                      for sample in samples]
    labels = [sample['label'] for sample in samples]

    return {'Headline': torch.stack(phrases), 'entities': torch.stack(entities), 'length_entities': torch.stack(length_entities), 'length_phrases': torch.stack(length_phrases), 'label': labels}


train_loader = DataLoader(dataset_train3, batch_size=batch_size,
                          shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(dataset_val3, batch_size= batch_size,
                        shuffle=False, collate_fn=custom_collate_fn)

test_loader = DataLoader(dataset_test3, batch_size= batch_size,
                        shuffle=False, collate_fn=custom_collate_fn)
#####################################################################################################################
################################ Training #####################################################################################################################################################################
#####################################################################################################################
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Boucle d'entraînement
best_loss = np.inf
early_stopping_counter = 0  # Compteur pour le critère d'arrêt précoce
loss_values = []
loss_test = []
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total_samples = 0

    for batch in train_loader:
        phrases = batch['Headline']
        entities = batch['entities']
        labels = torch.tensor(batch['label']).to(device)

        #  print(labels)
        length_entities = batch['length_entities']
        length_phrases = batch['length_phrases']

        # Réinitialiser les gradients
        optimizer.zero_grad()

        input_phrases_preprocess, input_entities_preprocess, input_length_preprocess, input_length_phrases_preprocess, labels = preprocessing_for_model(
            phrases, entities, length_entities, length_phrases, labels)
        labels_hot = F.one_hot(labels, 3).float()
        # Effectuer une propagation avant (forward)

        outputs = model(input_phrases_preprocess, input_entities_preprocess,
                        input_length_preprocess, input_length_phrases_preprocess).float()

        # Calculer la perte
        loss = loss_fn(outputs, labels_hot)
       # loss.requires_grad = True

        # Rétropropagation et mise à jour des poids
        loss.backward()
        optimizer.step()

        # Calculer la précision

        outputs = torch.argmax(outputs, dim=1)
        total_samples += labels.size(0)
        correct += (outputs == labels).sum().item()
        # print(outputs, labels)
        # Suivi de la perte totale
        train_loss += loss.item()

    accuracy_train = correct / total_samples
    average_loss_train = train_loss / len(train_loader)
    loss_values.append(average_loss_train)

   # print("Training finished!")

    model.eval()
    test_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            phrases = batch['Headline']
            entities = batch['entities']
            labels = torch.tensor(batch['label']).to(device)
            if len(labels) < 32:
                continue
          #  print(labels)
            length_entities = batch['length_entities']
            length_phrases = batch['length_phrases']

            input_phrases_preprocess, input_entities_preprocess, input_length_preprocess, input_length_phrases_preprocess, labels = preprocessing_for_model(
                phrases, entities, length_entities, length_phrases, labels)
            labels_hot = F.one_hot(labels, 3).float()

            outputs = model(input_phrases_preprocess, input_entities_preprocess,
                            input_length_preprocess, input_length_phrases_preprocess).float()

            loss = loss_fn(outputs, labels_hot)

            test_loss += loss
            outputs = torch.argmax(outputs, dim=1)
            correct += (outputs == labels).sum().item()
            total_samples += labels.size(0)
            
        accuracy_test = correct / total_samples
        average_loss_test = test_loss / len(val_loader)
        loss_test.append(average_loss_test)

        if average_loss_test < best_loss:
            best_loss = average_loss_test
            early_stopping_counter = 0
            # Sauvegarder le meilleur modèle
            torch.save(model.state_dict(), 'best_model_esg_sentiment.pth')
        else:
            early_stopping_counter += 1

            #   print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {average_loss_test:.4f}, Val Accuracy: {accuracy_test:.2%}")

            # Vérifier le critère d'arrêt précoce
        if early_stopping_counter >= early_stopping_threshold:
            print("Early stopping! Model performance has started to deteriorate.")
            break
    

    print(f"Epoch [{epoch+1}/{epochs}], Loss_train: {average_loss_train:.4f}, Train_Accuracy: {accuracy_train:.2%}, Val Loss: {average_loss_test:.4f}, Val Accuracy: {accuracy_test:.2%} ")

