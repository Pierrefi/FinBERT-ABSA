# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 09:15:08 2023

@author: fihey_p
"""

############### La différence avec class_and_preprocess est qu'il n'y a pas la date dans pre-processing ##########################################
###############     mais il y a les labels et que les opérations de la classe sont en cuda #########################################################

############################ Les classes sont différentes mais ce sont les mêmes couches donc on peut appliquer les poids du train sur le modèle ############################

import pandas as pd  
from transformers import  AutoModelForTokenClassification
import numpy as np

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from transformers import BertTokenizer

from datasets import Dataset

def preprocessing_for_model(phrases, entities, length, phrases_length, labels):
    
    input_phrases = torch.stack([tensor for tensor in phrases])
    inp_entities = torch.stack([tensor for tensor in entities])
    input_entities = torch.tensor([[item.item()] for item in inp_entities])
    inp_length = torch.stack([tensor for tensor in length])
    input_length = torch.tensor([[item.item()] for item in inp_length])
    #On raj phras_length
    inp_length_phrases = torch.stack([tensor for tensor in phrases_length])
    input_length_phrases = torch.tensor([[item.item()] for item in inp_length_phrases])
    
    input_phrases = input_phrases.to('cuda')
    input_entities = input_entities.to('cuda')
    input_length = input_length.to('cuda')
    input_length_phrases = input_length_phrases.to('cuda')
    
    indexes = torch.eq(input_phrases,input_entities)
    ######### Remove si il y a pas l'entité #######
    
    all_false_mask = torch.all(indexes == False, dim=1)
    # Trouver les indices des lignes où tous les éléments sont False
    indices_to_remove = torch.nonzero(all_false_mask).flatten()
    indices_to_keep = torch.ones_like(input_entities, dtype=torch.bool).squeeze().to('cuda')
    indices_to_keep[indices_to_remove] = False
    
    ## Supprimer les phrases, entités, labels et longueurs correspondantes
    input_phrases_new = torch.index_select(input_phrases, 0, indices_to_keep.nonzero().squeeze()).to('cuda')
    input_entities_new = torch.index_select(input_entities, 0, indices_to_keep.nonzero().squeeze())
    input_length_new = torch.index_select(input_length, 0, indices_to_keep.nonzero().squeeze())
    labels_new = torch.index_select(labels, 0, indices_to_keep.nonzero().squeeze())
    input_length_phrases_new = torch.index_select(input_length_phrases, 0, indices_to_keep.nonzero().squeeze())
   
    #print(input_entities_new)
    #### Trouver index entité dans phrase ############
    
    indexes_phrases_with_entities = torch.eq(input_phrases_new,input_entities_new).nonzero(as_tuple=True)[0].to('cuda')
    indexes_entities = torch.eq(input_phrases_new,input_entities_new).nonzero(as_tuple=True)[1].to('cuda')
    
    ######### Remove une des entités si il y en a deux #######
    
    unique_values, counts = torch.unique(indexes_phrases_with_entities, return_counts=True)
    duplicate_indices = torch.nonzero(counts == 2).squeeze()
  
    triple_indices = torch.nonzero(counts == 3).squeeze()

    
    indices_to_keep_bis = torch.ones_like(indexes_phrases_with_entities, dtype=torch.bool).to('cuda')
    indices_to_keep_bis[duplicate_indices] = False
    indices_to_keep_bis[triple_indices] = False
    indices_to_keep_bis[triple_indices+1] = False
    
    indexes_entities_in_phrases = torch.index_select(indexes_entities, 0, indices_to_keep_bis.nonzero().squeeze())
    
    
    return(input_phrases_new, indexes_entities_in_phrases, input_length_new, input_length_phrases_new, labels_new)





class BertSentimentAnalysisTrain(nn.Module):
    def __init__(self, bert_config):
        super(BertSentimentAnalysisTrain, self).__init__()
        self.bert = BertModel.from_pretrained('ProsusAI/finbert', config=bert_config)#.to('cuda')
        self.linear_layer = nn.Linear(768, 512)
        self.linear_layer_bis = nn.Linear(512, 3)
       # self.linear_layer_ter = nn.Linear(256, 3)

        
    def forward(self, input_phrases_new, indexes_entities_in_phrases, input_length_new, input_length_phrases_new, labels = None):
        # input_ids: input sequence tokens
        # attention_mask: sequence padding mask
        # entity_token_mask: binary mask indicating the entity tokens

        
        ############### Création de la matrice de masque pour retrouver les embeddings d'entités ##############  
        input_length_new2 = input_length_new.view(-1).to('cuda')
             
        num_elements = indexes_entities_in_phrases.size(0)
        max_index = 64
        
        matrix = torch.zeros(num_elements, max_index).to('cuda')
        
        
        row_indices = torch.arange(num_elements).unsqueeze(1).to('cuda')
        column_indices = torch.arange(max_index).unsqueeze(0).to('cuda')
        
        mask = (column_indices >= indexes_entities_in_phrases.unsqueeze(1)) & (column_indices < indexes_entities_in_phrases.unsqueeze(1) + input_length_new2.unsqueeze(1)).to('cuda')
        
        matrix[row_indices, column_indices] = mask.float()

        #### On fait une matrice de mask d'attention ### => Non concluant
        
        mask_matrix = matrix.unsqueeze(-1)

        outputs = self.bert(input_ids=input_phrases_new, output_hidden_states=True)
        

        last_hidden_state = outputs.last_hidden_state

        
        entity_hidden_state = last_hidden_state*mask_matrix

        entity_embedding = entity_hidden_state.mean(dim=1)

        entity_embedding = torch.nn.functional.normalize(entity_embedding)

        
        output = self.linear_layer(entity_embedding)
        output = torch.dropout(output, p=0.4, train=True)
        output = F.relu(output)
        
    #    output = self.linear_layer_bis(output)
     #   output = torch.dropout(output, p=0.4, train=True)
      #  output = F.relu(output)
        
        output = self.linear_layer_bis(output)
        output = torch.softmax(output, dim = 1)
               
        return output
