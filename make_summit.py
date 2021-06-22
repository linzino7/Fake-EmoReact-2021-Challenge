import json
import pandas as pd
import numpy as np
import re
from datetime import datetime

# Libraries

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries

from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
'''
torchtext.data.Field -> torchtext.legacy.data.Field
This means, all features are still available, but within torchtext.legacy instead of torchtext.
torchtext.data.Field has been moved to torchtext.legacy.data.Field
'''

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


'''read data'''
path = './pubilc_data_0522'
train_str = ''



with open(path+'/eval.json','r',encoding = 'utf-8')as f:
    train_str = f.read()
    #print(train_str)

f.close()

train = json.loads(train_str)


''' filter '''

#hashmap = {} # save item which is unique.
data = []

for dic in train:
  text = dic['text']+' '+dic['reply']
  text = text.replace('\n','')
  data.append( {'idx':dic['idx'],
          'context_idx':dic['context_idx'],
          'text':text})

#data = list(hashmap.values())
df = pd.DataFrame(data)

df.to_csv(path + '/eval.csv', index=False)

#raise 'outpt'
'''model'''

print('starting process data', datetime.now())

source_folder = './pubilc_data_0522'
destination_folder = './Model'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model parameter
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)

# idx	text	label
#fields = [('idx', text_field), ('context_idx', text_field), ('text', text_field), ('label', label_field)]
#field2 = [('idx', text_field), ('context_idx', text_field), ('text', text_field)]

tst_datafields = [('idx', text_field), ('context_idx', text_field), ('text', text_field)]

test = TabularDataset(path=source_folder+"/eval.csv", # the file path
           format='csv',
           skip_header=True, 
           fields=tst_datafields)

test_iter = Iterator(test, batch_size=1, device=device, train=False, shuffle=False, sort=False)

#print(next(iter(test_iter)))

class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea

# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Evaluation Function

def evaluate_s(model, test_loader):
    y_pred = []
    y_true = []
    c = 0
    model.eval()
    with torch.no_grad():
        for (idx, context_idx, titletext,), _ in test_loader:
                if c == 6975 or c==6976 or c==6977:
                    print('====',c)
                    print(idx, context_idx, titletext)


                titletext = titletext.type(torch.LongTensor)
                titletext = titletext.to(device)

                labels = torch.tensor([1]*len(titletext)).type(torch.LongTensor)
                labels = labels.to(device)
                if len(titletext)< 1:
                  print('===error===')
                  print(titletext)
                  print(labels)

                else:
                  #print(titletext)
                  output = model(titletext,labels)

                  _, output = output
                  y_pred.extend(torch.argmax(output, 1).tolist())

                  c = c+1
                  if True:
                    print(c)
                    #print(torch.argmax(output, 1).tolist())


    print(y_pred)
    #df['label'] = y_pred
    #df.to_csv(destination_folder + '/summit.csv', index=False)

    return y_pred

best_model = BERT().to(device)

print('starting eval', datetime.now())
load_checkpoint(destination_folder + '/model_mrr24.pt', best_model)

y_p = evaluate_s(best_model, test_iter)
'''output'''

print('starting output', datetime.now())
df['label'] = y_p
df2 = df.drop(['text'], axis=1)
df3 = df2.copy()
df3['label'] = df3['label'].replace(0,'real')
df3['label'] = df3['label'].replace(1,'fake')
df3['label']
df3.to_csv(destination_folder + '/eval_mrr24.csv', index=False)
