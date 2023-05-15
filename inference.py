import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch.utils.data as data
import datasets
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertForSequenceClassification, DistilBertTokenizerFast
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification, ElectraForSequenceClassification
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import torch.optim as optim
def test(model, tokenizer, iterator, device):

    model.eval()     # Enter Evaluation Mode
    correct = 0
    total = 0
    val_loss = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch_idx, (sentences, labels) in enumerate(iterator):
            
            # tokenize the sentences
            encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            attention_mask = attention_mask.to(device)
            
            # move to GPU if necessary
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            # generate prediction
            outputs = model(input_ids, attention_mask=attention_mask)  # NOT USING INTERNAL CrossEntropyLoss
            prob = outputs.logits.sigmoid()   # BCEWithLogitsLoss has sigmoid
            
            # record processed data count
            total += (labels.size(0)*labels.size(1))
            # take the index of the highest prob as prediction output
            THRESHOLD = 0.2
            prediction = prob.detach().clone()
            prediction[prediction > THRESHOLD] = 1
            prediction[prediction <= THRESHOLD] = 0
            correct += prediction.eq(labels).sum().item()
            # num_rows = len(prediction.cpu())
            # row_counts = [sum(prediction[i]) for i in range(num_rows)]
            # print(row_counts) 
            # append labels and predictions to calculate F1 score
            y_true_list.append(labels.cpu().data)
            y_pred_list.append(prediction.cpu())
            # f1 = f1_score(labels.cpu().data, prediction.cpu(), zero_division=1, average='macro')
            #matrix = multilabel_confusion_matrix(labels.cpu().data, prediction.cpu())
            # print(f1)
    # calculate accuracy and F1 score
    acc = 100.*correct/total
    y_pred = torch.cat(y_pred_list, dim=0).numpy() # shape : n_samples * n_classes
    # Here it's up to you to make sure the target is in the right format : [[1, 0, ..., ], [....]]
    y_true = torch.cat(y_true_list, dim=0).numpy()
    recall = recall_score(y_pred, y_true, zero_division=1, average='macro')
    precision = precision_score(y_pred, y_true, zero_division=1,average='macro')
    f1 = f1_score(y_pred, y_true, zero_division=1, average='macro')
    #f1 = f1_score(y_true, y_pred, average='macro')
    print('correct: %i  / total: %i / valid_acc: %f / f1: %f ' % (correct, total, acc,f1))
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #load test data
    test_loader = torch.load('/home/p76101372/patent_classification/data/test_loader.pth')

    #load training model
    model = DistilBertForSequenceClassification.from_pretrained('/home/p76101372/patent_classification/model/finetune_distilbert_model', num_labels=47)
    tokenizer = DistilBertTokenizerFast.from_pretrained('/home/p76101372/patent_classification/tokenizer/distilbert_tokenizer', max_length = 512)
    model.to(device)
    #testing
    test(model, tokenizer, test_loader, device)
if __name__ == '__main__':
    main()