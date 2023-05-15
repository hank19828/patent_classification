
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DistilBertForSequenceClassification, DistilBertModel,DistilBertTokenizer, DistilBertTokenizerFast
from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification, ElectraForSequenceClassification
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import torch.optim as optim
import preprocess
hidden_size = 768
n_class = 47
maxlen = 8

encode_layer=6
filter_sizes = [2, 2, 2]
num_filters = 3
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filter_total = num_filters * len(filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, n_class, bias=False)
        self.bias = nn.Parameter(torch.ones([n_class]))
        self.filter_list = nn.ModuleList([
        nn.Conv2d(1, num_filters, kernel_size=(size, hidden_size)) for size in filter_sizes
        ])

    def forward(self, x):
        # x: [bs, seq, hidden]
        x = x.unsqueeze(1) # [bs, channel=1, seq, hidden]
        
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x)) # [bs, channel=1, seq-kernel_size+1, 1]
            mp = nn.MaxPool2d(
                kernel_size = (encode_layer-filter_sizes[i]+1, 1)
            )
            # mp: [bs, channel=3, w, h]
            pooled = mp(h).permute(0, 3, 2, 1) # [bs, h=1, w=1, channel=3]
            pooled_outputs.append(pooled)
        
        h_pool = torch.cat(pooled_outputs, len(filter_sizes)) # [bs, h=1, w=1, channel=3 * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])
        
        output = self.Weight(h_pool_flat) + self.bias # [bs, n_class]
        return output
class SentenceDataset(data.Dataset):

    def __init__(self, database, LABEL_COLUMNS):
        self.database = database
        self.LABEL_COLUMNS = LABEL_COLUMNS
    def __len__(self):
        return self.database.shape[0]
        #return 512

    def __getitem__(self, idx):
        
        # return the sentence
        i = self.database["abstract&claim"][idx]
        
        # return the label array
        label = self.database.loc[idx, self.LABEL_COLUMNS]
        label = np.array(label, dtype=float)
        
        return i, label
def train(model, tokenizer, iterator, optimizer, criterion, device):
    
    model.train()     # Enter Train Mode
    train_loss = 0
    
    for batch_idx, (sentences, labels) in enumerate(iterator):
        # print(sentences)
        
        # tokenize the sentences
        encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        attention_mask = attention_mask.to(device)
        # move to GPU if necessary
        input_ids, labels = input_ids.to(device), labels.to(device)
        # print(input_ids.shape)
        # generate prediction
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)  # NOT USING INTERNAL CrossEntropyLoss
        # print(outputs.shape)
        # compute gradients and update weights
        loss = criterion(outputs, labels) # BCEWithLogitsLoss has sigmoid
        loss.backward()
        optimizer.step()
        # accumulate train loss
        train_loss += loss.item()
    # print completed result
    print('train_loss: %f' % (train_loss))
    return train_loss
def validation(model, tokenizer, iterator, optimizer, criterion, device):

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
            prob = outputs.sigmoid()   # BCEWithLogitsLoss has sigmoid
            
            # record processed data count
            total += (labels.size(0)*labels.size(1))
            loss = criterion(outputs, labels) # BCEWithLogitsLoss has sigmoid
            # accumulate train loss
            val_loss += loss.item()
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
    return acc, f1, val_loss
class CustomBERTModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomBERTModel, self).__init__()
        self.bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True, return_dict=True)
        ###New layers:
        self.linear = nn.Linear(hidden_size, n_class)
        self.textcnn = TextCNN() 
    def forward(self, input_ids, attention_mask):
        sequence_output = self.bert_model(input_ids, attention_mask=attention_mask)
        hidden_states = sequence_output.hidden_states
        # print(len(hidden_states))
        # print(hidden_states[11].shape)
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)
        for i in range(2, 7):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=12, hidden]
        logits = self.textcnn(cls_embeddings)
        return logits
        # # print(sequence_output.last_hidden_state.shape)
        # # sequence_output has the following shape:(batch_size, sequence_length, 768)
        # linear1_output = self.linear1(output[:,0,:].view(-1,768))
        # # print(linear1_output.shape)
        # linear2_output = self.linear2(linear1_output)
        # return linear2_output
def plot_fig(file, title, x_label, y_label, save_path):
    plt.plot(file)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    plt.close()

def main():
    df_merged = pd.read_csv('/home/p76101372/patent_classification/data/preprocess_df.csv')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load training dataset
    # print(len(df_merged.columns))
    LABEL_COLUMNS = df_merged.columns.tolist()[4:-1]
    print(len(LABEL_COLUMNS))
    dataset = SentenceDataset(df_merged, LABEL_COLUMNS)
    print("Total: %i" % len(dataset))

    # Split dataset into training, validation, and test sets
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    # Calculate lengths of train, valid, and test sets
    train_len = len(train_data)
    valid_len = len(valid_data)
    test_len = len(test_data)

    print("Training: %i / Validation: %i / Testing: %i" % (train_len, valid_len, test_len))

    #ML parameters
    lr = 1e-05
    epoch = 30
    batch_size = 16

    # Load into Iterator (each time get one batch)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    valid_loader = data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    
    #save data
    torch.save(train_loader,'/home/p76101372/patent_classification/data/train_loader.pth')
    torch.save(valid_loader,'/home/p76101372/patent_classification/data/valid_loader.pth')
    torch.save(test_loader,'/home/p76101372/patent_classification/data/test_loader.pth')
    
    #define model & tokenizer
    # tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    # model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=len(LABEL_COLUMNS))
    # model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(LABEL_COLUMNS))
    # tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = 512)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    #model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=len(LABEL_COLUMNS))#(num_labels=len(LABEL_COLUMNS))
    model = CustomBERTModel(num_labels=len(LABEL_COLUMNS))
    # tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/electra-base-emotion")
    # model = ElectraForSequenceClassification.from_pretrained("bhadresh-savani/electra-base-emotion", num_labels=len(LABEL_COLUMNS), problem_type="multi_label_classification",ignore_mismatched_sizes=True)
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(LABEL_COLUMNS))
    model.to(device)
    
    #define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    #define loss function
    criterion = nn.BCEWithLogitsLoss()

    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    val_f1_list = []
    for e in range(epoch):
        print("===== Epoch %i =====" % e)
        # training
        print("Training started ...")
        train_loss = train(model, tokenizer, train_loader, optimizer, criterion, device)
        train_loss_list.append(train_loss)
        # validation testing
        print("Validation started ...")
        acc, f1, val_loss = validation(model, tokenizer, valid_loader, optimizer, criterion, device)
        val_loss_list.append(val_loss)
        val_acc_list.append(acc)
        val_f1_list.append(f1)
    #save model & tokenizer
    torch.save(model.state_dict(),'/home/p76101372/patent_classification/model/distilbert_param.pt')
    #model.save_pretrained('/home/p76101372/patent_classification/model/finetune_distilbert_model_new')
    tokenizer.save_pretrained('/home/p76101372/patent_classification/tokenizer/distilbert_tokenizer_new')

    #plot fig and save
    plot_fig(train_loss_list, 'train_loss', 'epoch', 'loss', '/home/p76101372/patent_classification/img/distilbert_train_loss_new.png')
    plot_fig(val_loss_list, 'val_loss', 'epoch', 'loss', '/home/p76101372/patent_classification/img/distilbert_val_loss_new.png')
    plot_fig(val_acc_list, 'val_acc', 'epoch', 'acc', '/home/p76101372/patent_classification/img/distilbert_val_acc_new.png')
    plot_fig(val_f1_list, 'val_f1', 'epoch', 'f1', '/home/p76101372/patent_classification/img/distilbert_val_f1_new.png')

if __name__ == '__main__':
    main()
    