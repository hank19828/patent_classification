
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from matplotlib import pyplot as plt
from transformers import (
    AutoTokenizer,
    BertModel,
    RobertaModel,
    ElectraModel,
    DistilBertModel,
    RobertaForSequenceClassification,
)
from sklearn.preprocessing import MultiLabelBinarizer
import os

torch.cuda.empty_cache()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Choose BERT Model')
    parser.add_argument('--model', type=str, default='bert',
                        help='Pretrained model to use ')
    args = parser.parse_args()
    return args

class SentenceDataset(Dataset):
    def __init__(self, database, label_columns):
        self.database = database
        self.label_columns = label_columns

    def __len__(self):
        return self.database.shape[0]

    def __getitem__(self, idx):
        sentence = self.database["abstract&claim"][idx]
        labels = self.database.loc[idx, self.label_columns]
        labels = np.array(labels, dtype=float)
        return sentence, labels

class CustomBERTModel(nn.Module):
    def __init__(self, num_labels, pretrained_model, num_encode_layer):
        super(CustomBERTModel, self).__init__()
        # self.bert_model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True, return_dict=True)
        self.textcnn = TextCNN(num_encode_layer)
        self.linear = nn.Linear(self.textcnn.num_filter_total, num_labels)
        self.bert_model = pretrained_model
    def forward(self, input_ids, attention_mask):
        sequence_output = self.bert_model(input_ids, attention_mask=attention_mask)
        hidden_states = sequence_output.hidden_states
        cls_embeddings = hidden_states[1][:, 0, :].unsqueeze(1)
        for i in range(2, len(hidden_states)):
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        logits = self.textcnn(cls_embeddings)
        return logits 

class TextCNN(nn.Module):
    def __init__(self, num_encode_layer):
        super(TextCNN, self).__init__()
        self.num_filters = 3
        self.filter_sizes = [2, 2, 2]
        self.hidden_size = 768
        self.encode_layer = num_encode_layer
        self.n_class = 47
        self.num_filter_total = self.num_filters * len(self.filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, self.n_class, bias=False)
        self.bias = nn.Parameter(torch.ones([self.n_class]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, kernel_size=(size, self.hidden_size)) for size in self.filter_sizes
        ])

    def forward(self, x):
        x = x.unsqueeze(1)
        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(x))
            mp = nn.MaxPool2d(kernel_size=(self.encode_layer - self.filter_sizes[i] + 1, 1))
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)
        
        h_pool = torch.cat(pooled_outputs, len(self.filter_sizes))
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])
        output = self.Weight(h_pool_flat) + self.bias
        # print(output.shape)
        return output

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
    avg_train_loss = train_loss / len(iterator)
    print('avg_train_loss: %f' % (avg_train_loss))
    return avg_train_loss
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

def plot_fig(file, title, x_label, y_label, save_path):
    plt.plot(file)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(save_path)
    plt.close()

def main():
    # Parse command line arguments
    args = parse_arguments()

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
    
    if args.model == 'bert':
        pretrained_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True, return_dict=True)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        num_encode_layer = 12
    elif args.model == 'roberta':
        pretrained_model = RobertaModel.from_pretrained("roberta-base", output_hidden_states=True, return_dict=True)
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        num_encode_layer = 12
    elif args.model == 'distilbert':
        pretrained_model = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True, return_dict=True)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        num_encode_layer = 6
    elif args.model == 'electra':
        pretrained_model = ElectraModel.from_pretrained("google/electra-base-discriminator", output_hidden_states=True, return_dict=True)
        tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
        num_encode_layer = 12
    #define model & tokenizer
    # tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    # model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=len(LABEL_COLUMNS))
    # model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(LABEL_COLUMNS))
    # tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = 512)
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    #model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=len(LABEL_COLUMNS))#(num_labels=len(LABEL_COLUMNS))
    model = CustomBERTModel(len(LABEL_COLUMNS), pretrained_model, num_encode_layer)
    # tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/electra-base-emotion")
    # model = ElectraForSequenceClassification.from_pretrained("bhadresh-savani/electra-base-emotion", num_labels=len(LABEL_COLUMNS), problem_type="multi_label_classification",ignore_mismatched_sizes=True)
    # tokenizer = AutoTokenizer.from_pretrained("roberta-base")
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
    torch.save(model.state_dict(),'/home/p76101372/patent_classification/model/bertCNN_param.pt')
    #model.save_pretrained('/home/p76101372/patent_classification/model/finetune_distilbert_model_new')
    tokenizer.save_pretrained('/home/p76101372/patent_classification/tokenizer/bertCNN_tokenizer')

    #plot fig and save
    plot_fig(train_loss_list, 'train_loss', 'epoch', 'loss', '/home/p76101372/patent_classification/img/bertCNN_train_loss.png')
    plot_fig(val_loss_list, 'val_loss', 'epoch', 'loss', '/home/p76101372/patent_classification/img/bertCNN_val_loss.png')
    plot_fig(val_acc_list, 'val_acc', 'epoch', 'acc', '/home/p76101372/patent_classification/img/bertCNN_val_acc.png')
    plot_fig(val_f1_list, 'val_f1', 'epoch', 'f1', '/home/p76101372/patent_classification/img/bertCNN_val_f1.png')

if __name__ == '__main__':
    main()
    
