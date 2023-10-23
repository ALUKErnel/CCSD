

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import codecs
import json
import random
import logging
from sklearn.utils import shuffle
from typing import List, Dict

from data_prep.xstance_dataset import get_datasets_main
from options import opt
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
from transformers import (
    Trainer,
    BertTokenizer,
    TrainingArguments,
    BertForMaskedLM,
    EarlyStoppingCallback
)


random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)


if not os.path.exists(opt.model_save_file):
    os.makedirs(opt.model_save_file)
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
logging.basicConfig(level=logging.INFO if opt.local_rank in [-1, 0] else logging.WARN)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
log.addHandler(fh)


log.info('Fine-tuning mBERT with options:')
log.info(opt)

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")




# def compute_metrics(pred):
#     labels = pred.label_ids[:, 3]
#     preds = pred.predictions[:, 3].argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     acc = accuracy_score(labels, preds)
#     return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def get_metrics_f1(y_true, y_pred):
    average = 'macro'
    # f1 = f1_score(y_true, y_pred, average=average)
    f1_1 = f1_score(y_true == 1, y_pred == 1, labels=True)
    log.info('favor f1: {}'.format(100 * f1_1))
    f1_0 = f1_score(y_true == 0, y_pred == 0, labels=True)
    log.info('against f1: {}'.format(100 * f1_0))
    f1_avg = (f1_1 + f1_0) / 2
    acc = accuracy_score(y_true, y_pred)
    # print("classification report: \n", classification_report(y_true, y_pred, digits=4))
    return acc, f1_avg



def train(opt):

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased", num_labels=2)
    model = model.to(opt.device)

    train_file_path = os.path.join(opt.data_dir, "train.jsonl")
    valid_file_path = os.path.join(opt.data_dir, "valid.jsonl")
    test_file_path = os.path.join(opt.data_dir, "test.jsonl")

    # sub dataset and target setting 
    if opt.sub_dataset == "political":
        topic_dict = {"Foreign Policy": 4, "Immigration": 5}
        if opt.target_setting == "all":
            src_target_list = [15, 16, 17, 18, 19, 20, 35, 59, 60, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3391, 3427, 3428, 3429, 3430, 3431, 3468, 3469, 3470, 3471] 
            tgt_target_list = [15, 16, 17, 18, 19, 20, 35, 59, 60, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3391, 3427, 3428, 3429, 3430, 3431, 3468, 3469, 3470, 3471] 
        elif opt.target_setting == "partial":
            src_target_list = [15, 16, 17, 18, 19, 35, 59, 60, 62, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3427, 3428, 3429, 3430, 3468, 3470]
            tgt_target_list = [15, 18, 19, 20, 59, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 2715, 3391, 3427, 3429, 3430, 3431, 3469, 3471] 
        else: # none
            src_target_list = [15, 16, 20, 59, 61, 62, 63, 1449, 1493, 1495, 1497, 3391, 3431, 3469, 3470, 3471]
            tgt_target_list = [17, 18, 19, 35, 60, 64, 1452, 1453, 1496, 2715, 3427, 3428, 3429, 3430, 3468]

    else: # social
        topic_dict = {"Security": 7, "Society": 8}
        if opt.target_setting == "all":
            src_target_list = [21, 22, 24, 25, 26, 53, 54, 55, 56, 57, 58, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1487, 1488, 1489, 1490, 1491, 1492, 2716, 3392, 3398, 3432, 3433, 3435, 3461, 3462]
            tgt_target_list = [21, 22, 24, 25, 26, 53, 54, 55, 56, 57, 58, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1487, 1488, 1489, 1490, 1491, 1492, 2716, 3392, 3398, 3432, 3433, 3435, 3461, 3462]
        elif opt.target_setting == "partial":
            src_target_list = [21, 24, 25, 26, 53, 54, 56, 58, 1454, 1455, 1456, 1457, 1487, 1488, 1489, 1490, 1491, 2716, 3392, 3398, 3433, 3435, 3461, 3462]
            tgt_target_list = [22, 53, 54, 55, 56, 57, 58, 1454, 1456, 1457, 1458, 1459, 1460, 1487, 1488, 1489, 1490, 1491, 1492, 3392, 3398, 3432, 3433, 3435]
        else: # none
            src_target_list = [22, 25, 53, 54, 57, 58, 1455, 1456, 1457, 1459, 1490, 1491, 1492, 2716, 3432, 3433] 
            tgt_target_list = [21, 24, 26, 55, 56, 1454, 1458, 1460, 1487, 1488, 1489, 3392, 3398, 3435, 3461, 3462]

    # dataset for batch 

    de_train_dataset, de_valid_dataset, de_test_dataset  = get_datasets_main(train_file_path, valid_file_path, test_file_path, \
                                                     opt.num_train_lines, opt.max_seq_len, src_target_list, tgt_target_list, topic_dict)
    

    opt.num_labels = de_train_dataset.num_labels 
    label2id = {"favor":1, "against": 0}

    log.info("Done loading datasets.")
    
    de_train_loader = DataLoader(de_train_dataset, opt.batch_size, shuffle=True, drop_last=True) 
    de_valid_loader = DataLoader(de_valid_dataset, opt.batch_size, shuffle=False) 
    de_test_loader = DataLoader(de_test_dataset, opt.batch_size, shuffle=False)

    log.info('Done constructing DataLoader. ')
    
    optimizer = optim.Adam(list(model.parameters()), lr=opt.learning_rate)
    
    log.info('Done loading models. ')

    best_f1 = 0.0
    best_model = model

    for epoch in range(opt.max_epoch):
        model.train()
        train_iter = iter(de_train_loader)
        correct, total = 0, 0
        total_loss = 0.0
        
        max_iter_per_epoch = len(de_train_dataset) // opt.batch_size   
        for i, (inputs,labels, y) in tqdm(enumerate(train_iter), total=max_iter_per_epoch):

            model.zero_grad()
           
            y = y.to(opt.device)

            tokenized_inputs = tokenizer(inputs, padding=True, return_tensors="pt").to(opt.device)
            tokenized_labels = tokenizer(labels, padding=True, return_tensors="pt")["input_ids"].to(opt.device)
            labels = torch.where(tokenized_inputs.input_ids == tokenizer.mask_token_id, tokenized_labels, -100)
            outputs = model(**tokenized_inputs, labels=labels)
            
            # logits = model(**inputs).logits

            # index of [mask]
            # mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

            loss = outputs.loss
            total_loss += loss.item()


            # accuracy
            logits = outputs.logits 
            # mask_token_index = (tokenized_inputs.)



            loss.backward()
            optimizer.step()

        log.info('Ending epoch {}'.format(epoch+1))
        log.info('Loss: {}'.format(total_loss/max_iter_per_epoch))
        # log.info('Training Accuracy: {}%'.format(100.0*correct/total))
        
        log.info('Evaluating on valid set:')
        acc, f1 = evaluate(opt, de_valid_loader, model)
        
        if f1 > best_f1:
            log.info('Best f1 has been updated as {}'.format(f1))
            best_f1 = f1
            best_model = model
            
        log.info('Evaluating on test set:')
        acc, f1 = evaluate(opt, de_test_loader, model)
        
        
    log.info('Best valid f1 is {}'.format(best_f1))
    torch.save(best_model.state_dict(),
                    '{}/prompt-fine-tuned-model.pth'.format(opt.model_save_file)) 

def evaluate(opt, data_loader, model):

    model.eval()
    iter1 = iter(data_loader)
    correct, total = 0, 0
    preds = []
    labels_ = []
    label2id = {"favor":1, "against": 0}
    other_predicted_token = []
    with torch.no_grad():
        for inputs, labels, y in tqdm(iter1):
          
            y = y.to(opt.device)

            tokenized_inputs = tokenizer(inputs, padding=True, return_tensors="pt").to(opt.device)
            tokenized_labels = tokenizer(labels, padding=True, return_tensors="pt")["input_ids"].to(opt.device)
            
            labels = torch.where(tokenized_inputs.input_ids == tokenizer.mask_token_id, tokenized_labels, -100)
            
            logits = model(**tokenized_inputs).logits

            mask_token_index = (tokenized_inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)

            predicted_token_id = logits[mask_token_index[0], mask_token_index[1]].argmax(axis=-1)
            predicted_token_batch = tokenizer.batch_decode(predicted_token_id)

            for i, pt in enumerate(predicted_token_batch):
                predicted_token = "".join(pt.split(' '))
                if predicted_token not in ["favor", "against"]:
                    other_predicted_token.append(predicted_token)
                    predicted_token = "favor"
                    
                preds.append(label2id[predicted_token])

            labels_.append(y)
            total += y.size(0)
    
    y_pred = torch.tensor(preds).cpu()
    y_true = torch.cat(labels_, dim=0).cpu()
    acc, f1 = get_metrics_f1(y_true, y_pred)
    log.info('Accuracy on {} samples: {}%'.format(total, 100.0*acc))
    log.info('f1 on {} samples: {}'.format(total, 100.0*f1))
    return acc, f1


if __name__ == "__main__":
    train(opt)


            











    


   
    

   





