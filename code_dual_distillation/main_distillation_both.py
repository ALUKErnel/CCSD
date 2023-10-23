

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertModel, TrOCRConfig
import transformers
transformers.logging.set_verbosity_error()
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import random
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 
from tqdm import tqdm 
import logging

from data_prep.xstance_dataset_student import get_datasets_main, get_datasets_target

from modules import *
from options import opt



from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler




random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)


if not os.path.exists(opt.student_model_save_file):
    os.makedirs(opt.student_model_save_file)
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
logging.basicConfig(level=logging.INFO if opt.local_rank in [-1, 0] else logging.WARN)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.student_model_save_file, 'student_log.txt'))
log.addHandler(fh)


log.info('Fine-tuning mBERT with options:')
log.info(opt)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
favor_id = tokenizer._convert_token_to_id("favor") # 19122
against_id = tokenizer._convert_token_to_id("against") # 11327
    


def get_metrics_f1(y_true, y_pred):
    average = 'macro'
    # f1 = f1_score(y_true, y_pred, average=average)
    f1_1 = f1_score(y_true == 1, y_pred == 1, labels=True)
    log.info('favor f1: {}'.format(100 * f1_1))
    f1_0 = f1_score(y_true == 0, y_pred == 0, labels=True)
    log.info('against f1: {}'.format(100 * f1_0))
    f1_avg = (f1_1 + f1_0) / 2
    # print("classification report: \n", classification_report(y_true, y_pred, digits=4))
    return f1_avg 



def train(opt):

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    train_file_path = os.path.join(opt.data_dir, "train.jsonl")
    valid_file_path = os.path.join(opt.data_dir, "valid.jsonl")
    test_file_path = os.path.join(opt.data_dir, "test.jsonl")

    # sub dataset and target setting 
    if opt.sub_dataset == "politics":
        topic_dict = {"Foreign Policy": 4, "Immigration": 5}
        if opt.target_setting == "all":
            src_target_list = [15, 16, 17, 18, 19, 20, 35, 59, 60, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3391, 3427, 3428, 3429, 3430, 3431, 3468, 3469, 3470, 3471] 
            tgt_target_list = [15, 16, 17, 18, 19, 20, 35, 59, 60, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3391, 3427, 3428, 3429, 3430, 3431, 3468, 3469, 3470, 3471] 
        elif opt.target_setting == "partial":
            src_target_list = [15, 16, 17, 18, 19, 35, 59, 60, 62, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3427, 3428, 3429, 3430, 3468, 3470]
            tgt_target_list = [15, 18, 19, 20, 59, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 2715, 3391, 3427, 3429, 3430, 3431, 3469, 3471] 
        else:
            src_target_list = [15, 16, 20, 59, 61, 62, 63, 1449, 1493, 1495, 1497, 3391, 3431, 3469, 3470, 3471]
            tgt_target_list = [17, 18, 19, 35, 60, 64, 1452, 1453, 1496, 2715, 3427, 3428, 3429, 3430, 3468]

    else:
        topic_dict = {"Security": 7, "Society": 8}
        if opt.target_setting == "all":
            src_target_list = [21, 22, 24, 25, 26, 53, 54, 55, 56, 57, 58, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1487, 1488, 1489, 1490, 1491, 1492, 2716, 3392, 3398, 3432, 3433, 3435, 3461, 3462]
            tgt_target_list = [21, 22, 24, 25, 26, 53, 54, 55, 56, 57, 58, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1487, 1488, 1489, 1490, 1491, 1492, 2716, 3392, 3398, 3432, 3433, 3435, 3461, 3462]
        elif opt.target_setting == "partial":
            src_target_list = [21, 24, 25, 26, 53, 54, 56, 58, 1454, 1455, 1456, 1457, 1487, 1488, 1489, 1490, 1491, 2716, 3392, 3398, 3433, 3435, 3461, 3462]
            tgt_target_list = [22, 53, 54, 55, 56, 57, 58, 1454, 1456, 1457, 1458, 1459, 1460, 1487, 1488, 1489, 1490, 1491, 1492, 3392, 3398, 3432, 3433, 3435]
        else:
            src_target_list = [22, 25, 53, 54, 57, 58, 1455, 1456, 1457, 1459, 1490, 1491, 1492, 2716, 3432, 3433] 
            tgt_target_list = [21, 24, 26, 55, 56, 1454, 1458, 1460, 1487, 1488, 1489, 3392, 3398, 3435, 3461, 3462]

    # dataset for batch 
    de_targets, fr_targets, \
        de_train_dataset, de_valid_dataset, de_test_dataset, \
            fr_train_dataset, fr_valid_dataset, fr_test_dataset = get_datasets_main(train_file_path, valid_file_path, test_file_path, tokenizer, opt.num_train_lines, opt.max_seq_len, src_target_list, tgt_target_list, topic_dict)

    print(de_targets)
    print(fr_targets)
    
    # de(src) dataset per target
    # datasets_target_list = get_datasets_target(train_file_path, valid_file_path, test_file_path, tokenizer, opt.num_train_lines, opt.max_seq_len, de_targets, fr_targets, topic_dict)

    opt.num_target = len(de_targets)
    opt.num_labels = de_train_dataset.num_labels 

    log.info("Done loading datasets.")
    

    fr_train_loader = DataLoader(fr_train_dataset, opt.student_batch_size, shuffle=True, drop_last=True) 
    fr_valid_loader = DataLoader(fr_valid_dataset, opt.student_batch_size, shuffle=False) 
    fr_test_loader = DataLoader(fr_test_dataset, opt.student_batch_size, shuffle=False) 
    
    
    
    log.info('Done constructing DataLoader. ')

    teacher_X_param = torch.load('{}/cluster_initialized_{}_{}/teacher_model_X.pth'.format(opt.cross_target_teacher_model_load_file, opt.sub_dataset, opt.target_setting))
    teacher_P_param = torch.load('{}/cluster_initialized_{}_{}/teacher_model_P.pth'.format(opt.cross_target_teacher_model_load_file, opt.sub_dataset, opt.target_setting))

    teacher_CL_param = torch.load('{}/mbert_{}_{}/prompt-fine-tuned-model.pth'.format(opt.cross_lingual_teacher_model_load_file, opt.sub_dataset, opt.target_setting)) 


    teacher_X = EmbeddingModule(opt.tokenized_max_len, None, False)
    teacher_P = StanceClassifier(opt.P_layers, opt.hidden_size, opt.num_labels, opt.concat_stance, opt.dropout, opt.P_bn)
    teacher_CL = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")


    teacher_X.load_state_dict(teacher_X_param)
    teacher_P.load_state_dict(teacher_P_param)
    teacher_CL.load_state_dict(teacher_CL_param)

    teacher_X = teacher_X.to(opt.device)
    teacher_P = teacher_P.to(opt.device)
    teacher_CL = teacher_CL.to(opt.device)

    student_X = EmbeddingModule(opt.tokenized_max_len, None, False)
    student_P = StanceClassifier(opt.P_layers, opt.hidden_size, opt.num_labels, opt.concat_stance, opt.dropout, opt.P_bn)

    student_X = student_X.to(opt.device)
    student_P = student_P.to(opt.device)
    
    # optimizer_teacher = optim.Adam(list(teacher_X.parameters() + list(teacher_P.parameters())), lr=opt.teacher_learning_rate)
    optimizer_student = optim.Adam( list(student_X.parameters()) + list(student_P.parameters()) , lr=opt.student_learning_rate)
    
    log.info('Done loading models. ')
    
    # training 
    best_f1 = 0.0
    label2id = {"favor":1, "against": 0}
    
    log.info('Teacher-Student Distillation Begin! ')
    for epoch in range(opt.student_max_epoch): 
        teacher_X.eval()
        teacher_P.eval()
        teacher_CL.eval()
        student_X.train()
        student_P.train()

        train_iter = iter(fr_train_loader)
        correct, total = 0, 0
        total_loss = 0.0
        cross_entropy_loss = CELoss()
        max_iter_per_epoch = len(fr_train_dataset) // opt.student_batch_size   
        for i, (inputs_prompt, inputs, y_prompt, y_t, y, y_l) in tqdm(enumerate(train_iter), total=max_iter_per_epoch):

            student_X.zero_grad()
            student_P.zero_grad()
            
            y = y.to(opt.device)
            y_t = y_t.to(opt.device)
            y_l = y_l.to(opt.device)
            
            

            # teacher soft supervision signals
            with torch.no_grad():
                embeds_teacher = teacher_X(inputs)
                outputs_teacher = teacher_P(embeds_teacher)
            
            with torch.no_grad():
                tokenized_inputs = tokenizer(inputs_prompt, padding=True, return_tensors="pt").to(opt.device)
                tokenized_labels = tokenizer(y_prompt, padding=True, return_tensors="pt")["input_ids"].to(opt.device)
                labels_compute = torch.where(tokenized_inputs.input_ids == tokenizer.mask_token_id, tokenized_labels, -100)
                mask_token_index = (tokenized_inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
                outputs = teacher_CL(**tokenized_inputs)
                
                logits = outputs.logits
                # probs_favor = logits[mask_token_index[0], mask_token_index[1]][:, favor_id]
                # probs_against = logits[mask_token_index[0], mask_token_index[1]][:, against_id]
                # probs = torch.vstack((probs_favor, probs_against)).reshape(-1,2).softmax(dim=-1)
                predicted_token_id = logits[mask_token_index[0], mask_token_index[1]].argmax(axis=-1)
                predicted_token_batch = tokenizer.batch_decode(predicted_token_id)

                probs = torch.zeros(opt.student_batch_size, 2).to(opt.device)


                for i, pt in enumerate(predicted_token_batch):
                    predicted_token = "".join(pt.split(' '))
                    if predicted_token not in ["favor", "against"]:
                        predicted_token = "favor"
                    probs[i][label2id[predicted_token]] = 1


            embeds_student = student_X(inputs)
            outputs_stance = student_P(embeds_student)
            log_outputs_stance = torch.log(outputs_stance)
            loss_ct = cross_entropy_loss(log_outputs_stance, outputs_teacher)

            loss_cl = cross_entropy_loss(log_outputs_stance, probs)
            loss = opt.alpha * loss_ct + loss_cl
            total_loss += loss.item()
            
            # correct 
            _, pred = torch.max(outputs_stance, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            # print("loss: ", loss.item())
            loss.backward()
            optimizer_student.step()
            
            
        # end of epoch
        log.info('Student Ending epoch {}'.format(epoch+1))
        log.info('Student Loss: {}'.format(total_loss/max_iter_per_epoch))
        log.info('Student Training Accuracy: {}%'.format(100.0*correct/total))
        
        log.info('Student Evaluating on valid set:')
        acc, f1 = evaluate(opt, fr_valid_loader, student_X, student_P)
        
        if f1 > best_f1:
            log.info('Student Best f1 has been updated as {}'.format(f1))
            best_f1 = f1
            # best_teacher = teacher 
            
        log.info('Student Evaluating on test set:')
        acc, f1 = evaluate(opt, fr_test_loader, student_X, student_P)
        
        
    log.info('Student Best valid f1 is {}'.format(best_f1))
    


def evaluate(opt, data_loader, X, P):
    X.eval()
    P.eval()
    iter1 = iter(data_loader)
    correct, total = 0, 0
    preds = []
    labels = []
    with torch.no_grad():
        for inputs_prompt, inputs, y_prompt, y_t, y, y_l in tqdm(iter1):
            
            y = y.to(opt.device)
            y_t = y_t.to(opt.device)
            y_l = y_l.to(opt.device)

            embeds = X(inputs)
            outputs_stance = P(embeds)
            _, pred = torch.max(outputs_stance, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            preds.append(pred)
            labels.append(y)
    
    y_pred = torch.cat(preds, dim=0).cpu()
    y_true = torch.cat(labels, dim=0).cpu()
    f1 = get_metrics_f1(y_true, y_pred)
    accuracy = correct / total
    log.info('Accuracy on {} samples: {}%'.format(total, 100.0*accuracy))
    log.info('f1 on {} samples: {}'.format(total, 100.0*f1))
    return accuracy, f1
    
 
if __name__ == '__main__':
    
    train(opt)

    



    

