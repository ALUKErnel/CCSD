
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertModel, BertForMaskedLM
import transformers
transformers.logging.set_verbosity_error()
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import random
import numpy as np
from tqdm import tqdm 
import logging

from data_prep.xstance_dataset import get_datasets_main, get_datasets_target

from modules import *
from criterions import *
from options import opt
from utils2 import * 



from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler




random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)


if not os.path.exists(opt.cross_target_teacher_model_save_file):
    os.makedirs(opt.cross_target_teacher_model_save_file)
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
logging.basicConfig(level=logging.INFO if opt.local_rank in [-1, 0] else logging.WARN)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.cross_target_teacher_model_save_file, 'log.txt'))
log.addHandler(fh)


log.info('Fine-tuning mBERT with options:')
log.info(opt)
    
    
# opt.batch_size_target 
def get_target_feature(datasets_target_list, target_list, model_target, tokenizer):
    all_target_feature = torch.zeros([len(target_list), opt.emb_size]).to(opt.device)     
    for ti in range(len(target_list)):
        log.info(f"{ti}th target dataset begin encode")
        curr_dataset = datasets_target_list[ti]
        data_loader = DataLoader(curr_dataset, opt.batch_size_target, shuffle=False)
        cls_list = []
        model_target.eval()
        iter1 = iter(data_loader)
        max_iter1 = len(data_loader) // opt.batch_size_target 
        with torch.no_grad():
            for i, (inputs, y) in tqdm(enumerate(iter1), total=max_iter1):
                model_target.zero_grad()
            
           
                tokenized_inputs = tokenizer(inputs[1], padding=True, truncation=True, max_length=200, return_tensors="pt").to(opt.device)
                outputs = model_target(**tokenized_inputs, output_hidden_states=True)

                output_cls = outputs.hidden_states[12][:, 0] # batch * seq_len * 768 # cls 
            
                cls_list.append(output_cls)
        
        cls_list_tensor = torch.vstack(cls_list)
        cls_mean = torch.mean(cls_list_tensor, dim=0)
        all_target_feature[ti] = cls_mean 
        
    
    return all_target_feature   # n x emb_size


def get_question_feature(datasets_target_list, target_list, model, tokenizer):
    '''
    get the textual representation for the target 
    and the encoder is optimized each epoch 
    '''
    all_question_feature = torch.zeros([len(target_list), opt.emb_size]).to(opt.device)  
    for ti in range(len(target_list)):
        log.info(f"{ti}th target begin encode")
        curr_dataset = datasets_target_list[ti]
        # 此处无需batch和dataloader
        inputs, y = curr_dataset[0]
        model.eval()
        model.zero_grad()
        tokenized_inputs = tokenizer(inputs[0], padding=True, truncation=True, max_length=200, return_tensors="pt").to(opt.device)
        outputs = model(**tokenized_inputs, output_hidden_states=True)

        output_cls = outputs.hidden_states[12][:, 0].view(-1, 768) # batch * seq_len * 768 # cls 

        all_question_feature[ti] = output_cls

    return all_question_feature # n * emb_size


def calculate_similarity(all_target_feature, measurement):
    # mask = (torch.ones([opt.num_target, opt.num_target])- torch.eye(opt.num_target)).to(opt.device)
    if measurement == "dot product":
        similarity_matrix = torch.matmul(all_target_feature, all_target_feature.T) 
        # similarity_matrix = similarity_matrix * mask
        
        
    elif measurement == "cosine similarity":
        norm_all_target_feature = all_target_feature / torch.norm(all_target_feature, p=2, dim=-1, keepdim=True)
        similarity_matrix = torch.matmul(norm_all_target_feature, norm_all_target_feature.T) 
        # similarity_matrix = similarity_matrix * mask
    
    elif measurement == "fully-connected":
        similarity_matrix = torch.ones([opt.num_target, opt.num_target])
        
    else:
        print("Error measurement in calculating similarity!")
        similarity_matrix = torch.ones([opt.num_target, opt.num_target])
    
    return similarity_matrix 
        
def calculate_adj(similarity_matrix, threshold):
    adj = torch.where(similarity_matrix>threshold, 1, 0)
    
    return adj 

def data_normal(data):
    d_min = data.min()
    if d_min < 0:
        data += torch.abs(d_min)
        d_min = data.min()
    d_max = data.max()
    dst = d_max - d_min
    norm_data = (data - d_min).true_divide(dst)
    return norm_data 


def distance(point1, point2):  
    return np.sqrt(np.sum((point1 - point2) ** 2))

def k_means(data_torch, k, max_iter=10000):
    '''
    data: target_vector: n_s x emb_size 
    '''
    data = data_torch.cpu().numpy()
    n_target = data.shape[0]
    data_id = range(n_target)
    centers = {}  
    n_data = data.shape[0]   
    for idx, i in enumerate(random.sample(range(n_data), k)):
        centers[idx] = data[i]  

    for i in range(max_iter):  
        clusters = {}   
        clusters_id = {} 
        for j in range(k):  
            clusters[j] = []
            clusters_id[j] = []
            
        for ii, (idd, sample) in enumerate(zip(data_id,data)):  
            distances = []  
            for c in centers:  
                distances.append(distance(sample, centers[c])) 
            idx = np.argmin(distances)  
            clusters[idx].append(sample)  
            clusters_id[idx].append(idd) 
            
        pre_centers = centers.copy()  

        for c in clusters.keys():
            centers[c] = np.mean(clusters[c], axis=0)
  
        is_convergent = True
        for c in centers:
            if distance(pre_centers[c], centers[c]) > 1e-8:  
                is_convergent = False
                break
        if is_convergent == True:  

            break
    return centers, clusters, clusters_id

def convert_clusters_to_yd(clusters_id, target_label):

    # reverse the clusters dict result 
    cluster_ids_reverse = {}
    for key, value in clusters_id.items():
        for item in value:
            cluster_ids_reverse[item] = key

    batch_size = target_label.shape[0]
    y_d = torch.zeros(batch_size)

    for j, tl in enumerate(target_label.tolist()):
        y_d[j] = cluster_ids_reverse[tl]
    
    return y_d 

def combine_target_rep(target_vectors, question_features):
    return torch.hstack((target_vectors, question_features)) # n x 2emb


def get_category_rep(cluster_ids, target_reps):
    category_reps = torch.zeros(opt.k, opt.emb_size*2 )
    for i, (k,v) in enumerate(cluster_ids.items()):
        # concatenate 
        cluster_reps = target_reps[v]
        category_reps[i] = torch.mean(cluster_reps, dim=0) 

    return category_reps


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

    model_target = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
    param = torch.load('{}/mbert_{}_{}/prompt-fine-tuned-model.pth'.format(opt.cross_lingual_teacher_model_load_file, opt.sub_dataset, opt.target_setting)) # opt.model_load_file
    model_target.load_state_dict(param)
    model_target = model_target.to(opt.device)

    freeze_net(model_target)
    
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
    datasets_target_list = get_datasets_target(train_file_path, valid_file_path, test_file_path, tokenizer, opt.num_train_lines, opt.max_seq_len, de_targets, fr_targets, topic_dict)

    opt.num_target = len(de_targets)
    opt.num_labels = de_train_dataset.num_labels 

    log.info("Done loading datasets.")
    
    
    de_train_loader = DataLoader(de_train_dataset, opt.teacher_batch_size, shuffle=True, drop_last=True) 
    de_valid_loader = DataLoader(de_valid_dataset, opt.teacher_batch_size, shuffle=False) 
    de_test_loader = DataLoader(de_test_dataset, opt.teacher_batch_size, shuffle=False) 
    
    
    
    log.info('Done constructing DataLoader. ')

    param = torch.load('{}/mbert_{}_{}/prompt-fine-tuned-model.pth'.format(opt.cross_lingual_teacher_model_load_file, opt.sub_dataset, opt.target_setting)) # opt.model_load_file
    
    X = EmbeddingModule(opt.tokenized_max_len, param, True)
    X = X.to(opt.device)

    P = StanceClassifier(opt.P_layers, opt.hidden_size, opt.num_labels, opt.concat_stance, opt.dropout, opt.P_bn)
    P = P.to(opt.device)


    Gt = GAT(opt.emb_size, opt.gnn_dims, opt.att_heads, opt.attn_dropout, opt.concat_dropout, opt.leaky_alpha) 
    Gt = Gt.to(opt.device)

    DCL = DomainContrastiveLoss(opt.temperature)
    RLoss = Refinement()

    optimizer_teacher = optim.Adam(list(X.parameters()) + 
                                   list(Gt.parameters()) +
                                   list(P.parameters()), lr=opt.teacher_learning_rate)
   
    
    log.info('Done loading models. ')
    
    # target associations graph
    node_features = get_target_feature(datasets_target_list, de_targets, model_target, tokenizer)
    question_features = get_question_feature(datasets_target_list, de_targets, model_target, tokenizer) # n_target x emb_size
    similarity_matrix = calculate_similarity(node_features, opt.measurement)
    normed_similarity_matrix = data_normal(similarity_matrix)
    adj = calculate_adj(normed_similarity_matrix, opt.sim_threshold)
    # adj = torch.ones(opt.num_target, opt.num_target).to(opt.device)

    
    
    # training 
    best_f1 = 0.0
    best_teacher_X = X
    best_teacher_P = P

    
    for epoch in range(opt.teacher_max_epoch):

        # target semantic representation
       
        X.train()
        Gt.train()
        P.train()
        train_iter = iter(de_train_loader)
        correct, total = 0, 0
        total_loss = 0.0
        max_iter_per_epoch = len(de_train_dataset) // opt.teacher_batch_size   
        for i, (inputs,y_t, y, y_l) in tqdm(enumerate(train_iter), total=max_iter_per_epoch):

            X.zero_grad()
            Gt.zero_grad()
            P.zero_grad()
            
            y = y.to(opt.device)
            y_t = y_t.to(opt.device)
            y_l = y_l.to(opt.device)

            embeds = X(inputs)
            target_vectors, weight_matrix = Gt(node_features, adj)
            weight_matrix.detach()

            # target vectors clustering k=3
            centers, clusters, clusters_id = k_means(target_vectors.detach(), opt.k)
            y_d = convert_clusters_to_yd(clusters_id, y_t)
            y_d = y_d.to(opt.device)

            target_reps = torch.hstack((target_vectors, question_features)).to(opt.device)
            
            category_reps = get_category_rep(clusters_id, target_reps).to(opt.device)


            r_loss = RLoss(target_reps, category_reps.detach(), clusters_id)

            features =torch.cat([embeds, target_vectors[y_t]], -1)
            domain_con_loss = DCL(features, y, y_t, y_d) # 此处暂时没带domain loss
            
            outputs_stance = P(embeds)
            log_outputs_stance = torch.log(outputs_stance)
            ce_loss = F.nll_loss(log_outputs_stance, y)
            teacher_loss = ce_loss + domain_con_loss + 0.0001 * r_loss

            total_loss += teacher_loss.item()
            
            # correct 
            _, pred = torch.max(outputs_stance, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            # print("loss: ", loss.item())
            teacher_loss.backward()
            optimizer_teacher.step()
            
            
        # end of epoch
        log.info('Ending epoch {}'.format(epoch+1))
        log.info('Loss: {}'.format(total_loss/max_iter_per_epoch))
        log.info('Training Accuracy: {}%'.format(100.0*correct/total))
        
        log.info('Evaluating on valid set:')
        acc, f1 = evaluate(opt, de_valid_loader, X, P)
        
        if f1 > best_f1:
            log.info('Best f1 has been updated as {}'.format(f1))
            best_f1 = f1
            best_teacher_X = X
            best_teacher_P = P

           
            
        log.info('Evaluating on test set:')
        acc, f1 = evaluate(opt, de_test_loader, X, P)
        
        
    log.info('Teacher Best valid f1 is {}'.format(best_f1))

    torch.save(best_teacher_X.state_dict(),'{}/teacher_model_X.pth'.format(opt.cross_target_teacher_model_save_file)) 
    torch.save(best_teacher_P.state_dict(),'{}/teacher_model_P.pth'.format(opt.cross_target_teacher_model_save_file)) 
    
            
def evaluate(opt, data_loader, X, P):
    X.eval()
    P.eval()
    iter1 = iter(data_loader)
    correct, total = 0, 0
    preds = []
    labels = []
    with torch.no_grad():
        for inputs, y_t, y, y_l in tqdm(iter1):
            
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

    



    

