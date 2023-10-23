

import torch
from torch import autograd, nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import XLMConfig, XLMTokenizer, XLMModel
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertModel
from layers import * 
from options import opt
import random
import numpy as np


    
class Refinement(nn.Module): 
    def __init__(self):
        super(Refinement,self).__init__()
    
    def forward(self, target_reps, category_reps, clusters_id):
        loss_e = 0.0 
        clusters_id_reverse = {} 
        for key, values in clusters_id.items():
            for v in values:
                clusters_id_reverse[v] = key  
            category_rep = category_reps[key]
            clusters_reps = target_reps[values]
            for rep in clusters_reps:
                loss_e += self.d(rep, category_rep)
        
        final_loss_e = loss_e.mean()

        t_c = list(clusters_id_reverse.values())
        t_c = torch.tensor(t_c)
        t_c = t_c.contiguous().view(-1, 1)
        mask_c = torch.eq(t_c, t_c.T).float().to(opt.device)
        
        all_distances = self.d_all(target_reps, target_reps)
        neg_mask = 1 - mask_c 
        ori_n = neg_mask * all_distances

        rev_n = opt.m - ori_n 
        other_set = torch.zeros_like(ori_n)

        loss_n = torch.where(rev_n > 0, rev_n, other_set)

        loss_n = torch.sum(loss_n, dim=1) 

        final_loss_n = loss_n.mean()

        return final_loss_e + final_loss_n 


    
    def d(self, a, b):
        return torch.sum((a-b)**2)

    def d_all(self, a, b):
        sq_a = a**2
        sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
        sq_b = b**2
        sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
        bt = b.t() 
        # return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))
        res = sum_sq_a+sum_sq_b-2*a.mm(bt)
        return res

   


class DomainContrastiveLoss(nn.Module): 
    def __init__(self, temperature, scale_by_temperature=True):
        super(DomainContrastiveLoss,self).__init__()
        self.temperature = temperature
        self.scale = scale_by_temperature
    
    def forward(self, features, labels, labels_t, labels_d):
        
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        
        
        labels = labels.contiguous().view(-1, 1)
        labels_t = labels_t.contiguous().view(-1, 1)
        labels_d = labels_d.contiguous().view(-1, 1)
    

        mask_y = torch.eq(labels, labels.T).float().to(opt.device)
        mask_y -= torch.eye(batch_size).to(opt.device)

        mask_t = torch.eq(labels_t, labels_t.T).float().to(opt.device)
        mask_d = torch.eq(labels_d, labels_d.T).float().to(opt.device)


        
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)


        # mask
        
        # mask_mask = target_positive_mask - torch.eye(batch_size).to(opt.device) 
        positive_mask = mask_y * mask_d
        negative_mask = 1 - positive_mask 
        negative_mask = negative_mask - torch.eye(batch_size).to(opt.device)
        
        #
        num_positives_per_row = torch.sum(positive_mask, dim=1)
        denominator = torch.sum(exp_logits * negative_mask, dim=1, keepdim=True) + \
            torch.sum(exp_logits * positive_mask, dim=1, keepdim=True)

        log_probs = logits - torch.log(denominator) # ??

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(log_probs * positive_mask, dim=1)[num_positives_per_row > 0] / \
            num_positives_per_row[num_positives_per_row>0]
        
        
        loss = - log_probs 
        if self.scale:
            loss *= self.temperature
        loss = loss.mean()
        return loss    
