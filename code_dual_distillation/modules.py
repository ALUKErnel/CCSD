import torch
from torch import autograd, nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import XLMConfig, XLMTokenizer, XLMModel
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertModel, BertForMaskedLM
from layers import * 
from options import opt
import random
import numpy as np

random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)       
 
 
 
class EmbeddingModule(nn.Module):
    def __init__(self, max_length, params, teacher):
        super(EmbeddingModule, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.bert_model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
        self.max_length = max_length
        if teacher:
            self.bert_model.load_state_dict(params)
        
    def forward(self, inputs):
        tokenized_inputs = self.tokenizer(inputs[0], inputs[1], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(opt.device)
        outputs = self.bert_model(**tokenized_inputs, output_hidden_states=True) 
        # outputs_cls = outputs.pooler_output

        output_cls = outputs.hidden_states[12][:, 0] # batch * seq_len * 768 # cls 
        
        return output_cls       
    
    
    
    
class StanceClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 output_size,
                 concat,
                 dropout,
                 batch_norm=False):
        super(StanceClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        torch.manual_seed(opt.random_seed)
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0 and concat:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size*2, hidden_size))
            else:
                self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        self.net.add_module('p-logsoftmax', nn.Softmax(dim=-1))

    def forward(self, input):
        return self.net(input)
    

    
class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, log_probs, probs):
        middle = - probs * log_probs
        cross_entropy_loss = torch.mean(torch.sum(middle, dim=-1))

        return cross_entropy_loss






