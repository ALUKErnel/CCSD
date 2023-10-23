import argparse
import torch
import os
parser = argparse.ArgumentParser()

# path
parser.add_argument('--data_dir', default='./dataset/')
parser.add_argument('--model_save_file', default='./save_cross-lingual_teacher/mbert_political_none')

# random seed 
parser.add_argument('--random_seed', type=int, default=1)

# data readin
parser.add_argument('--num_train_lines', type=int, default=0)  # set to 0 to use all training data
parser.add_argument('--max_seq_len', type=int, default=250) # Dataset readin (settle)
parser.add_argument('--num_target', type=int, default=0) # set 0 to select all targets

# data setting
parser.add_argument('--sub_dataset', type=str, default='politics')  # sub_dataset: political/ social
parser.add_argument('--target_setting', type=str, default='none')  # target setting: all/ paritial/ none

# X EmbeddingModule
parser.add_argument('--tokenized_max_len', type=int, default=0) # Dataset readin 


# training 
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_epoch', type=int, default=15)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--local_rank', type=int, default=0)

opt = parser.parse_args()

if not torch.cuda.is_available():
    opt.device = 'cpu'

