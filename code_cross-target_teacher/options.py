import argparse
import torch
import os
parser = argparse.ArgumentParser()

# path
parser.add_argument('--data_dir', default='./dataset/')


# for cross-target teacher 
parser.add_argument('--cross_lingual_teacher_model_load_file', default='./save_cross-lingual_teacher')
parser.add_argument('--cross_target_teacher_model_save_file', default='./save_cross-target_teacher/cluster_initialized_politics_all')

# random seed 
parser.add_argument('--random_seed', type=int, default=1)

# data readin
parser.add_argument('--num_train_lines', type=int, default=0)  # set to 0 to use all training data
parser.add_argument('--max_seq_len', type=int, default=1000) # Dataset readin (settled)
parser.add_argument('--num_target', type=int, default=0) # set 0 to select all targets

# data setting
parser.add_argument('--sub_dataset', type=str, default='politics')  # sub_dataset: politics/ society
parser.add_argument('--target_setting', type=str, default='all')  # target setting: all/ paritial/ none


# preprocessing
parser.add_argument('--sim_threshold', type=float, default=0.4)
parser.add_argument('--measurement', type=str, default='cosine similarity')

# X EmbeddingModule
parser.add_argument('--tokenized_max_len', type=int, default=120) # Dataset readin 
parser.add_argument('--emb_size', type=int, default=768)

# G GAT 
parser.add_argument('--gnn_dims', type=str, default='192')
parser.add_argument('--att_heads', type=str, default='4')
parser.add_argument('--attn_dropout', type=float, default=0.2)
parser.add_argument('--concat_dropout', type=float, default=0.2)
parser.add_argument('--leaky_alpha', type=float, default=0.2)

# kmeans 
parser.add_argument('--k', type=int, default=3)

# refinement
parser.add_argument('--m', type=float, default=100)


# P StanceClassifier
parser.add_argument('--P_layers', type=int, default=2)
parser.add_argument('--P_bn', default=True)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.2)

# concat for stance classifier
parser.add_argument('--concat_stance', default=False)

# contrastive criterion
parser.add_argument('--temperature', type=float, default=0.3)


# training 
parser.add_argument('--teacher_learning_rate', type=float, default=2e-5)
parser.add_argument('--teacher_batch_size', type=int, default=32)
parser.add_argument('--batch_size_target', type=int, default=64)
parser.add_argument('--teacher_max_epoch', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--local_rank', type=int, default=0)

opt = parser.parse_args()

if not torch.cuda.is_available():
    opt.device = 'cpu'

