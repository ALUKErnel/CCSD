import argparse
import torch
import os
parser = argparse.ArgumentParser()

# path
parser.add_argument('--data_dir', default='./dataset/')


# for student both 
parser.add_argument('--cross_lingual_teacher_model_load_file', default='./save_cross-lingual_teacher')
parser.add_argument('--cross_target_teacher_model_load_file', default='./save_cross-target_teacher')
parser.add_argument('--student_model_save_file', default='./save_dual_distillation/politics_all')

parser.add_argument('--alpha', type=float, default=0.2)

# random seed 
parser.add_argument('--random_seed', type=int, default=1)

# data readin
parser.add_argument('--num_train_lines', type=int, default=0)  # set to 0 to use all training data
parser.add_argument('--max_seq_len', type=int, default=1000) # Dataset readin (settle)
parser.add_argument('--num_target', type=int, default=0) # set 0 to select all targets

# data setting
parser.add_argument('--sub_dataset', type=str, default='politics')  # sub_dataset: politics/ society
parser.add_argument('--target_setting', type=str, default='all')  # target setting: all/ paritial/ none


# X EmbeddingModule
parser.add_argument('--tokenized_max_len', type=int, default=120) # Dataset readin 
parser.add_argument('--emb_size', type=int, default=768)


# P StanceClassifier
parser.add_argument('--P_layers', type=int, default=2)
parser.add_argument('--P_bn', default=True)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.2)

# concat for stance classifier
parser.add_argument('--concat_stance', default=False)



# training 
parser.add_argument('--student_learning_rate', type=float, default=2e-5)
parser.add_argument('--student_batch_size', type=int, default=32)
parser.add_argument('--student_max_epoch', type=int, default=15)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--local_rank', type=int, default=0)

opt = parser.parse_args()

if not torch.cuda.is_available():
    opt.device = 'cpu'

