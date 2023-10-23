import os
import numpy as np

import pickle
import torch
from torch.utils.data import Dataset
import time
import jsonlines
import jsonlines


import torch
from transformers import XLMConfig, XLMTokenizer, XLMModel


# de_train de_valid de_test
# fr_train fr_valid fr_test

class XStanceDataset(Dataset):
    '''
    train and valid
    '''
    def __init__(self,
                 lang,
                 split, 
                 settype,
                 file_path,
                 num_train_lines,
                 max_seq_len,
                 target_list,
                 topic_dict
                 ): 
        self._settype = settype # test set type
        self._lang = lang  
        self._max_seq_len = max_seq_len

        self.raw_X_question = []
        self.raw_X_comment = [] 
        self.question_id = []
        self.X = [] # (target, text)
        self.X_prompt = [] 
        self.X_prompt_cross = []
        self.Y = [] # stance label
        self.Y_prompt = []
        self.Y_prompt_cross = []
        self.Y_t = [] # target label
        self.Y_l = [] # lang label
        self.num_labels = 2 
        
        self.label_dict = {"FAVOR": 1, "AGAINST": 0} # From dataset

        self.id2label = {1: "favor", 0: "against"}  
        self.label2id = {"favor": 1, "against":0} 
        

        self.topic_dict = topic_dict
        self.lang_dict = {"de": 1, "fr": 0} 

        

        self.target_list = target_list

        
        
        with jsonlines.open(file_path, 'r') as inf:
            print("fine ")
            cnt = 0
            for i, answer in enumerate(inf):
                # only take the lang instances
                if split == "test":  
                    # 筛选条件
                    if answer["test_set"] != self._settype: 
                        continue
                  
                    lang = answer["language"] 
                    if lang != self._lang:
                        continue

                    topic = answer["topic"] 
                    if topic not in self.topic_dict.keys():  
                        continue

                    question_id = answer["question_id"]
                    if question_id not in self.target_list: 
                        continue

                    question = answer["question"]
                    self.raw_X_question.append(question)
                    self.question_id.append(question_id)

                    comment = answer["comment"]
                    self.raw_X_comment.append(comment)
                    
                    self.X.append((question, comment[:self._max_seq_len]))   
                    
                    label = answer.get("label", None)
                    
                    label_index = self.label_dict[label]

                    if lang == 'de':
                        prompt_template = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist [MASK]."
                        prompt_template_cross = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est [MASK]."
                        self.X_prompt.append(prompt_template)
                        self.X_prompt_cross.append(prompt_template_cross)
                        prompt_truth = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist " + self.id2label[label_index] + "."
                        prompt_truth_cross = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est " + self.id2label[label_index] + "."
                        self.Y_prompt.append(prompt_truth)
                        self.Y_prompt_cross.append(prompt_truth_cross)
                    else : # lang = 'fr'
                        prompt_template = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est [MASK]."
                        prompt_template_cross = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist [MASK]."
                        self.X_prompt.append(prompt_template)
                        self.X_prompt_cross.append(prompt_template_cross)
                        prompt_truth = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est " + self.id2label[label_index] + "."
                        prompt_truth_cross = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist " + self.id2label[label_index] + "."
                        self.Y_prompt.append(prompt_truth)
                        self.Y_prompt_cross.append(prompt_truth_cross)

                    lang_index = self.lang_dict[lang]

                    self.Y.append(label_index)
                    self.Y_l.append(lang_index)
                    
                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break  
                    

                else: 

                    lang = answer["language"] 
                    if lang != self._lang: 
                        continue
                    
                    topic = answer["topic"]
                    if topic not in self.topic_dict.keys():
                        continue

                    question_id = answer["question_id"]
                    if question_id not in self.target_list:
                        continue

                    question = answer["question"]
                    self.raw_X_question.append(question)
                    self.question_id.append(question_id)

                    comment = answer["comment"]
                    self.raw_X_comment.append(comment)
                    
                    self.X.append((question, comment[:self._max_seq_len]))   
                    
                    label = answer.get("label", None)
                    
                    label_index = self.label_dict[label]

                    if lang == 'de':
                        prompt_template = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist [MASK]."
                        prompt_template_cross = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est [MASK]."
                        self.X_prompt.append(prompt_template)
                        self.X_prompt_cross.append(prompt_template_cross)
                        prompt_truth = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist " + self.id2label[label_index] + "."
                        prompt_truth_cross = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est " + self.id2label[label_index] + "."
                        self.Y_prompt.append(prompt_truth)
                        self.Y_prompt_cross.append(prompt_truth_cross)
                    else : # lang = 'fr'
                        prompt_template = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est [MASK]."
                        prompt_template_cross = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist [MASK]."
                        self.X_prompt.append(prompt_template)
                        self.X_prompt_cross.append(prompt_template_cross)
                        prompt_truth = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est " + self.id2label[label_index] + "."
                        prompt_truth_cross = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist " + self.id2label[label_index] + "."
                        self.Y_prompt.append(prompt_truth)
                        self.Y_prompt_cross.append(prompt_truth_cross)

                    lang_index = self.lang_dict[lang]

                    self.Y.append(label_index)
                    self.Y_l.append(lang_index)
                    
                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break  
                    
              

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X_prompt[idx], self.X_prompt_cross[idx], self.Y_prompt[idx], self.Y_prompt_cross[idx], self.Y[idx])   
    
    def get_all_questions(self):
        
        question_set = set()
        for q_id in self.question_id:
            question_set.add(q_id)

        all_question = list(question_set)
        sorted_all_question = sorted(all_question)

            
        return sorted_all_question
    
    def to_Y_t(self, sorted_questions):   # all_targets_list
        sorted_questions_list = list(sorted_questions)
        sorted_questions_dict = {}
        for i, q in enumerate(sorted_questions_list):
            sorted_questions_dict[q] = i
            
        for q in self.question_id:
            label_t = sorted_questions_dict[q]
            self.Y_t.append(label_t)
            

def get_datasets_main(train_file_path,
                 valid_file_path,
                 test_file_path,
                 num_train_lines,
                 max_seq_len,
                 src_target_list,
                 tgt_target_list,
                 topic_dict
                 ):
    
    de_train_dataset = XStanceDataset('de', 'train', None, train_file_path, num_train_lines, max_seq_len, src_target_list, topic_dict)
    de_valid_dataset = XStanceDataset('de', 'valid', None, valid_file_path, num_train_lines, max_seq_len, src_target_list, topic_dict)

    de_all_targets_list = de_train_dataset.get_all_questions()

    de_train_dataset.to_Y_t(de_all_targets_list)
    de_valid_dataset.to_Y_t(de_all_targets_list)

    de_test_dataset = XStanceDataset('de', 'test', 'new_comments_defr', test_file_path, num_train_lines, max_seq_len, src_target_list, topic_dict)
    de_test_dataset.to_Y_t(de_all_targets_list)

    return de_train_dataset, de_valid_dataset, de_test_dataset




            
        

            
            

if __name__ == "__main__":
    data_dir = "./dataset/" # mention the work dir

    tokenizer = ""

    train_file_path = os.path.join(data_dir, "train.jsonl")
    valid_file_path = os.path.join(data_dir, "valid.jsonl")
    test_file_path = os.path.join(data_dir, "test.jsonl")

    max_seq_len = 500 
    num_train_lines = 0

    topic_dict = {"Foreign Policy": 4, "Immigration": 5}
    src_target_list = [15, 16, 17, 18, 19, 20, 35, 59, 60, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3391, 3427, 3428, 3429, 3430, 3431, 3468, 3469, 3470, 3471] 
    tgt_target_list = [15, 16, 17, 18, 19, 20, 35, 59, 60, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3391, 3427, 3428, 3429, 3430, 3431, 3468, 3469, 3470, 3471] 

    
    de_train, de_valid, de_test  = get_datasets_main(train_file_path, valid_file_path, test_file_path, num_train_lines, max_seq_len, src_target_list, tgt_target_list, topic_dict)


    print(de_train[1])

    print(len(de_train))
    print(len(de_valid))
    print(len(de_test))




                    

    




   
   