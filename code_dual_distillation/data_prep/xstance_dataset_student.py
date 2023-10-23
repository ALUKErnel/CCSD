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
    def __init__(self,
                 lang,
                 split, 
                 settype,
                 file_path,
                 tokenizer,
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
        self.Y = [] # stance label
        self.Y_prompt = []
        self.Y_t = [] # target label
        self.Y_l = [] # lang label
        self.num_labels = 2 
        
        self.label_dict = {"FAVOR": 1, "AGAINST": 0}
        self.topic_dict = topic_dict
        self.lang_dict = {"de": 1, "fr": 0} 

        self.target_list = target_list

        self.id2label = {1: "favor", 0: "against"}  
        self.label2id = {"favor": 1, "against":0} 

        
        
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
                    lang_index = self.lang_dict[lang]
                    topic_index = self.topic_dict[topic]
                    
                    self.Y.append(label_index)
                    self.Y_l.append(lang_index)

                    if lang == 'de':
                        prompt_template = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist [MASK]."
                        self.X_prompt.append(prompt_template)
                        prompt_truth = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist " + self.id2label[label_index] + "."
                        self.Y_prompt.append(prompt_truth)
                    else : # lang = 'fr'
                        prompt_template = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est [MASK]."
                        self.X_prompt.append(prompt_template)
                        prompt_truth = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est " + self.id2label[label_index] + "."
                        self.Y_prompt.append(prompt_truth)


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
                    lang_index = self.lang_dict[lang]
                    
                    self.Y.append(label_index)
                    self.Y_l.append(lang_index)


                    if lang == 'de':
                        prompt_template = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist [MASK]."
                        self.X_prompt.append(prompt_template)
                        prompt_truth = "Die Haltung von '" + comment[:self._max_seq_len] + "' gegenüber '" + question[:self._max_seq_len] +  "' ist " + self.id2label[label_index] + "."
                        self.Y_prompt.append(prompt_truth)
                    else : # lang = 'fr'
                        prompt_template = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est [MASK]."
                        self.X_prompt.append(prompt_template)
                        prompt_truth = "La position de '" + comment[:self._max_seq_len] +  "' envers '" + question[:self._max_seq_len] + "' est " + self.id2label[label_index] + "."
                        self.Y_prompt.append(prompt_truth)
                    
                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break  
                    
              

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X_prompt[idx], self.X[idx], self.Y_prompt[idx], 
                self.Y_t[idx], self.Y[idx],self.Y_l[idx])   
    
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
            



class XStanceDataset_target(Dataset):
    def __init__(self,
                 target,
                 lang,
                 split, 
                 settype,
                 file_path,
                 tokenizer,
                 num_train_lines,
                 max_seq_len, 
                 topic_dict
                 ): 
        self._settype = settype # test set type 
        self._max_seq_len = max_seq_len
        self._target = target 
        self._lang = lang 

        self.raw_X_question = []
        self.raw_X_comment = [] 
        self.X = [] 
        self.Y = []
        self.Y_d = []
        self.Y_l = []
        self.num_labels = 2  
        self.label_dict = {"FAVOR": 1, "AGAINST": 0}
        self.topic_dict = topic_dict
        self.lang_dict = {"de": 1, "fr": 0} 
        

        
        with jsonlines.open(file_path, 'r') as inf:
            print("fine target ")
            cnt = 0
            for i, answer in enumerate(inf):
                # only take the lang instances
                if split == "test": 
                    if answer["test_set"] != self._settype: 
                        continue

                    lang = answer["language"] 
                    if lang != self._lang:
                        continue
                    
                    topic = answer["topic"] 
                    if topic not in self.topic_dict.keys():  
                        continue
                    
                    if answer["question_id"] != self._target:
                        continue
                    
                    question = answer["question"]
                    self.raw_X_question.append(question)
                   
                    comment = answer["comment"]
                    self.raw_X_comment.append(comment)
                   
                    self.X.append((question, comment[:self._max_seq_len]))   
                    
                    label = answer.get("label", None)
                    
                    label_index = self.label_dict[label]
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
                    
                    if answer["question_id"] != self._target:
                        continue
                    
                    question = answer["question"]
                    self.raw_X_question.append(question)
                    
                    comment = answer["comment"]
                    self.raw_X_comment.append(comment)

                    self.X.append((question, comment[:self._max_seq_len]))
                    
                    label = answer.get("label", None)
                    
                    label_index = self.label_dict[label]
                    lang_index = self.lang_dict[lang]
                    
                    self.Y.append(label_index)
                    self.Y_l.append(lang_index)
                    
                    cnt += 1
                    if num_train_lines > 0 and cnt >= num_train_lines:
                        break


    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])    
    
    def get_target(self):
        return self._target


def get_datasets_main(train_file_path,
                 valid_file_path,
                 test_file_path,
                 tokenizer,
                 num_train_lines,
                 max_seq_len,
                 src_target_list,
                 tgt_target_list,
                 topic_dict
                 ):
    
    de_train_dataset = XStanceDataset('de', 'train', None, train_file_path, tokenizer, num_train_lines, max_seq_len, src_target_list, topic_dict)
    de_valid_dataset = XStanceDataset('de', 'valid', None, valid_file_path, tokenizer, num_train_lines, max_seq_len, src_target_list, topic_dict)
    de_test_dataset = XStanceDataset('de', 'test','new_comments_defr', test_file_path, tokenizer, num_train_lines, max_seq_len, src_target_list, topic_dict)

    de_all_targets_list = de_train_dataset.get_all_questions()

    de_train_dataset.to_Y_t(de_all_targets_list)
    de_valid_dataset.to_Y_t(de_all_targets_list)
    de_test_dataset.to_Y_t(de_all_targets_list)

    fr_train_dataset = XStanceDataset('fr', 'train', None, train_file_path, tokenizer, num_train_lines, max_seq_len, tgt_target_list, topic_dict)
    fr_valid_dataset = XStanceDataset('fr', 'valid', None, valid_file_path, tokenizer, num_train_lines, max_seq_len, tgt_target_list, topic_dict)
    fr_test_dataset = XStanceDataset('fr', 'test','new_comments_defr', test_file_path, tokenizer, num_train_lines, max_seq_len, tgt_target_list, topic_dict)

    fr_all_targets_list = fr_train_dataset.get_all_questions()

    fr_train_dataset.to_Y_t(fr_all_targets_list)
    fr_valid_dataset.to_Y_t(fr_all_targets_list)
    fr_test_dataset.to_Y_t(fr_all_targets_list)

    
    return de_all_targets_list, fr_all_targets_list, de_train_dataset, de_valid_dataset, de_test_dataset, fr_train_dataset, fr_valid_dataset, fr_test_dataset




def get_datasets_target(train_file_path,
                        valid_file_path,
                        test_file_path,
                        tokenizer,
                        num_train_lines,
                        max_seq_len,
                        de_all_targets_list, 
                        fr_all_targets_list,
                        topic_dict
                        ):
    
    dataset_list = [] 

    for t in de_all_targets_list:
        train_dataset_each_target = XStanceDataset_target(t, "de", 'train', None, train_file_path, tokenizer, num_train_lines, max_seq_len, topic_dict)
        dataset_list.append(train_dataset_each_target)
    
    return dataset_list

            
        

            
            

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

    
    de_targets, fr_targets, de_train, de_valid, de_test, fr_train, fr_valid, fr_test = get_datasets_main(train_file_path, valid_file_path, test_file_path, tokenizer, num_train_lines, max_seq_len, src_target_list, tgt_target_list, topic_dict)

    print(de_targets)
    print(fr_targets)

    print(de_train[1])

    print(len(de_train))
    print(len(de_valid))
    print(len(de_test))

    print(len(fr_train))
    print(len(fr_valid))
    print(len(fr_test))


    print("_"*25)
    dataset_per_target_list = get_datasets_target(train_file_path, valid_file_path, test_file_path, tokenizer, num_train_lines, max_seq_len, de_targets, fr_targets, topic_dict)

    sum = 0
    for dt in dataset_per_target_list:
        t = dt.get_target()
        t_size = len(dt)
        sum += t_size

        print(t, t_size)

    print(sum)

                    

    




   
   