
# Overview 
This project is the source code of Cross-lingual Cross-target Stance Detection Method (CCSD), which is proposed in the paper "Cross-Lingual Cross-Target Stance Detection with Dual Knowledge Distillation Framework" (accepted by EMNLP 2023).

## Abstract 
Stance detection aims to identify the user’s attitude toward specific targets from *text*, which is an important research area in text mining and benefits a variety of application domains. Existing studies on stance detection were conducted mainly in English. Due to the low-resource problem in most non-English languages, cross-lingual stance detection was proposed to transfer knowledge from high-resource (source) language to low-resource (target) language. However, previous research has ignored the practical issue of no labeled training data available in target language. Moreover, target inconsistency in cross-lingual stance detection brings about the additional issue of unseen targets in target language, which in essence requires the transfer of both language and target-oriented knowledge from source to target language. To tackle these challenging issues, in this paper, we propose the new task of cross-lingual cross-target stance detection and develop the first computational work with dual knowledge distillation. Our proposed framework designs a cross-lingual teacher and a cross-target teacher using the source language data and a dual distillation process that transfers the two types of knowledge to target language. To bridge the target discrepancy between languages, cross-target teacher mines target category information and generalizes it to the unseen targets in target language via category-oriented learning. Experimental results on multilingual stance datasets demonstrate the effectiveness of our method compared to the competitive baselines.

## Organization of Project

The project contains three main folders of code:
- code_cross-lingual_teacher: pre-training code for cross-lingual teacher model 
- code_cross-target_teacher: pre-training code for cross-target teacher model 
- code_dual_distillation: dual knowledge distillation process for student model 

Besides,
- dataset: dataset of X-stance
- requirements.txt: setup



# Setup 

To run the project, the dependencies in ```requirements.txt``` need to be installed. Please run the following command: 
```
pip install -r requirements.txt 
```

# Dataset and Experimental Settings 

We construct **two datasets** from X-stance [(https://zenodo.org/records/3831317)](https://zenodo.org/records/3831317):
- Politics (P): is comprised of all the data in domains “Foreign Policy” and “Immigration”, with 31 different targets in total (removing one target which exists only in German data). There are 7064 samples in German and 2582 samples in French; 

- Society (S): consists of all the data in domains “Society” and “Security”, with 32 different targets in total (removing one target which exists only in German data). It contains 7362 samples in German and 2467 samples in French.

We establish **three different target settings** on the two datasets:
1. All: All the targets in source language and target language are the same; 

2. Partial: Part of the targets in source language and target language are the same. We randomly select 50% overlap of all the targets between source and target languages; 

3. None: None of the targets in source language and target language are the same. We randomly select 50% of the targets in source language and remaining 50% for target language. Thus, we get 6 different experimental settings (i.e., 2 datasets with 3 target settings) to verify the effectiveness of our method.

We set up two Parsers to control the dataset and the target setting, They are ```--sub_dataset``` (which can be politics/society) and ```--target_setting``` (which can be all/none/partial).

# Train 

## Cross-Lingual Teacher Training
Please run the following command to train cross-lingual teacher: 

```shell
python code_cross-lingual_teacher/prompt_cross_kl.py --model_save_file ./save_cross-lingual_teacher/mbert_politics_none --sub_dataset politics --target_setting partial
```

The results and the trained cross-lingual teacher model will be saved in "./save_cross-lingual_teacher/mbert_[```SUB_DATASET```]_[```TARGET_SETTING```]".


## Cross-target Teacher Training 

To bridge the language gap, the encoder of cross-target teacher is initialized with the cross-lingual teacher. Therefore, please run the following command to train cross-target teacher based on the path of cross-lingual teacher:

```shell
python code_cross-target_teacher/main_teacher_initialized.py --cross_target_teacher_model_save_file ./save_cross-target_teacher/cluster_initialized_politics_all --sub_dataset politics --target_setting all
```

The results and the trained cross-teacher teahcer model will be saved in "./save_cross-target_teacher/cluster_initialized_[```SUB_DATASET```]_[```TARGET_SETTING```]".


## Dual Knowledge Distillation 


The language and target knowledge learned from the two teachers are transferred to the student with dual distillation process. 

```shell
python code_dual_distillation/main_distillation_both.py --student_model_save_file ./save_dual_distillation/politics_all --sub_dataset politics --target_setting all
```

The results will be saved in "./save_dual_distillation/[```SUB_DATASET```]_[```TARGET_SETTING```]". We select the epoch with the best performance on the validation set, and the test performance in that epoch is the final result of the cross-lingual cross-target stance detection task. 



# Citation

If you find this repo useful, please cite our paper: 

Ruike Zhang, Hanxuan Yang, and Wenji Mao. 2023. Cross-Lingual Cross-Target Stance Detection with Dual Knowledge Distillation Framework. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 10804–10819, Singapore. Association for Computational Linguistics.


# Acknowledgment 

- Dataset X-stance [(https://zenodo.org/records/3831317)](https://zenodo.org/records/3831317)
- Huggingface (Transformers-BERT) [(https://huggingface.co/docs/transformers/model_doc/bert)](https://huggingface.co/docs/transformers/model_doc/bert)
- ADAN (basic structure) [(https://github.com/ccsasuke/adan)](https://github.com/ccsasuke/adan)
- JointCL (k-means) [(https://github.com/HITSZ-HLT/JointCL)](https://github.com/HITSZ-HLT/JointCL)

All the above references have been cited in the paper. We thank them for their contributions to open source.






