python code_dual_distillation/main_distillation_both.py --student_model_save_file ./save_dual_distillation/politics_all --sub_dataset politics --target_setting all
python code_dual_distillation/main_distillation_both.py --student_model_save_file ./save_dual_distillation/politics_partial --sub_dataset politics --target_setting partial
python code_dual_distillation/main_distillation_both.py --student_model_save_file ./save_dual_distillation/politics_none --sub_dataset politics --target_setting none


python code_dual_distillation/main_distillation_both.py --student_model_save_file ./save_dual_distillation/society_all --sub_dataset society --target_setting all
python code_dual_distillation/main_distillation_both.py --student_model_save_file ./save_dual_distillation/society_partial --sub_dataset society --target_setting partial
python code_dual_distillation/main_distillation_both.py --student_model_save_file ./save_dual_distillation/society_none --sub_dataset society --target_setting none