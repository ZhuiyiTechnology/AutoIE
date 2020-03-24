
# NLPCC-2020 Shared Task on AutoIE

                                                                  
## Background

Entity extraction is the fundamental problem in language technology, and usually utilized as inputs for many downstream tasks, especially dialogue system, question answering etc. Most previous work focus on the scenario that labelled data is provided for interesting entities, however, the categories of entities are hierarchical and cannot be exhausted, the general solution cannot depend on the hypothesis that enough data with label is given. Therefore, how to build IE system for new entity type under low resource is becoming the common problem for both academic and industry.        

## Task

The task is to build IE system with Noise and Incomplete annotations. Given a list of entities of specific type and a unlabelled corpus containing these entity types, the task aims to build an IE system which may recognize and extract the interesting entities of given types. 

Note:  
1.	entity is a general concept of named entity in this task. Some words without a specific name are also very important for downstream applications, therefore, they are included in this  information extraction task  
2.	No human annotation and correction are allowed for train and test dataset. 
3.	Dev dataset with full label may be used in the training step in any way.

## Data

The corpus are from caption text of YouKu video. Three categories of information are considered in this task, which are TV, person and series. All data are split into 3 datasets for training, developing and testing.

Train dataset
1. Unlabelled corpus containing 10000 samples, the entities are labelled by string matching with the given entity lists.
2. Entity lists with specific category, which may cover around 30% of entities appearing in the unlabelled corpus 

Dev dataset
1. 1000 samples with full label
 
Test dataset
1. 2000 samples with full label



## Submission & Evaluation

For submission, please write the prediction result into a single file and email it to Xuefeng Yang (杨雪峰) email：ryan@wezhuiyi.com

The submission file  format should be the same as the format of given dev dataset. To be specific, each sample is separated by a blank line and each char in sample is labelled by BIE format. All labels are B-TV, I-TV, E-TV, B-PER, I-PER, E-PER, B-NUM, I-NUM, E-NUM, and O. 

For evaluation. all the system will be evaluated against 2000 test samples with full annotation. Ranking of submissions are based on the f1 score of these test samples.  

A eval.py script is provided to calculate the accuracy and valid prediction format. You may use the script like this " python3 eval.py your_prediction_file_path gold_standard_file_path ".

## Organizers: 

Xuefeng Yang (ZhuiYi Technology)
email: ryan@wezhuiyi.com

Benhong Wu (ZhuiYi Technology)
email: wubenhong@wezhuiyi.com

Zhanming Jie (Singapore University of Technology and Design) 
email: zhanming_jie@mymail.sutd.edu.sg
