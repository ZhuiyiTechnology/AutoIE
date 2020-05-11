
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

## Baseline
The evaluation provides a baseline system for participants. The solution is based on the paper "Better Modeling of Incomplete Annotations for Named Entity Recognition", please check the readme file in the baseline folder for more detail

## Submission & Evaluation
For submission, please write the prediction result into a single file and email it to Xuefeng Yang (杨雪峰) email：ryan@wezhuiyi.com

The submission file  format should be the same as given YourTeamName.json file under Submission folder. To be specific, each line is a json string containing the prediction result of one sample. 

For evaluation. all the system will be evaluated against 2000 test samples with full annotation. Ranking of submissions are based on the f1 score of these test samples.  

The test dataset will be provided in 2020/05/13, and each team has three oppotunities to submit their results in the week 05/13--05/20. The results are public available in this github page and ranked by the f1 score.

## Organizers: 

Xuefeng Yang (ZhuiYi Technology)
email: ryan@wezhuiyi.com

Benhong Wu (ZhuiYi Technology)
email: wubenhong@wezhuiyi.com

Zhanming Jie (Singapore University of Technology and Design) 
email: zhanming_jie@mymail.sutd.edu.sg
