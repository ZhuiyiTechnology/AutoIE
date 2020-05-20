
# NLPCC-2020 Shared Task on AutoIE

## Leaderboard

| Rank | TeamName | Organization | F1-score |
| --- | :---         |     :---:      |          ---: |
| 1 | Hair Loss Knight | 美团点评NLP中心 | 84.02 |
| 2 | Sophie | 搜狗杭州研究院知识图谱组 | 83.73 |
| 3 | 一只小绵羊 | 北京语言大学 | 82.78 |
| 4 | Null | 中国科学院自动化研究所 | 82.50 |
| 4 | Hermers | 武汉汉王大数据 | 82.23 |
| 5 | BUTAUTOJ | 北京工业大学信息学部 | 80.91 |
| 6 | AI surfing | Nanjing University of Posts and Telecommunications | 80.28 |
| 7 | STAM | 中国科学院信息工程研究所 | 80.24 |
| 8 | Circle | 北京林业大学 | 79.66 |
| 9 | Yulong | 武汉大学 | 77.84 |
| 10 | 小牛队 | 东北大学自然语言处理实验室 | 75.24 |
| 11 | augmented_autoner | PATech | 75.09 |
| 12 | Auto-IE | 北京航空航天大学计算机系实体抽取组 | 74.84 |
| 13 | AutoIE_ISCAS | Institute of Software, Chinese Academy of Sciences | 74.59 |
| 14 | yunke_ws | 加拿大皇后大学 | 71.96 |
| 15 | FIGHTING | 大连民族大学 | 65.16 |
| 16 | BaselineSystem   | NLPCC | 63.98 |

## Leaderboard without Valid Data

| Rank | TeamName | Organization | F1-score |
| --- | :---         |     :---:      |          ---: |
| 1 | Hair Loss Knight | 美团点评NLP中心 | 77.32 |
| 2 | Hermers | 武汉汉王大数据 | 71.86 |
| 3 | augmented_autoner | PATech | 70.76 |
| 4 | Circle | 北京林业大学 | 66.27 |
| 5 | BaselineSystem   | NLPCC | 63.98 |
                                
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

For evaluation. all the system will be evaluated against 2000 test samples with full annotation. Ranking of submissions are based on the f1 score of these test samples.  The test dataset includes 2000 real test samples and 8000 mixed samples, the score is only based on the prediction of the real 2000 samples.

The test dataset will be provided in 2020/05/15, and each team has three oppotunities to submit their results in the week 05/15--05/20. The results are public available in this github page and ranked by the f1 score.

## Organizers: 

Xuefeng Yang (ZhuiYi Technology)
email: ryan@wezhuiyi.com

Benhong Wu (ZhuiYi Technology)
email: wubenhong@wezhuiyi.com

Zhanming Jie (Singapore University of Technology and Design) 
email: zhanming_jie@mymail.sutd.edu.sg
