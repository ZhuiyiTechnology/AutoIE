This repository is the baseline for the AutoIE evaluation. The solution is based on the work “Better Modeling of Incomplete Annotations for Named Entity Recognition”。 Different from the implementation in the origin work, we add pretrained language model. Please cite it if you use it. 

### Requirements
* PyTorch >= 1.1
* Python 3
* tqdm(pip install tqdm)
* transformers(pip install transformers)
* numpy >= 1.16
* termcolor >= 1.1(pip install termcolor)


### Data:
*   Please copy all the files and folder under data path in the AutoIE repository into the data folder under this repository. 
*   Data include the "train_dict", "valid", "train".
*   Train_dict: including 3 types entity dict: TV, PER and NUM.

### Train model:
*  1.Download the bert pretrained model chinese_wwm_ext released by 哈工大讯飞联合实验室. You can down load it in "https://drive.google.com/open?id=1iNeYFhCBJWeUsIlnW_2K6SMwXkM4gLb_" or "https://pan.iflytek.com/link/B9ACE1C9F228A0F42242672EF6CE1721"(password:XHu4).
*  2.Put the model file {"bert_config.json", "pytorch_model.bin", "vocab.txt"} to the folder: "bert-base-chinese-pytorch".
*  3.Run "run.sh" to train model.

### Performance: 
*   f1 = 65.5 on valid set.

### Reference
@inproceedings{jie-etal-2019-better,
    title = "Better Modeling of Incomplete Annotations for Named Entity Recognition",
    author = "Jie, Zhanming  and
      Xie, Pengjun  and
      Lu, Wei  and
      Ding, Ruixue  and
      Li, Linlin",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    doi = "10.18653/v1/N19-1079",
    pages = "729--734",
}