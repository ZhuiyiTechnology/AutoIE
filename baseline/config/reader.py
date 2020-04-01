from tqdm import tqdm
from common import Sentence, Instance
from typing import List
import re
from transformers import BertTokenizer

bert_model_dir = "bert-base-chinese-pytorch"  # change this if needed
tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)


class Reader:
    def __init__(self, digit2zero: bool=True):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        self.digit2zero = digit2zero
        self.vocab = set()

    def read_txt(self, file: str, number: int = -1) -> List[Instance]:  # expected type -> return type
        count_0 = 0
        print("Reading file: " + file)
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    assert len(words) == len(labels)
                    inst = Instance(Sentence(words), labels)
                    inst.set_id(len(insts))
                    insts.append(inst)
                    words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue

                #for
                x = line.split()
                if len(x) == 1:
                    word, label = '&', x[0]    # '&'
                elif len(x) == 2:
                    word, label = x[0], x[1]
                else:
                    print(x)

                if self.digit2zero:
                    word = re.sub('\d', '0', word)  # replace all digits with 0. '\d' - unicode decimal digits [0-9]
                    count_0 += 1
                words.append(word)
                self.vocab.add(word)

                labels.append(label)
        print("numbers being replaced by zero:", count_0)
        print("number of sentences: {}".format(len(insts)))
        return insts


if __name__ == "__main__":
    reader = Reader(False)
    tmp_file = "data/" + 'ecommerce' + "/train.txt"
    trains = reader.read_txt(tmp_file, -1)



