from typing import List
from common import Instance
import torch


START = "<START>"
STOP = "<STOP>"
PAD = "<PAD>"


class Config:
    def __init__(self, args) -> None:
        """
        Construct the arguments and some hyperparameters
        :param args:
        """

        # Predefined label string.
        self.PAD = PAD
        self.B = "B-"
        self.I = "I-"
        self.E = "E-"
        self.O = "O"
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.UNK = "<UNK>"
        self.unk_id = -1

        self.seed = args.seed
        self.digit2zero = args.digit2zero

        # Data specification
        self.dataset = args.dataset
        self.train_file = None
        self.dev_file = None
        self.label2idx = {}
        self.idx2labels = []
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.num_folds = 2    # this is the k mention in paper

        # Training hyperparameter
        self.model_folder = args.model_folder
        self.optimizer = args.optimizer.lower()
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        self.use_dev = True
        self.batch_size = args.batch_size
        self.clip = 5
        self.lr_decay = args.lr_decay
        self.device = torch.device(args.device)
        self.num_outer_iterations = args.num_outer_iterations

        # transformer hyperparameters
        self.bert_model_dir = args.bert_model_dir
        self.max_len = args.max_len
        self.full_finetuning = args.full_finetuning
        self.clip_grad = args.clip_grad

    def build_label_idx(self, insts: List[Instance]) -> None:
        """
        Build the mapping from label to index and index to labels.
        :param insts: list of instances.
        :return:
        """
        self.label2idx[self.PAD] = len(self.label2idx) # in this case, idx for PAD is always 0, and [START], [STOP] are in the end
        self.idx2labels.append(self.PAD)
        for inst in insts:
            for label in inst.output:
                if label not in self.label2idx:
                    self.idx2labels.append(label)
                    self.label2idx[label] = len(self.label2idx)

        for inst in insts:
            inst.output_ids = [] if inst.output else None
            if inst.output:
                for label in inst.output:
                    inst.output_ids.append(self.label2idx[label])   # assign label ids

        self.label2idx[self.START_TAG] = len(self.label2idx)
        self.idx2labels.append(self.START_TAG)
        self.label2idx[self.STOP_TAG] = len(self.label2idx)
        self.idx2labels.append(self.STOP_TAG)
        self.label_size = len(self.label2idx)
        self.start_label_id = self.label2idx[self.START_TAG]
        self.stop_label_id = self.label2idx[self.STOP_TAG]
        print("#labels: {}".format(self.label_size))
        print("label 2idx: {}".format(self.label2idx))

