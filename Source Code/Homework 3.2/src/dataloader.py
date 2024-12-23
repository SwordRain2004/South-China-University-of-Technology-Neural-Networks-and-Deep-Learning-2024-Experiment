import torch
import torch.nn as nn
import os
import numpy as np
import logging
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from copy import deepcopy

from src.config import get_params

logger = logging.getLogger()
params = get_params()
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

# domain name to entity list
domain2entity={
    # run function get_entity_list() to determine the entity list
    'conll2003':['misc', 'location', 'person', 'organisation']
}

def get_entity_list(datapth):
    label_list=[]
    with open(datapth, "r", encoding="utf-8") as fr:
        for i, line in enumerate(fr):
            line = line.strip()
            splits = line.split("\t")
            if line=="":
                continue
            label_list.append(splits[1])
    # Contains only entity type w/o B-/I-
    label_list = sorted(list(set(label_list)))
    entity_list = []
    for l in label_list:
        if ('B-' in l) or ('I-' in l): 
            entity_list.append(l[2:])
    entity_list = list(set(entity_list))
    print("entity_list = %s"%str(entity_list))
    return entity_list

def get_default_label_list(entity_list):
    default_label_list = []
    default_label_list.append('O')
    for e in entity_list:
        default_label_list.append('B-'+str(e))
        default_label_list.append('I-'+str(e))
    return default_label_list

def read_ner(datapath, phase, label_list):
    if isinstance(datapath,list):
        if len(datapath)>1 and phase!="train":
            logger.warning("In %s phase, more than one domain data are combined!!!"%(phase))
        data_path_lst = [os.path.join(_path, phase+".txt") for _path in datapath]
    else:
        data_path_lst = [os.path.join(datapath, phase+".txt")]
    
    inputs, ys = [], []
    for _datapath in data_path_lst:
        _inputs, _ys = [], []
        with open(_datapath, "r", encoding="utf-8") as fr:
            token_list, y_list = [], []
            for i, line in enumerate(fr):
                line = line.strip() 
                if line == "":
                    if len(token_list) > 0:
                        assert len(token_list) == len(y_list)
                        _inputs.append([auto_tokenizer.cls_token_id] + token_list + [auto_tokenizer.sep_token_id])
                        _ys.append([pad_token_label_id] + y_list + [pad_token_label_id])

                    token_list, y_list = [], []
                    continue
                splits = line.split("\t")
                token = splits[0]
                label = splits[1]

                subs_ = auto_tokenizer.tokenize(token)
                if len(subs_) > 0:
                    y_list.extend([label_list.index(label)] + [pad_token_label_id] * (len(subs_) - 1))
                    token_list.extend(auto_tokenizer.convert_tokens_to_ids(subs_))
                else:
                    print("length of subwords for %s is zero; its label is %s" % (token, label))
        inputs.append(_inputs)
        ys.append(_ys)

    # combine data from different domains (only for training data)
    sample_cnt_lst = [len(_ys) for _ys in ys]
    max_cnt = max(sample_cnt_lst)
    inputs_all, ys_all = [], []
    for _inputs, _ys in zip(inputs, ys):
        ratio = int(max_cnt/len(_ys))
        inputs_all.extend(_inputs*ratio) 
        ys_all.extend(_ys*ratio)

    return inputs_all, ys_all

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, ys):
        self.X = inputs
        self.y = ys
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

def collate_fn(data):
    X, y = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(auto_tokenizer.pad_token_id)
    padded_y = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)
    for i, (seq, y_) in enumerate(zip(X, y)):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
        padded_y[i, :length] = torch.LongTensor(y_)

    return padded_seqs, padded_y

def get_label_distribution(y_lists,label_list, count=False):
    label_distribution = dict()
    count_tok_test = 0
    for y_list in y_lists:
        for y in y_list:
            if y != pad_token_label_id:
                label_name = label_list[y]
                if "B-" in label_name or "S-" in label_name:
                    count_tok_test += 1
                    label_name = label_name.split("-")[1]
                    if label_name not in label_distribution:
                        label_distribution[label_name] = 1
                    else:
                        label_distribution[label_name] += 1
    if count:
        return label_distribution
    else:
        for key in label_distribution:
            freq = label_distribution[key] / count_tok_test
            label_distribution[key] = round(freq, 2)
        return label_distribution

class NER_dataloader():
    def __init__(self, data_path, domain_name, batch_size, entity_list=[]):

        self.batch_size = batch_size

        # Get entity list
        if entity_list==[]:
            logger.info('Loading the default entity list from domain %s...'%domain_name)
            self.entity_list = domain2entity[domain_name]
        else:
            logger.info('Loading the pre-defined entity list...')
            self.entity_list = entity_list

        # Get label list
        self.label_list = get_default_label_list(self.entity_list)
        self.O_index = self.label_list.index('O')
        logger.info('label_list = %s'%str(self.label_list))

        # Load data
        logger.info("Load training set data")
        inputs_train, y_train = read_ner(data_path, 
                                        phase="train", 
                                        label_list=self.label_list)
        
        # Only evaluate on the target domain (default in the first item)
        if isinstance(data_path,list):
            target_data_path = data_path[0]
        logger.info("Load development set data")
        inputs_dev, y_dev = read_ner(target_data_path, 
                                    phase="dev", 
                                    label_list=self.label_list)
        logger.info("Load test set data")
        inputs_test, y_test = read_ner(target_data_path,
                                    phase="test",
                                    label_list=self.label_list)
        # Data statistic
        # logger.info("label distribution for train set")
        # logger.info(get_label_distribution(y_train,self.label_list))
        # logger.info("label distribution for dev set")
        # logger.info(get_label_distribution(y_dev,self.label_list))
        # logger.info("label distribution for test set")
        # logger.info(get_label_distribution(y_test,self.label_list))
        logger.info("train size: %d; dev size %d; test size: %d;" % (len(inputs_train), len(inputs_dev), len(inputs_test)))

        self.inputs_train, self.y_train = inputs_train, y_train
        self.inputs_dev, self.y_dev = inputs_dev, y_dev
        self.inputs_test, self.y_test = inputs_test, y_test        

    def get_dataloader(self):

        dataset_train = Dataset(self.inputs_train, self.y_train)
        dataset_dev = Dataset(self.inputs_dev, self.y_dev)
        dataset_test = Dataset(self.inputs_test, self.y_test)
        
        dataloader_train = DataLoader(dataset=dataset_train, 
                                        batch_size=self.batch_size, 
                                        shuffle=False, 
                                        collate_fn=collate_fn)
        dataloader_dev = DataLoader(dataset=dataset_dev, 
                                        batch_size=self.batch_size, 
                                        shuffle=False, 
                                        collate_fn=collate_fn)
        dataloader_test = DataLoader(dataset=dataset_test, 
                                        batch_size=self.batch_size, 
                                        shuffle=False, 
                                        collate_fn=collate_fn)

        return dataloader_train, dataloader_dev, dataloader_test

if __name__ == "__main__":
    get_entity_list('datasets/NER_data/conll2003/train.txt')