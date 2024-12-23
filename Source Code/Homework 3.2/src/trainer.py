import os
import torch
import torch.nn as nn
import logging
import numpy as np
from seqeval.metrics import f1_score

from src.dataloader import *
from src.utils import *

logger = logging.getLogger()
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

class BaseTrainer(object):
    def __init__(self, params, model, entity_list, label_list):
        # parameters
        self.params = params
        self.model = model
        self.label_list = label_list
        self.entity_list = entity_list
        
        # training
        self.lr = float(params.lr)
        self.early_stop = params.early_stop
        self.no_improvement_num = 0

        self.mu = 0.9
        self.weight_decay = 5e-4
    
        # build scheduler and optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.lr,
                                        momentum=self.mu,
                                        weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=eval(self.params.schedule),
                                                            gamma=self.params.gamma)
       
    def batch_forward(self, inputs):
        self.logits = self.model.forward(inputs)
          
    def batch_loss(self, labels):
        self.loss = 0
        # Cross-Entropy Loss
        ce_loss = nn.CrossEntropyLoss()(self.logits.view(-1, self.logits.shape[-1]), 
                                labels.flatten().long())
        self.loss = ce_loss
            
    def batch_backward(self):
        self.model.train()
        self.optimizer.zero_grad()        
        self.loss.backward()
        self.optimizer.step()
        
        return self.loss.item()

    def evaluate(self, dataloader, each_class=False, entity_order=[]):
        with torch.no_grad():
            self.model.eval()

            y_list = []
            x_list = []
            logits_list = []

            # Showing progress bar
            # pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            # for i, (x, y) in pbar:
            # Not showing progress bar
            for x, y in dataloader: 
                x, y = x.cuda(), y.cuda()
                self.batch_forward(x)
                _logits = self.logits.view(-1, self.logits.shape[-1]).detach().cpu()
                logits_list.append(_logits)
                x = x.view(x.size(0)*x.size(1)).detach().cpu()
                x_list.append(x) 
                y = y.view(y.size(0)*y.size(1)).detach().cpu()
                y_list.append(y)
            
            y_list = torch.cat(y_list)
            x_list = torch.cat(x_list)
            logits_list = torch.cat(logits_list)   
            pred_list = torch.argmax(logits_list, dim=-1)

            ### calcuate f1 score
            pred_line = []
            gold_line = []
            for pred_index, gold_index in zip(pred_list, y_list):
                gold_index = int(gold_index)
                if gold_index != pad_token_label_id:
                    pred_token = self.label_list[pred_index]
                    gold_token = self.label_list[gold_index]
                    # lines.append("w" + " " + pred_token + " " + gold_token)
                    pred_line.append(pred_token) 
                    gold_line.append(gold_token) 

            # Check whether the label set are the same,
            # ensure that the predict label set is the subset of the gold label set
            gold_label_set, pred_label_set = np.unique(gold_line), np.unique(pred_line)
            if set(gold_label_set)!=set(pred_label_set):
                O_label_set = []
                for e in pred_label_set:
                    if e not in gold_label_set:
                        O_label_set.append(e)
                if len(O_label_set)>0:
                    # map the predicted labels which are not seen in gold label set to 'O'
                    for i, pred in enumerate(pred_line):
                        if pred in O_label_set:
                            pred_line[i] = 'O'

            # compute overall f1 score
            f1 = f1_score([gold_line], [pred_line])*100
            if not each_class:
                return f1

            # compute f1 score for each class
            f1_list = f1_score([gold_line], [pred_line], average=None)
            f1_list = list(np.array(f1_list)*100)
            gold_entity_set = set()
            for l in gold_label_set:
                if 'B-' in l or 'I-' in l:
                    gold_entity_set.add(l[2:])
            gold_entity_list = sorted(list(gold_entity_set))
            f1_score_dict = dict()
            for e, s in zip(gold_entity_list,f1_list):
                f1_score_dict[e] = round(s,2)
            # using the default order for f1_score_dict
            if entity_order==[]:
                return f1, f1_score_dict
            # using the pre-defined order for f1_score_dict
            assert set(entity_order)==set(gold_entity_list),\
                "gold_entity_list and entity_order has different entity set!"
            ordered_f1_score_dict = dict()
            for e in entity_order:
                ordered_f1_score_dict[e] = f1_score_dict[e]
            return f1, ordered_f1_score_dict

    def save_model(self, save_model_name, path=''):
        """
        save the best model
        """
        if len(path)>0:
            saved_path = os.path.join(path, str(save_model_name))
        else:
            saved_path = os.path.join(self.params.dump_path, str(save_model_name))
        torch.save({
            "model": self.model
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)

    def load_model(self, load_model_name, path=''):
        """
        load the checkpoint
        """
        if len(path)>0:
            load_path = os.path.join(path, str(load_model_name))
        else:
            load_path = os.path.join(self.params.dump_path, str(load_model_name))
        ckpt = torch.load(load_path)

        self.model = ckpt['model']
        logger.info("Model has been load from %s" % load_path)