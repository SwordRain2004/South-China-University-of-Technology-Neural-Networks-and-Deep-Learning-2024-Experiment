import torch
import numpy as np
from transformers import AutoTokenizer

from src.utils import *
from src.dataloader import *
from src.trainer import *
from main import *
from src.config import *

auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index


def main(params):
    # ===========================================================================
    # BERT-based NER Tagger
    if params.model_name in ['bert-base-cased', 'roberta-base']:
        model = BertTagger(hidden_dim=768, output_dim=9, model_name=params.model_name)
    else:
        raise Exception('model name %s is invalid' % params.model_name)
    model.cuda()
    trainer = BaseTrainer(params, model, [], [])
    trainer.load_model("best_finetune_domain_conll2003.pth", path=r'experiments\default\1')

    # Label List 
    label_list = np.array(get_default_label_list(domain2entity['conll2003']))

    # Convert words into tensor
    sentence = "South China University of Technology is located in Guangzhou"
    word_list = sentence.split(" ")
    token_list = []
    mask_list = []  # Mask the predictions of the subwords
    for word in word_list:
        subs_ = auto_tokenizer.tokenize(word)
        if len(subs_) > 0:
            token_list.extend(auto_tokenizer.convert_tokens_to_ids(subs_))
            mask_list.extend([True] + (len(subs_) - 1) * [False])
        else:
            print("length of subwords %s is zero; its label is %s" % (word))

    X = torch.tensor([[auto_tokenizer.cls_token_id] + token_list + [auto_tokenizer.sep_token_id]])
    mask_list = torch.tensor([[False] + mask_list + [False]])

    # do prediction
    trainer.batch_forward(X.cuda())
    predict_idx = torch.max(trainer.logits, dim=2)[1]
    predict_idx = predict_idx.cpu().detach()
    predict_idx = torch.masked_select(predict_idx, mask_list)
    predict_labels = label_list[predict_idx]
    print(sentence)
    print(predict_labels)


if __name__ == "__main__":
    params = get_params()
    main(params)
