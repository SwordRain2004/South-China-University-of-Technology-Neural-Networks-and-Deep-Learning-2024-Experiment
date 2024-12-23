import torch
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt  # Add this for visualization
from transformers import AutoConfig, AutoModelWithLMHead
from src.utils import *
from src.dataloader import *
from src.trainer import *
from src.config import *


class BertTagger(nn.Module):
    def __init__(self, hidden_dim, output_dim, model_name):
        super(BertTagger, self).__init__()
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.bert_model = AutoModelWithLMHead.from_pretrained(model_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, output_dim)

    def forward(self, X):
        outputs = self.bert_model(X)
        hidden_states = outputs.hidden_states
        features = hidden_states[-1]
        logits = self.classifier(features)
        return logits


def main(params):
    # Set random seeds for reproducibility
    if params.seed:
        random.seed(params.seed)
        np.random.seed(params.seed)
        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        torch.backends.cudnn.deterministic = True

    # Initialize experiment logging
    logger = init_experiment(params, logger_filename=params.logger_filename)
    logger.info(params.__dict__)

    # Set domain name and data loading
    domain_name = os.path.basename(params.data_path[0]) if params.data_path[0] else os.path.basename(
        params.data_path[0][:-1])
    ner_dataloader = NER_dataloader(data_path=params.data_path, domain_name=domain_name, batch_size=params.batch_size,
                                    entity_list=params.entity_list)
    dataloader_train, dataloader_dev, dataloader_test = ner_dataloader.get_dataloader()
    label_list = ner_dataloader.label_list
    entity_list = ner_dataloader.entity_list

    # Initialize model and trainer
    if params.model_name in ['bert-base-cased', 'roberta-base']:
        model = BertTagger(hidden_dim=params.hidden_dim, output_dim=len(label_list), model_name=params.model_name)
    else:
        raise Exception('Invalid model name %s' % params.model_name)
    model.cuda()
    trainer = BaseTrainer(params, model, entity_list, label_list)

    # Initialize tracking variables for metrics
    epoch_loss = []  # Store loss for each epoch
    epoch_f1 = []  # Store F1 score for each epoch

    # Training loop
    logger.info("Training ...")
    no_improvement_num = 0
    best_f1 = 0
    step = 0

    for e in range(1, params.training_epochs + 1):
        logger.info("============== epoch %d ==============" % e)
        loss_list = []
        total_cnt = 0
        correct_cnt = 0

        # Train over batches
        pbar = tqdm(dataloader_train, total=len(dataloader_train))
        for X, y in pbar:
            step += 1
            X, y = X.cuda(), y.cuda()
            trainer.batch_forward(X)

            # Accuracy and loss computation
            correct_cnt += int(torch.sum(torch.eq(torch.max(trainer.logits, dim=2)[1], y).float()).item())
            total_cnt += trainer.logits.size(0) * trainer.logits.size(1)
            trainer.batch_loss(y)
            loss = trainer.batch_backward()
            loss_list.append(loss)

            # Update progress bar
            mean_loss = np.mean(loss_list)
            pbar.set_description("Epoch %d, Step %d: Loss=%.4f, Training_acc=%.2f%%" % (
                e, step, mean_loss, correct_cnt / total_cnt * 100
            ))

        # Store loss for current epoch
        epoch_loss.append(mean_loss)

        # Evaluation on dev set and store F1 score
        if e % params.evaluate_interval == 0:
            f1_dev, _ = trainer.evaluate(dataloader_dev, each_class=True)
            epoch_f1.append(f1_dev)  # Save F1 for current epoch

            logger.info("Epoch %d, Dev F1: %.4f" % (e, f1_dev))
            if f1_dev > best_f1:
                best_f1 = f1_dev
                no_improvement_num = 0
                trainer.save_model("best_finetune_domain_%s.pth" % domain_name, path=params.dump_path)
            else:
                no_improvement_num += 1
                if no_improvement_num >= params.early_stop:
                    logger.info("Stopping early due to no improvement")
                    break

    # Visualize loss and F1 score
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(epoch_loss) + 1), epoch_loss, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(epoch_f1) + 1), epoch_f1, label="F1 Score", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score on Dev Set per Epoch")
    plt.legend()

    plt.show()

    # Final evaluation on test set
    logger.info("Testing on test set ...")
    trainer.load_model("best_finetune_domain_%s.pth" % domain_name, path=params.dump_path)
    f1_test, f1_score_dict = trainer.evaluate(dataloader_test, each_class=True)
    logger.info("Test Set F1: %.4f" % f1_test)
    f1_score_dict = sorted(f1_score_dict.items(), key=lambda x: x[0])
    logger.info("F1_list: %s" % (f1_score_dict))


if __name__ == "__main__":
    params = get_params()
    main(params)
