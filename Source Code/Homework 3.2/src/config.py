import yaml
import argparse

# ===========================================================================
# LOAD CONFIGURATIONS
def get_params():
    parser = argparse.ArgumentParser(description="NER")
    # experiment
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="train.log")
    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed")

    # model
    parser.add_argument("--model_name", type=str, default="bert-base-cased", help="model name (e.g., bert-base-cased, roberta-base or wide_resnet)")
    parser.add_argument("--is_load_ckpt_if_exists", default=False, action='store_true', help="Loading the ckpt if best finetuned ckpt exists")
    parser.add_argument("--ckpt", type=str, default=None, help="the pretrained lauguage model")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden layer dimension")

    # data
    parser.add_argument("--data_path", type=str, default="./datasets/NER_data/conll2003/", help="source domain")
    parser.add_argument("--entity_list", type=list, default=[], help="entity list")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size in target domain") 
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--mu", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--info_per_epochs", type=int, default=1, help="Print information every how many epochs")
    parser.add_argument("--save_per_epochs", type=int, default=0, help="Save checkpoints every how many epochs")
    parser.add_argument("--training_epochs", type=int, default=10, help="Number of training epochs in target domain")
    parser.add_argument("--schedule", type=str, default='(3, 6)', help="Multistep scheduler")
    parser.add_argument("--gamma", type=float, default=0.2, help="Factor of the learning rate decay")
    parser.add_argument("--early_stop", type=int, default=3, help="No improvement after several epoch, we stop training")
    parser.add_argument("--evaluate_interval", type=int, default=1, help="Evaluation interval")

    # config
    parser.add_argument("--cfg", default="./config/default.yaml", help="Hyper-parameters")

    params = parser.parse_args()

    with open(params.cfg) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)  # Specify SafeLoader here
        for k, v in config.items():
            # for parameters set in the args
            params.__setattr__(k, v)

    return params
