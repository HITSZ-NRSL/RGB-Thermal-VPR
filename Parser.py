import os
import argparse
import yaml

'''
TODO
--model
get different models
'''

class Parser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=None,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        ### Training settings
        self.parser.add_argument("--optim", type=str, default='adam', help="_", choices=['adam', 'sgd'])
        self.parser.add_argument("--margin", type=float, default=0.1, help="_")
        self.parser.add_argument("--lr", type=float, default=0.00001, help="_")
        self.parser.add_argument("--epochs_num", type=int, default=50, help="number of epochs to train for")
        self.parser.add_argument("--train_batch_size", type=int, default=4,
                            help="Batch size for train")
        self.parser.add_argument("--patience", type=int, default=5)
        
        self.parser.add_argument("--soft_positives_dist_threshold", type=int, default=10, help="_")
        self.parser.add_argument("--hard_positives_dist_threshold", type=int, default=10, help="_")

        self.parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random"])
        self.parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries")
        self.parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
        self.parser.add_argument("--negs_num_per_query", type=int, default=10, help="_")
        self.parser.add_argument("--neg_samples_num", type=int, default=1000, help="How many negatives to use to compute the hardest ones")

        ### Inference settings
        self.parser.add_argument("--infer_batch_size", type=int, default=64,
                            help="Batch size for inference (caching and testing)")
        self.parser.add_argument('--test_method', type=str, default="hard_resize",
                            choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                            help="This includes pre/post-processing methods and prediction refinement")

        ### Evaluation settings
        self.parser.add_argument("--majority_weight", type=float, default=0.01, 
                            help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
        self.parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                            help="Recalls to be computed, such as R@5.")
        self.parser.add_argument('--fuse', type=str, default=None, choices=[None, 'cat', 'add'])
        
        ### Model settings
        self.parser.add_argument('--resize', type=int, default=[224, 224], nargs=2, help="Resizing shape for images (HxW) to be fed into the network.")
        self.parser.add_argument("--features_dim", type=int, default=None, help="_")
        self.parser.add_argument("--foundation_model_path", type=str, default=None, help="_")

        ### Dataset parameters
        self.parser.add_argument("--img_time", type=str, default="allday", choices=["allday", "daytime", "nighttime"])
        self.parser.add_argument("--sequences", type=str, default=['KAIST', 'SNU', 'Valley'], nargs="+",
                                 help="List of sequences to load from the dataset. Default: ['KAIST', 'SNU', 'Valley']")
        self.parser.add_argument("--test_seq", type=str, default=None, help="path of the dataset")
        
        # Data augmentation parameters, # applyed to the training set
        self.parser.add_argument("--brightness", type=float, default=None, help="_")
        self.parser.add_argument("--contrast", type=float, default=None, help="_")
        self.parser.add_argument("--saturation", type=float, default=None, help="_")
        self.parser.add_argument("--hue", type=float, default=None, help="_")
        self.parser.add_argument("--rand_perspective", type=float, default=None, help="_")
        self.parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
        self.parser.add_argument("--random_rotation", type=float, default=None, help="_")
        self.parser.add_argument("--horizontal_flip", action='store_true', help="_")

        ### Record settings
        self.parser.add_argument("--config", type=str, default=None, help="Path to args config file")
        self.parser.add_argument("--save_dir", type=str, default="default", help="_")

        ### Common settings
        self.parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
        self.parser.add_argument("--num_workers", type=int, default=4, help="num_workers for all dataloaders")
        self.parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--resume", type=str, default=None, nargs='*',
                            help="Path to load checkpoint from, for resuming training or testing.")
        self.parser.add_argument("--save_all", type=bool, default=False)
    
    def parse_arguments(self):
        args = self.parser.parse_args()

        if args.config is not None:
            with open(args.config, 'r') as file:
                config = yaml.safe_load(file)
        
            self.parser.set_defaults(**config)
            args = self.parser.parse_args()
            args.save_dir = os.path.basename(os.path.dirname(args.save_dir))
        
        return args
