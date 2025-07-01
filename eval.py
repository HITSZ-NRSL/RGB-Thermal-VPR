import torch
from Parser import Parser
import logging
import os
from datetime import datetime
import torchvision.models as models
import numpy as np

import commons
import utils
import inference
import datasets_dual
import network

'''Setup'''
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
parser = Parser()
args = parser.parse_arguments()

commons.setup_logging(args.save_dir)
commons.seed_everything(args.seed)

utils.save_to_yaml(args)
logging.debug(f"The outputs are being saved in {args.save_dir}")

args.recall_values = list(range(1, 26))

'''Model'''
model = network.RGBTVPR_Net(pretrained_foundation = True, foundation_model_path = args.foundation_model_path)
model = model.to(args.device)

resume_path= args.resume[0]
print(resume_path)
model = utils.resume_model(resume_path, model)

'''Dataset'''
DATASET_FOLDER = "./STHEREO_Mat"
test_ds = datasets_dual.BaseSTheReODual(args, DATASET_FOLDER, split='test')

recalls, recalls_str = inference.inference(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

