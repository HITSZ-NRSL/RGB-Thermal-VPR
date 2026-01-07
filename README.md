# RGBT-VPR: RGB-Thermal Visual Place Recognition via Vision Foundation Model
This is the official repository for the IROS 2025 paper: [RGB-Thermal Visual Place Recognition via Vision Foundation Model]()

## Abstract
*Visual place recognition is a critical component of robust simultaneous localization and mapping systems. Conventional approaches primarily rely on RGB imagery, but their performance degrades significantly in extreme environments, such as those with poor illumination and airborne particulate interference (e.g., smoke or fog), which significantly degrade the performance of RGB-based methods. Furthermore, existing techniques often struggle with cross-scenario generalization.
To overcome these limitations, we propose an RGB-thermal multimodal fusion framework for place recognition, specifically designed to enhance robustness in extreme environmental conditions. Our framework incorporates a dynamic RGB-thermal fusion module, coupled with dual fine-tuned vision foundation models as the feature extraction backbone. Experimental results on public datasets and our self-collected dataset demonstrate that our method significantly outperforms state-of-the-art RGB-based approaches, achieving generalizable and robust retrieval capabilities across day and night scenarios.*

## Getting started
### Try our model
You can run the `quickstart.ipynb` to try using our model for visual place recognition.

You can download the checkpoint [HERE](https://github.com/HITSZ-NRSL/RGB-Thermal-VPR/releases/tag/v1.0.0)
### Prepare Data
Our network is trained and evaluated on the [SThReO Dataset](https://sites.google.com/view/rpmsthereo/). You should first download the dataset [HERE](https://sites.google.com/view/rpmsthereo/download).

After that, run the `Dataset/STheReO_train_split.ipynb` and `Dataset/STheReO_test_split.ipynb` to split the original dataset and get the `.mat` file.

### Train the model
Our model use pretrained [DINOv2](https://dinov2.metademolab.com/) as backbone, so download pretrained weights of ViT-B/14 size from [HERE](https://github.com/facebookresearch/dinov2?tab=readme-ov-file#pretrained-models) before training our model.

Then you can start training by following command:
```
python3 train.py --save_dir /path/to/your/save/directory features_dim 768 --sequences KAIST --foundation_model_path /path/to/pretrained/weights
```

### Evaluation
You can evaluate the performance by following command:
```
python3 eval.py --resume /path/to/checkpoint --save_dir /path/to/save/directory --sequences SNU --img_time allday --features_dim 768
```
* `sequences`: choose from {SNU, Valley}
* `img_time`: choose from {daytime, nighttime, allday}

# Video
[![](https://img.youtube.com/vi/DdcS2P67XFQ/hqdefault.jpg)](https://youtu.be/DdcS2P67XFQ)

# Citation
If you find this project useful in your research, please consider citing:
```bibtex
@inproceedings{ye2025rgb,
  title={RGB-Thermal Visual Place Recognition via Vision Foundation Model},
  author={Ye, Minghao and Liu, Xiao and Wang, Yu and Liu, Lu and Chen, Haoyao},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4954--4960},
  year={2025},
  organization={IEEE}
}
```
