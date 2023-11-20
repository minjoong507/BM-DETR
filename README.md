# Overcoming Weak Visual-Textual Alignment for Video Moment Retrieval
![model](./res/model.jpg)
PyTorch Implementation of paper:

**[Overcoming Weak Visual-Textual Alignment for Video Moment Retrieval](https://arxiv.org/abs/2306.02728)**

[Minjoon Jung](https://minjoong507.github.io/), Youwon Jang, Seongho Choi, Joochan Kim, [Jin-Hwa Kim](http://wityworks.com/), [Byoung-Tak Zhang](https://bi.snu.ac.kr/~btzhang/)
***
## Updates

* [2023.11.20] Our preprint has been updated on arxiv.


## Requirements

To install requirements:

1. Prepare feature files

Please refer to [data_reparation]().

2. Install dependencies.

We recommend creating conda environment and installing all the dependencies as follows:
```
# create conda env
conda create --name bm_detr python=3.9
# activate env
conda actiavte bm_detr
# install pytorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
# install other python packages
pip install tqdm ipython easydict tensorboard tabulate scikit-learn pandas
```

### Training

Training can be launched by running the following command:
```
bash bm_detr/scripts/train.sh {dset_name} {v_feat_type}
```
where `dset_name` is the dataset name for training.

It can be one of `charades`, `charades-CD`, `tacos`, `activitynet`, and `hl` (QVHighlights).

Please check the `train.sh`.

For more model options, please check our config file bm_detr/config.py

The checkpoints and other experiment log files will be written into `results_{dset_name}`.

### Inference
You can use the following command for inference to check the performance of trained model:
```
bash bm_detr/scripts/inference.sh CHECKPOINT_PATH SPLIT_NAME  
``` 

where `CHECKPOINT_PATH` is the path to the saved checkpoint, `SPLIT_NAME` is the split name for inference.

Please check the `inference.sh` for setting right evaluation path and split name.

## License

We used resources from [MDETR](https://github.com/jayleicn/moment_detr) and [DAB-DETR](https://github.com/IDEA-Research/DAB-DETR).
We thank for the authors for making their projects open-sources.
All the code are under [MIT](https://opensource.org/licenses/MIT) license, see [LICENSE](./LICENSE).
