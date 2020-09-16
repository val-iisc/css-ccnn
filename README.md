# Completely Self-Supervised Crowd Counting via Distribution Matching

This repository provides a [PyTorch](http://pytorch.org/) implementation and pretrained models for CSS-CCNN, as described in the paper [Completely Self-Supervised Crowd Counting
via Distribution Matching](http://arxiv.org/abs/2009.06420).

![CSS-CCNN Architecture](/resources/cssccnn_architecture.png)
Existing self-supervised approaches can learn good representations, but require some labeled data to map these features to the end task of crowd density estimation. We mitigate this issue with the proposed paradigm of complete self-supervision, which does not need even a single labeled image. Our method dwells on the idea that natural crowds follow a power law distribution, which could be leveraged to yield error signals for backpropagation. A density regressor is first pretrained with self-supervision and then the distribution of predictions is matched to the prior by optimizing Sinkhorn distance between the two.

# Dataset Requirements
Download Shanghaitech dataset from [here](https://github.com/desenzhou/ShanghaiTechDataset).
Download UCF-QNRF dataset from [here](http://crcv.ucf.edu/data/ucf-qnrf/).

Place the dataset in `../dataset/` folder. (`dataset` and `css-cnn` folders should have the same parent directory). So the directory structure should look like the following:
```
-- css-cnn
   -- network.py
   -- stage1_main.py
   -- ....
-- dataset
   --ST_partA
     -- test_data
      -- ground-truth
      -- images
     -- train_data
      -- ground-truth
      -- images
  --UCF-QNRF
    --Train
      -- ...
    --Test
      -- ...
```

# Dependencies and Installation
We strongly recommend to run the codes in Nvidia-Docker. Install both `docker` and `nvidia-docker` (please find instructions from their respective installation pages).
After the docker installations, pull pytorch docker image with the following command:
`docker pull nvcr.io/nvidia/pytorch:18.04-py3`
and run the image using the command:
`nvidia-docker run --rm -ti --ipc=host nvcr.io/nvidia/pytorch:18.04-py3`

Further software requirements are listed in `requirements.txt`. 

To install them type, `pip install -r requirements.txt`

The code has been run and tested on `Python 3.6.4`, `Cuda 9.0, V9.0.176` and `PyTorch 0.4.1`. 

# Usage

## Pretrained Models

The pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1KhAzNrOvyN5oiFUePfnzibjY3w_6DML6?usp=sharing). The directory structure is as follows:

```
-- parta
   -- models_stage_1
     -- unsup_vgg_best_model_meta.pkl
     -- stage1_epoch_parta.pth
   -- models_stage_2
     -- stage2_epoch_parta_cssccnn.pth
     -- stage2_epoch_parta_cssccnn.pth
-- ucfqnrf
   -- models_stage_1
     -- ...
   -- models_stage_2
     -- ...
```

* For testing the Stage-2 pretrained models, save the pretrained weights files from `{dataset}/models_stage_2` in `models_stage_2/train2/snapshots/` and follow the steps outlined in Testing section.

* For training only Stage-2 using Stage-1 pretrained model, save the pretrained weights files from `{dataset}/models_stage_1` in `models_stage_1/train2/snapshots/` and follow steps for Stage-2 CSS-CCNN or CSS-CCNN++ training.

## Testing

After either finishing the training or downloading pretrained models, the model can be tested using the below script.
The model must be present in `models_stage_2/train2/snapshots`.

* `python test_model.py --best_model_name parta_cssccnnv2 --dataset parta`
```
--dataset = parta / ucfqnrf + cssccnn / cssccnnv2
--best_model_name = Name of the model checkpoint to be tested
```

## Training
After downloading the datasets and installing all dependencies, proceed with training as follows:

### Stage-1 Training:
* `python stage1_main.py --dataset parta --gpu 0`
```
  -b = Batch size [For ucfqnrf, set 16]
  --dataset = parta / ucfqnrf
  --gpu = GPU Number
  --epochs = Number of epochs to train
```

### Stage-2 CSS-CCNN Training: 
* `python stage2_main.py --dataset parta --gpu 0 --cmax 3000 --num_samples 482`
```
  --dataset = parta / ucfqnrf
  --cmax = Max count value [For ucfqnrf, set 12000]
  --num_samples = Number of samples [For ucfqnrf, set 1500]
  --epochs = Epochs [For ucfqnrf, set 200]
```

### Stage-2 CSS-CCNN++ Training:
* `python stage2_main++.py --dataset parta --gpu 0 --cmax 3500 --num_samples 482`
```
  --dataset = parta / ucfqnrf
  --cmax = Max count value [For ucfqnrf, set 12000]
  --num_samples = Number of samples [For ucfqnrf, set 1500]=
  --epochs = Epochs [For ucfqnrf, set 200]
```

# Results

![Visualisations](/resources/cssccnn_main_prediction_results.png)

## License

See the [LICENSE](https://github.com/val-iisc/css-ccnn/blob/master/LICENSE) file for more details.

## Citation

If you find this work useful in your research, please consider citing the paper:

```
@article{CSSCNN20,
    title = {Completely Self-Supervised Crowd Counting via Distribution Matching},
    author = {Babu Sam, Deepak and Agarwalla, Abhinav and Joseph, Jimmy and Sindagi, A. Vishwanath and Babu, R. Venkatesh and Patel, M. Vishal},
    journal = {arXiv preprint arXiv:2009.06420},
    Year = {2020}
}