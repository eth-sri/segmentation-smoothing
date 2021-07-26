# Code for Scalable Certified Segmentation via Randomized Smoothing

This code base contains code for both semantic image segmentation and pointcloud part segmentation.
The folder `HRNet-Semantic-Segmentation` contains a modified version of the [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation) repository for semantic segmentation
and the folder `Pointnet_Pointnet2_pytorch` contains a modified version of the [Pointnet](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) for pointclouds.


## Setup
We recommend the creation of two separate `conda` environments for the two code bases.

``` shell
bash setup.sh # patch codes bases

conda create -n HrNet python=3.6
conda activate HrNet
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
pip install -r HRNet-Semantic-Segmentation/requirements.txt
pip install -r requirements.txt
conda deactivate


conda create -n PointNet python=3.6
conda activate PointNet
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements_pointnet.txt
pip install -r requirements.txt
conda deactivate
```

## Usage 

### Semantic Image Segmentation
All commands in this section assume that the HrNet environment is activated (`conda activate HrNet`).

#### Training 
Follow the instructions in [HRNet-Semantic-Segmentation/README.md](HRNet-Semantic-Segmentation/README.md) to download the datasets (install the pascal context api and follow the instructions under the heading Data preparation) and pre-trained models.
Then inside the folder `HrNet-Semnatic-Segmentation` training can be started via

``` shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train_adv.py --cfg experiments/cityscapes/train.yml
python -m torch.distributed.launch --nproc_per_node=8 tools/train_adv.py --cfg experiments/pascal_ctx/train.yaml
```

for Citycapes and Pascal Context respectively. 
To train the model with consistency regularization (Table 5, Appendix B.3) use:

``` shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py --cfg experiments/cityscapes/train --sigma 0.25  --name cityscapes_025_conistency_m2_l5_e05 --consistency True --consistency-m 2 TRAIN.BATCH_SIZE_PER_GPU 1 TEST.BATCH_SIZE_PER_GPU 1

```

Obtain the trained models and place them directly in `HerNet-Semantic-Segmentation`:

``` shell
cp HRNet-Semantic-Segmentation/cityscapes/cityscapes/train_rand/best.pth cityscapes.pth
cp HRNet-Semantic-Segmentation/pascal/pascal_ctx/cls_59_rand_longer/best.pth pascal.pth
```

Alternatively our pre-trained models can also be [downloaded](https://files.sri.inf.ethz.ch/segmentation-smoothing/models.tar.gz).

#### Inference
Inference can be performed via 

``` shell
python tools/test_smoothing.py --cfg experiments/cityscapes/train.yml --sigma <sigma> --tau <tau> -n <n> -n0 <n0> -N <N> TEST.MODEL_FILE cityscapes.pth TEST.SCALE_LIST <scale>, TEST.FLIP_TEST False GPUS 0, TEST.BATCH_SIZE_PER_GPU <batch_size> 

python tools/test_smoothing.py --cfg experiments/pascal_ctx/train.yaml --sigma <sigma> --tau <tau> -n <n> -n0 <n0> -N <N> TEST.MODEL_FILE pascal.pth TEST.SCALE_LIST <scale>, TEST.FLIP_TEST False GPUS 0, TEST.BATCH_SIZE_PER_GPU <batch_size> 
```

for Cityscapes and Pascal Context respectively. The parameters in angle brackets are names as in the paper and can be specified as desired. Note the comma after `<scale>`, a correct specification would thus be `0.25,`. For multiple scales these can be provided as a comma separated list, e.g. `0.25,0.5`. For the `batch_size`please consult Table 3 in the appendix of the paper.

#### Result Table

Running the previously discussed inference statements produces logs in an `logs` folder as a sibling of this folder.
By running 

``` shell
python get_semantic_results.py --dataset cityscapes
python get_semantic_results.py --dataset pascal_ctx
```

in the current folder result tables can be obtained.


#### Adversarial Attack

To perform the adversarial attack shown in Figures 1, 7, 8 use:

``` shell
python tools/attack.py --crop False --cfg experiments/cityscapes/train.yml TEST.MODEL_FILE hrnet_w48_cityscapes_cls19_1024x2048_ohem_trainvalset.pth GPUS 0, TEST.SCALE_LIST 1.0,
```

### Pointcloud Part Segmentation
All commands in this section assume that the PointNet environment is activated (`conda activate PointNet`) and commands are run from the `Pointnet_Pointnet2_pytorch` folder.


#### Training
Follow the instructions in [Pointnet_Pointnet2_pytorch/README.md](Pointnet_Pointnet2_pytorch/README.md) to download and prepare the ShapeNet dataset.
Then training can be run, using:

``` shell
python train_partseg_rand.py --model pointnet2_part_seg_msg --normal --sigma 0.25 --log_dir sigma025_normal
python train_partseg_rand.py --model pointnet2_part_seg_msg --sigma 0.25 --log_dir sigma025
python train_partseg_rand.py --model pointnet2_part_seg_msg --normal --sigma 0.5 --log_dir sigma05_normal
python train_partseg_rand.py --model pointnet2_part_seg_msg --sigma 0.5 --log_dir sigma05
```

We also provide pre-trained models for [download](https://files.sri.inf.ethz.ch/segmentation-smoothing/models.tar.gz).  Place the downloaded `part_seg` folder in `Pointnet_Pointnet2_pytorch/log/`.

#### Inference

To obtain the l2-results from the paper run:

``` shell
python test_partseg_smooth.py  --log_dir sigma025_normal --normal --batch_size 50 --tau 0.75 0.85 0.95 0.99 -n 1000 1000 10000 10000 -n0 100 -N 100 --sigma 0.25
python test_partseg_smooth.py  --log_dir sigma05_normal --normal --batch_size 50 --tau 0.75 0.85 0.95 0.99 -n 1000 1000 10000 10000 -n0 100 -N 100 --sigma 0.50
python test_partseg_smooth.py  --log_dir sigma05_normal --normal --batch_size 50 --tau 0.75 -n 1000 -n0 100 -N 100 --sigma 0.125
python test_partseg_smooth.py  --log_dir sigma05_normal --normal --batch_size 50 --tau 0.75 -n 1000 -n0 100 -N 100 --sigma 0.25
python test_partseg_smooth.py  --log_dir sigma05 --batch_size 50 --tau 0.75 0.85 0.95 0.99 -n 1000 1000 10000 10000 -n0 100 -N 100 --sigma 0.50
python test_partseg_smooth.py  --log_dir sigma05 --batch_size 50 --tau 0.75 -n 1000 -n0 100 -N 100 --sigma 0.125
python test_partseg_smooth.py  --log_dir sigma05 --batch_size 50 --tau 0.75 -n 1000 -n0 100 -N 100 --sigma 0.25

```


To obtain the rotation results from the paper run:

``` shell
python test_partseg_smooth_rot.py --log_dir sigma05_normal --batch_size 50 --tau 0.75 -n 1000  -n0 100 -N 100 --sigma 0.25 --normal
python test_partseg_smooth_rot.py --log_dir sigma05_normal --batch_size 50 --tau 0.75 -n 1000  -n0 100 -N 100 --sigma 0.125 --normal
```


#### Result Table
Finally result tables can be obtained by running the following commands in the current folder:

``` shell
python get_pointcloud_results.py --dataset partseg_l2
python get_pointcloud_results.py --dataset partseg_rot
```


### Synthetic Data
To recreate the plots for Section 6.1 run (from either environment):

``` shell
python synthetic.py
```

### Plots
To recreate the remaining plots run:

``` shell
python plot_tau_vs_cert.py
```

