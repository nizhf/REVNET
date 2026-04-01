# REVNET: Rotation-Equivariant Point Cloud Completion via Vector Neuron Anchor Transformer
This repository contains PyTorch implementation for **REVNET: Rotation-Equivariant Point Cloud Completion via Vector Neuron Anchor Transformer**.

<a href="https://arxiv.org/abs/2601.08558"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>

## Get Start
### Environment setup
Our experiments are conducted on Ubuntu 24.04.  
We use Conda to manage the Python environment.  
```
conda create -n revnet python=3.12 -c conda-forge
conda activate revnet
conda install cuda -c nvidia/label/cuda-12.8.1 -c conda-forge # install development kit for CUDA extensions
pip install -r requirements.txt
```
### Compile CUDA extensions
Chamfer distance (borrowed from https://github.com/ThibaultGROUEIX/ChamferDistancePytorch), EMD (borrowed and adapted from https://github.com/daerduoCarey/PyTorchEMD), and pointnet2 ops (borrowed and adapted from https://github.com/erikwijmans/Pointnet2_PyTorch)
```
export CUDA_HOME=$CONDA_PREFIX
pip install --no-build-isolation ./extensions/chamfer3D ./extensions/PyTorchEMD ./extensions/pointnet2_ops_lib
```
### Dockerfile
Also, we provide a docker file with pre-installed environment. Use `docker build -t revnet .` to build the image.  
Run with the following command, replace GPU configurations and the dataset paths to yours. 
<pre>
docker run --rm -it --gpus all -v /path/to/KITTI:/home/revnet/workspace/data/KITTI -v /path/to/MVP:/home/revnet/workspace/data/MVP -v /path/to/PCN:/home/revnet/workspace/data/PCN revnet /bin/bash
</pre>

## Datasets
### MVP
Following [VRCNet](https://github.com/paul007pl/VRCNet), download all h5 files from [here](https://drive.google.com/drive/folders/1ylC-dYFM45KW4K9tPyljBSVyetazCEeH) and put them in data/MVP/.
### PCN Car
Following **PCN Dataset** section of [PoinTr repository](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md) to download and unzip the file into data/PCN/.
### KITTI
The download link from **KITTI** section of [PoinTr repository](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md) is no longer valid. We share them [here](https://tumde-my.sharepoint.com/:f:/g/personal/zhifan_ni_tum_de/IgAts7x7FVflQ7p7QZdPHtASAQJ27dB32oExfJ1E3SbbmkQ). If this violates your rights, please contact us to delete them.
### File Structure
The structure of this workspace should be like:
<pre>
.
├── README.md
├── requirements.txt
├── train.py
├── eval.py
├── data
│   ├── KITTI
│   │   ├── bboxes
│   │   ├── cars
│   │   └── tracklets
│   ├── MVP
│   │   ├── mvp_train_input.h5
│   │   ├── mvp_train_gt_8192pts.h5
│   │   ├── mvp_test_input.h5
│   │   ├── mvp_test_gt_8192pts.h5
│   │   └── ...
│   └── PCN
│       ├── test
│       ├── train
│       └── val
├── experiments
│   └── runs
│       ├── revnet_mvp
│       └── ...  # experiments are stored here
└── ...  # other files and folders
</pre>

## Evaluation with Pretrained Weights
You can download our pretrained weights [here](https://tumde-my.sharepoint.com/:f:/g/personal/zhifan_ni_tum_de/IgAts7x7FVflQ7p7QZdPHtASAQJ27dB32oExfJ1E3SbbmkQ) and put them in ./experiments/runs
### On MVP Dataset
```
python eval.py --exp_name revnet_mvp --ckpt best_cd_p.pth --save_path ./experiments
```
### On KITTI Dataset
```
python eval.py --exp_name revnet_kitti_pcn --ckpt best_cd_p.pth --save_path ./experiments
python run_kitti_metrics.py --experiment_path ./experiments/runs/revnet_kitti_pcn --dataset PCN --batch_size 512 --num_workers 8
```

## Training
### On MVP Dataset
```
python train.py --config cfgs/revnet_mvp.yaml --exp_name revnet_mvp --save_path ./experiments
python eval.py --exp_name revnet_mvp --ckpt best_cd_p.pth --save_path ./experiments
```
### Train on PCN Car and test on KITTI Dataset
```
python train.py --config cfgs/revnet_kitti_pcn.yaml --exp_name revnet_kitti_pcn --save_path ./experiments
python eval.py --exp_name revnet_kitti_pcn --ckpt best_cd_p.pth --save_path ./experiments
python run_kitti_metrics.py --experiment_path ./experiments/runs/revnet_kitti_pcn --dataset PCN --batch_size 512 --num_workers 8
```

## License
MIT License

## Acknowledgements
Our code base is inspired by [PoinTr](https://github.com/yuxumin/PoinTr).

## Citation
If our work is helpful for your research, please consider citing our publication:
<pre>
@article{revnet,
  title={REVNET: Rotation-Equivariant Point Cloud Completion via Vector Neuron Anchor Transformer},
  author={Zhifan Ni and Eckehard Steinbach},
  journal = {arXiv preprint arXiv:2601.08558},
  year={2026},
}
</pre>