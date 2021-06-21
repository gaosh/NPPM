# Network Pruning via Performance Maximization
PyTorch Implementation of [Network Pruning via Performance Maximization](https://openaccess.thecvf.com/content/CVPR2021/papers/Gao_Network_Pruning_via_Performance_Maximization_CVPR_2021_paper.pdf) (CVPR 2021).

Code uploaded.
# Requirements
pytorch 1.7.1  
# Usage
To train a base model
```
CUDA_VISIBLE_DEVICES=0 python train_model.py
```
To train the pruning algorithm
```
CUDA_VISIBLE_DEVICES=0 python resnet_pm.py --epm_flag True --nf 15 --reg_w 2 --base 3.0
```
To prune the model
```
python pruning_resnet.py 
```
To finetune the model 
```
CUDA_VISIBLE_DEVICES=0 python train_model.py --train_base False
```
# Citation
```
If you found this repository is helpful, please consider to cite:
@InProceedings{Gao_2021_CVPR,
    author    = {Gao, Shangqian and Huang, Feihu and Cai, Weidong and Huang, Heng},
    title     = {Network Pruning via Performance Maximization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {9270-9280}
}
