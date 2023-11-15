# optimizer_comparison

Download bird dataset and extract it to dataset folder
or set path manually

For training:
```bash
cd src
#AdamW
python train.py --opt adamw --momentum 0.9 --weight-decay 1e-04 --lr 0.001 --model resnet34 --experiment-name resnet34-adamw
#Lion
python train.py --opt lion --momentum 0.99 --weight-decay 1e-03 --lr 0.0001 --model resnet34 --experiment-name resnet34-lion
```


AGH dataset:
- extract
- run transforms_dataset.py
- remove _6_0_0 class with one image

To set:
- batch size (but same for all optimizers)
- worker (--workers) to optimize workloads
Training scripts:
```bash
#CIFAR100
python train.py --opt adamw --momentum 0.9 --weight-decay 1e-04 --lr 0.001 --model efficientnet_v2_s --experiment-name adamw-cifar100 --dataset CIFAR100
python train.py --opt sgd --momentum 0.9 --weight-decay 1e-04 --lr 0.001 --model efficientnet_v2_s --experiment-name sgd-cifar100 --dataset CIFAR100
python train.py --opt lion --momentum 0.9 --weight-decay 3e-04 --lr 0.0003 --model efficientnet_v2_s --experiment-name lion-cifar100 --dataset CIFAR100

python train.py --opt adamw --momentum 0.9 --weight-decay 1e-04 --lr 0.001 --model efficientnet_v2_s --experiment-name adamw-plane --dataset plane
python train.py --opt sgd --momentum 0.9 --weight-decay 1e-04 --lr 0.001 --model efficientnet_v2_s --experiment-name sgd-plane --dataset plane
python train.py --opt lion --momentum 0.9 --weight-decay 3e-04 --lr 0.0003 --model efficientnet_v2_s --experiment-name lion-plane --dataset plane
```

Epochs can be reduced if model is not improving