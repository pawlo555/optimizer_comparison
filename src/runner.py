import subprocess


commands = [
    "python train.py --opt sgd   --weight-decay 0.0001 --lr 0.001  -b 128 --experiment-name sgd-resnet18-b128       --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt adam  --weight-decay 0.0001 --lr 0.001  -b 128 --experiment-name adam-resnet18-b128      --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt adamw --weight-decay 0.0001 --lr 0.001  -b 128 --experiment-name adamw-resnet18-b128     --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt lion  --weight-decay 0.001  --lr 0.0001 -b 128 --experiment-name lion-resnet18-b128-0001 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt lion  --weight-decay 0.0003 --lr 0.0003 -b 128 --experiment-name lion-resnet18-b128-0003 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt lion  --weight-decay 0.0002 --lr 0.0005 -b 128 --experiment-name lion-resnet18-b128-0005 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",

    "python train.py --opt sgd   --weight-decay 0.0001 --lr 0.001  -b 256 --experiment-name sgd-resnet18-b256       --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt adam  --weight-decay 0.0001 --lr 0.001  -b 256 --experiment-name adam-resnet18-b256      --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt adamw --weight-decay 0.0001 --lr 0.001  -b 256 --experiment-name adamw-resnet18-b256     --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt lion  --weight-decay 0.001  --lr 0.0001 -b 256 --experiment-name lion-resnet18-b256-0001 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt lion  --weight-decay 0.0003 --lr 0.0003 -b 256 --experiment-name lion-resnet18-b256-0003 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt lion  --weight-decay 0.0002 --lr 0.0005 -b 256 --experiment-name lion-resnet18-b256-0005 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",

    "python train.py --opt sgd   --weight-decay 0.0001 --lr 0.001  -b 512 --experiment-name sgd-resnet18-b512       --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt adam  --weight-decay 0.0001 --lr 0.001  -b 512 --experiment-name adam-resnet18-b512      --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt adamw --weight-decay 0.0001 --lr 0.001  -b 512 --experiment-name adamw-resnet18-b512     --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt lion  --weight-decay 0.001  --lr 0.0001 -b 512 --experiment-name lion-resnet18-b512-0001 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt lion  --weight-decay 0.0003 --lr 0.0003 -b 512 --experiment-name lion-resnet18-b512-0003 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",
    "python train.py --opt lion  --weight-decay 0.0002 --lr 0.0005 -b 512 --experiment-name lion-resnet18-b512-0005 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 60 --dataset CIFAR100",

    "python train.py --opt sam-sgd  --weight-decay 0.0001 --lr 0.001  -b 128 --experiment-name sam-sgd-resnet18-b128  --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 180 --dataset CIFAR100",
    "python train.py --opt asam-sgd --weight-decay 0.0001 --lr 0.001  -b 128 --experiment-name asam-sgd-resnet18-b128 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 180 --dataset CIFAR100",

    "python train.py --opt sam-sgd  --weight-decay 0.0001 --lr 0.001  -b 256 --experiment-name sam-sgd-resnet18-b256  --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 180 --dataset CIFAR100",
    "python train.py --opt asam-sgd --weight-decay 0.0001 --lr 0.001  -b 256 --experiment-name asam-sgd-resnet18-b256 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 180 --dataset CIFAR100",

    "python train.py --opt sam-sgd  --weight-decay 0.0001 --lr 0.001  -b 512 --experiment-name sam-sgd-resnet18-b512  --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 180 --dataset CIFAR100",
    "python train.py --opt asam-sgd --weight-decay 0.0001 --lr 0.001  -b 512 --experiment-name asam-sgd-resnet18-b512 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom  --epochs 180 --dataset CIFAR100",
]

for command in commands:
    return_code = subprocess.call(command)
