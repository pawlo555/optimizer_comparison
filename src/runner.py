import subprocess


commands = ["python train.py --opt adamw --momentum 0.90 --weight-decay 0.00001 --lr 0.001 --model resnet50_custom --experiment-name my-resnet50-cifar100-2 --epochs 1 -b 96 --dataset CIFAR100",
            "python train.py --opt adamw --momentum 0.90 --weight-decay 0.00001 --lr 0.001 --model resnet50_custom --experiment-name my-resnet50-cifar100-3 --epochs 1 -b 96 --dataset CIFAR100"]

for command in commands:
    return_code = subprocess.call(command)
