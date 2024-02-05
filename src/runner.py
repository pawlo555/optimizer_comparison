import subprocess


commands = [
    "python train.py --opt lion  --weight-decay 0.001  --lr 0.0001 -b 128 --experiment-name lion-resnet18-b128-0001               --epochs 80 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom --dataset CIFAR100",
    "python train.py --opt lion  --weight-decay 0.001  --lr 0.0001 -b 128 --experiment-name lion-resnet18-b128-0001-restarted     --epochs 80 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom --dataset CIFAR100 --permute-weights --resume=model_79.pth",

    "python train.py --opt adam  --weight-decay 0.0001 --lr 0.001  -b 256 --experiment-name adam-resnet18-b256                    --epochs 80  --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom --dataset CIFAR100",
    "python train.py --opt adam  --weight-decay 0.0001 --lr 0.001  -b 256 --experiment-name adam-resnet18-b256-restarted          --epochs 80  --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom --dataset CIFAR100 --permute-weights --resume=model_79.pth",

    "python train.py --opt asam-adam --weight-decay 0.0001 --lr 0.001  -b 256 --experiment-name asam-adam-resnet18-b256           --epochs 200 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom --dataset CIFAR100",
    "python train.py --opt asam-adam --weight-decay 0.0001 --lr 0.001  -b 256 --experiment-name asam-adam-resnet18-b256-restarted --epochs 200 --lr-scheduler cosineannealinglr --momentum 0.90 --model resnet18_custom --dataset CIFAR100 --permute-weights --resume=model_199.pth",
]

for command in commands:
    return_code = subprocess.call(command)
