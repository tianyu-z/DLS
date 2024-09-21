from torchvision.datasets import CIFAR100
import torchvision.transforms as tfs
transforms = tfs.Compose(
            [
                tfs.Resize((32, 32)),
                tfs.ToTensor(),
                tfs.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ]
        )

train_set = CIFAR100('/apdcephfs/csp/mmvision/home/lwh/FLASH-RL/data', True, transforms, download=True)