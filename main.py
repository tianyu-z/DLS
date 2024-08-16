import os
import copy
import torch
import socket
import datetime
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from our_datasets import load_dataset
from networks import load_model, load_valuemodel
from workers.worker_vision import Worker_Vision, Worker_Vision_AMP, DQNAgent
from utils.scheduler import Warmup_MultiStepLR
from utils.utils import (
    set_seed,
    add_identity,
    generate_P,
    save_model,
    evaluate_and_log,
    update_center_model,
    update_dsgd,
    update_dqn_chooseone,
    update_csgd,
    update_heuristic,
    Merge_History,
    update_heuristic_2
)
from easydict import EasyDict
import wandb
from utils.dirichlet import (
    dirichlet_split_noniid,
    create_dataloaders,
    create_simple_preference,
    create_IID_preference,
)
from torchvision.datasets import CIFAR10
import numpy as np
from fire import Fire
from tqdm import trange

# torch.set_num_threads(4)


def main(
    dataset_path="datasets",
    dataset_name="cifar10",  # cifar10_test
    image_size=56,
    batch_size=512,
    n_swap=None,
    mode="dqn_chooseone",
    shuffle="fixed",
    size=10,
    port=29500,
    backend="gloo",
    model="ResNet18_M",
    pretrained=1,
    lr=0.1,
    wd=0.0,
    gamma=0.1,
    momentum=0.0,
    warmup_step=0,
    early_stop=6000,
    milestones=[2400, 4800],
    seed=666,
    device="cuda:0",
    amp=False,
    sample=0,
    n_components=0,
    nonIID=True,
    project_name="decentralized",
    alpha=0.3,
    state_size=144,
    valuemodel_hiddensize=320,
    merge_step=1
):
    sub_dict_keys = [
        # "dataset_name",
        "size",
        # "image_size",
        # "batch_size",
        "mode",
        # "model",
        # "pretrained",
        # "alpha",
        "early_stop",
        "nonIID",
        "merge_step"
    ]
    args = EasyDict(locals().copy())
    # set_seed(args)
    set_seed(seed, torch.cuda.device_count())
    dir_path = os.path.dirname(__file__)
    args = add_identity(args, dir_path)
    sub_dict_str = "_".join([key + str(args[key]) for key in sub_dict_keys])
    # 初始化Wandb项目
    wandb.init(project=project_name, name=sub_dict_str)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    dir_path = os.path.dirname(__file__)

    # set_seed(args)
    set_seed(seed, torch.cuda.device_count())
    args = EasyDict(locals().copy())
    args = add_identity(args, dir_path)

    # check nfs dataset path

    log_id = (
        datetime.datetime.now().strftime("%b%d_%H:%M:%S")
        + "_"
        + socket.gethostname()
        + "_"
        + args.identity
    )
    writer = SummaryWriter(log_dir=os.path.join(args.runs_data_dir, log_id))

    probe_train_loader, probe_valid_loader, _, classes = load_dataset(
        root=args.dataset_path,
        name=args.dataset_name,
        image_size=args.image_size,
        train_batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
        return_dataloader=True,
    )

    train_set, valid_set, _, nb_class = load_dataset(
        dataset_path, dataset_name, image_size, return_dataloader=False
    )
    if nonIID:
        all_class_weights = create_simple_preference(
            args.size, nb_class, important_prob=0.8
        )
    else:
        all_class_weights = create_IID_preference(args.size, nb_class)
    train_dataloaders = create_dataloaders(
        train_set, args.size, len(train_set) // args.size, args.batch_size, all_class_weights, 
    )
    valid_dataloaders = create_dataloaders(valid_set, args.size, 
                                           len(train_set) // args.size, args.batch_size, all_class_weights, )
    worker_list = []
    trainloader_length_list = []
    
    for rank in range(args.size):

        # train_loader, _, _, classes = load_dataset(
        #     root=args.dataset_path,
        #     name=args.dataset_name,
        #     image_size=args.image_size,
        #     train_batch_size=args.batch_size,
        #     distribute=True,
        #     rank=rank,
        #     split=split,
        #     seed=args.seed,
        # )
        train_loader = train_dataloaders[rank]
        trainloader_length_list.append(len(train_loader))
        model = load_model(args.model, nb_class, pretrained=args.pretrained).to(
            args.device
        )
        if args.mode == "dqn_chooseone":
            value_model = load_valuemodel(
                args.state_size, valuemodel_hiddensize, args.size
            )  # 这里的144是pca weights的压缩后的维度 pca weights的shape是[144,144]

        optimizer = SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum
        )
        # scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        scheduler = Warmup_MultiStepLR(
            optimizer,
            warmup_step=args.warmup_step,
            milestones=args.milestones,
            gamma=args.gamma,
        )
        # worker是训练器
        if args.amp:
            worker = Worker_Vision_AMP(
                model, rank, optimizer, scheduler, train_loader, args.device
            )
        else:
            if args.mode == "dqn_chooseone":
                worker = DQNAgent(
                    model, value_model, rank, optimizer, scheduler, train_loader, args, wandb, max_epsilon=0.2
                )
            elif args.mode == "heuristic":
                worker = Worker_Vision(
                    model, rank, optimizer, scheduler, train_loader, args.device
                )
                merge_history = Merge_History(args.size, 5)
            else:
                worker = Worker_Vision(
                    model, rank, optimizer, scheduler, train_loader, args.device
                )
        worker_list.append(worker)
        # model_list.append(model)

    # 定义 中心模型 center_model
    center_model = copy.deepcopy(worker_list[0].model)
    for name, param in center_model.named_parameters():
        for worker in worker_list[1:]:
            param.data += worker.model.state_dict()[name].data
        param.data /= args.size

    P = generate_P(args.mode, args.size)

    for iteration in trange(args.early_stop):
        # 这里的epoch是平均epoch
        epoch = iteration // (
            sum(trainloader_length_list) / len(trainloader_length_list)
        )

        # if iteration % len(train_loader) == 0:
        # for worker in worker_list:
        # worker.update_iter()

        # 对每个trainloader检测是否遍历完，如果遍历完 则对应的worker 更新trainloader
        for i in range(0, args.size):
            if iteration % trainloader_length_list[i] == 0:
                worker_list[i].update_iter()

        if args.mode == "csgd":
            update_csgd(worker_list, center_model)
        elif args.mode == "dqn_chooseone":
            update_dqn_chooseone(worker_list, iteration, wandb, merge_step)
        elif args.mode == "heuristic":
            update_heuristic_2(worker_list, args, merge_history)
        else:  # dsgd
            update_dsgd(worker_list, P, args)

        center_model = update_center_model(worker_list)

        if iteration % 50 == 0:

            train_acc, train_loss, valid_acc, valid_loss = evaluate_and_log(
                center_model,
                probe_train_loader,
                probe_valid_loader,
                iteration,
                epoch,
                writer,
                args,
                wandb,
                mode,
                worker_list,
                train_dataloaders,
                valid_dataloaders
                
            )

        if iteration == args.early_stop:
            save_model(center_model, train_acc, epoch, args, log_id)
            break

    writer.close()
    print("Ending")


if __name__ == "__main__":

    Fire(main(dataset_path="/mnt/nas/share2/home/lwh/cifar_datasets",
    dataset_name="CIFAR10",
    image_size=56,
    batch_size=512,
    n_swap=None,
    mode="heuristic",
    shuffle="fixed",
    size=5,
    port=29500,
    backend="gloo",
    model="ResNet18_M",
    pretrained=1,
    lr=0.1,
    wd=0.0,
    gamma=0.1,
    momentum=0.0,
    warmup_step=0,
    # epoch=6000,
    early_stop=6000,
    milestones=[2400, 4800],
    seed=666,
    device="cuda:0",
    amp=False,
    sample=5,
    n_components=0, # 这个参数没有用
    nonIID=True,
    project_name="decentralized",
    alpha=0.3,
    state_size=154,
    valuemodel_hiddensize=320,
    merge_step=32))
