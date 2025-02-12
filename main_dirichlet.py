import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
import copy
import torch
import socket
import datetime
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from our_datasets import load_dataset
from networks import load_model
from workers.worker_vision import *
from utils.scheduler import Warmup_MultiStepLR
from utils.utils import *
from utils_match.weight_matching import (
    mlp_permutation_spec,
    resnet20_permutation_spec,
    weight_matching,
    apply_permutation,
)
from utils_match.utils import flatten_params, lerp
from utils_match.plot import plot_interp_acc
from our_datasets.cifar10 import load_cifar10
from our_datasets.tinyimagenet import load_tinyimagenet
from torchvision.datasets import CIFAR10

# from networks.share_fix import fix
from utils.dirichlet import dirichlet_split_noniid

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

dir_path = os.path.dirname(__file__)
nfs_dataset_path1 = "D:\\sgd\\ICML-2023-DSGD-and-SAM\\datasets\\"
nfs_dataset_path2 = "/nfs4-p1/ckx/datasets/"

# torch.set_num_threads(4)


def main(args):
    set_seed(args)

    # check nfs dataset path
    if os.path.exists(nfs_dataset_path1):
        args.dataset_path = nfs_dataset_path1
    elif os.path.exists(nfs_dataset_path2):
        args.dataset_path = nfs_dataset_path2

    log_id = (
        datetime.datetime.now().strftime("%b%d_%H:%M:%S")
        + "_"
        + socket.gethostname()
        + "_"
        + args.identity
    )
    # log_id = datetime.datetime.now().strftime('%b%d_%H:%M:%S') + '_' + args.identity
    # log_id = "321"
    print(args.runs_data_dir)
    print("log_id:", log_id)
    writer = SummaryWriter(log_dir=os.path.join(args.runs_data_dir, log_id))

    probe_train_loader, probe_valid_loader, _, classes = load_dataset(
        root=args.dataset_path,
        name=args.dataset_name,
        image_size=args.image_size,
        train_batch_size=256,
        valid_batch_size=64,
    )
    train_set = CIFAR10(
        root=args.dataset_path, train=True, transform=None, download=True
    )
    label = np.array(train_set.targets)
    split = dirichlet_split_noniid(
        train_labels=label, alpha=args.alpha, n_clients=args.size
    )
    worker_list = []
    # osplit = [1.0 / args.size for _ in range(args.size)]
    for rank in range(args.size):
        train_loader, _, _, classes = load_dataset(
            root=args.dataset_path,
            name=args.dataset_name,
            image_size=args.image_size,
            train_batch_size=args.batch_size,
            distribute=True,
            rank=rank,
            split=split,
            seed=args.seed,
            dirichlet=True,
        )
        # model = load_model(args.model, classes, pretrained=args.pretrained).to(args.device)
        model = load_model(args.model, classes, pretrained=args.pretrained).cuda()
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

        if args.amp:
            worker = Worker_Vision_AMP(
                model, rank, optimizer, scheduler, train_loader, args.device
            )
        else:
            worker = Worker_Vision(
                model, rank, optimizer, scheduler, train_loader, args.device
            )
        worker_list.append(worker)

    # define center_model
    center_model = copy.deepcopy(worker_list[0].model)
    # center_model = copy.deepcopy(worker_list[0].model)
    for name, param in center_model.named_parameters():
        for worker in worker_list[1:]:
            param.data += worker.model.state_dict()[name].data
        param.data /= args.size

    P = generate_P(args.mode, args.size)
    iteration = 0
    for epoch in range(args.epoch):
        for worker in worker_list:
            worker.update_iter()
        if args.model == "mlp_turn":
            for worker in worker_list:
                if epoch % 40 >= 10:
                    worker.model.fix0()
                else:
                    worker.model.fix1()
        for _ in range(train_loader.__len__()):
            if args.mode == "csgd":
                for worker in worker_list:
                    worker.model.load_state_dict(center_model.state_dict())
                    worker.step()
                    worker.update_grad()
            else:  # dsgd

                if args.shuffle == "random":
                    P_perturbed = np.matmul(
                        np.matmul(PermutationMatrix(args.size).T, P),
                        PermutationMatrix(args.size),
                    )
                elif args.shuffle == "fixed":
                    P_perturbed = P
                model_dict_list = []
                for worker in worker_list:
                    model_dict_list.append(worker.model.state_dict())
                if args.affine_type == "type1":
                    for worker in worker_list:
                        worker.step()
                        for name, param in worker.model.named_parameters():
                            if iteration % args.loc_step == 0:
                                param.data = torch.zeros_like(param.data)
                                for i in range(args.size):
                                    p = P_perturbed[worker.rank][i]
                                    param.data += model_dict_list[i][name].data * p
                        worker.update_grad()
                elif args.affine_type == "type0":
                    for worker in worker_list:
                        worker.step()
                        if (
                            iteration >= args.breakpoint
                            and iteration % args.loc_step == 0
                        ):  # and iteration != 0:
                            # param.data = torch.zeros_like
                            neibor_model_list = []
                            eps = 1e-6
                            for i in range(args.size):
                                p = P_perturbed[worker.rank][i]
                                if p <= eps:
                                    continue
                                tmp_model = copy.deepcopy(worker_list[i].model.cpu())
                                if i == worker.rank:
                                    neibor_model_list.append((tmp_model, p))
                                else:
                                    # permutation_spec = mlp_permutation_spec(4)
                                    permutation_spec = resnet20_permutation_spec()
                                    final_permutation = weight_matching(
                                        permutation_spec,
                                        flatten_params(worker.model.cpu()),
                                        flatten_params(worker_list[i].model.cpu()),
                                    )

                                    updated_params = apply_permutation(
                                        permutation_spec,
                                        final_permutation,
                                        flatten_params(worker_list[i].model.cpu()),
                                    )
                                    tmp_model = copy.deepcopy(
                                        worker_list[i].model.cpu()
                                    )
                                    tmp_model.load_state_dict(updated_params)

                                    neibor_model_list.append((tmp_model, p))
                                worker_list[i].model = worker_list[i].model.cuda()
                            for name, param in worker.model.named_parameters():
                                param.data = torch.zeros_like(param.data).cpu()
                                for model, p in neibor_model_list:
                                    # print("haha")
                                    param.data += (
                                        model.state_dict()[name].data.cpu() * p
                                    )
                            worker.model = worker.model.cuda()
                            worker.update_grad()
                        else:
                            for name, param in worker.model.named_parameters():
                                if iteration % args.loc_step == 0:
                                    param.data = torch.zeros_like(param.data)
                                    for i in range(args.size):
                                        p = P_perturbed[worker.rank][i]
                                        param.data += model_dict_list[i][name].data * p
                            worker.update_grad()

                elif args.affine_type == "type2":
                    for worker in worker_list:
                        worker.step()
                        for name, param in worker.model.named_parameters():
                            if "bn" in name or iteration % args.loc_step == 0:
                                param.data = torch.zeros_like(param.data)
                                for i in range(args.size):
                                    p = P_perturbed[worker.rank][i]
                                    param.data += model_dict_list[i][name].data * p
                        worker.update_grad()
                elif args.affine_type == "type3":
                    for worker in worker_list:
                        worker.step()
                        for name, param in worker.model.named_parameters():
                            if "bn" in name:
                                param.data = torch.zeros_like(param.data)
                                for i in range(args.size):
                                    p = P_perturbed[worker.rank][i]
                                    param.data += model_dict_list[i][name].data * p
                        worker.update_grad()

                else:
                    for worker in worker_list:
                        worker.step()
                        for name, param in worker.model.named_parameters():
                            param.data = torch.zeros_like(param.data)
                            for i in range(args.size):
                                p = P_perturbed[worker.rank][i]
                                param.data += model_dict_list[i][name].data * p
                        # worker.step() #  performance will get worse
                        worker.update_grad()

            center_model = copy.deepcopy(worker_list[0].model)
            if iteration % args.loc_step == 0:
                for name, param in center_model.named_parameters():
                    for worker in worker_list[1:]:
                        param.data += worker.model.state_dict()[name].data
                    param.data /= args.size

            if iteration % 50 == 0:
                start_time = datetime.datetime.now()
                eval_iteration = iteration
                if args.amp:
                    train_acc, train_loss, valid_acc, valid_loss = eval_vision_amp(
                        center_model,
                        probe_train_loader,
                        probe_valid_loader,
                        None,
                        iteration,
                        writer,
                        args.device,
                    )
                else:
                    train_acc, train_loss, valid_acc, valid_loss = eval_vision(
                        center_model,
                        probe_train_loader,
                        probe_valid_loader,
                        None,
                        iteration,
                        writer,
                        args.device,
                    )
                print(
                    f"\n|\033[0;31m Iteration:{iteration}|{args.early_stop}, epoch: {epoch}|{args.epoch},\033[0m",
                    f"train loss:{train_loss:.4}, acc:{train_acc:.4%}, "
                    f"valid loss:{valid_loss:.4}, acc:{valid_acc:.4%}.",
                    flush=True,
                    end="\n",
                )
            else:
                end_time = datetime.datetime.now()
                print(
                    f"\r|\033[0;31m Iteration:{eval_iteration}-{iteration}, time: {(end_time - start_time).seconds}s\033[0m",
                    flush=True,
                    end="",
                )
            iteration += 1
            if iteration == args.early_stop:
                break
        if iteration == args.early_stop:
            break

    state = {"acc": train_acc, "epoch": epoch, "state_dict": center_model.state_dict()}
    if not os.path.exists(args.perf_dict_dir):
        os.mkdir(args.perf_dict_dir)
    torch.save(state, os.path.join(args.perf_dict_dir, log_id + ".t7"))

    writer.close()
    print("ending")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ## dataset
    parser.add_argument("--dataset_path", type=str, default="datasets")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "CIFAR100", "TinyImageNet"],
    )
    parser.add_argument("--image_size", type=int, default=56, help="input image size")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_swap", type=int, default=None)

    # mode parameter
    parser.add_argument(
        "--mode",
        type=str,
        default="ring",
        choices=["csgd", "ring", "meshgrid", "exponential"],
    )
    parser.add_argument(
        "--shuffle", type=str, default="fixed", choices=["fixed", "random"]
    )
    parser.add_argument("--affine_type", type=str, default="type1")
    parser.add_argument("--loc_step", type=int, default=10)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--backend", type=str, default="gloo")

    # deep model parameter
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet18_M",
        help="deep model name",
        choices=[
            "ResNet18",
            "AlexNet",
            "DenseNet121",
            "AlexNet_M",
            "ResNet18_M",
            "ResNet34_M",
            "DenseNet121_M",
            "mlp",
            "mlp_fix",
            "mlp_position",
            "mlp_turn",
            "resnet_rebasin",
        ],
    )
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--breakpoint", type=int, default=4000)

    # optimization parameter
    parser.add_argument("--lr", type=float, default=0.8, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--warmup_step", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=6000)
    parser.add_argument(
        "--early_stop", type=int, default=6000, help="w.r.t., iterations"
    )
    parser.add_argument("--milestones", type=int, nargs="+", default=[2400, 4800])
    parser.add_argument("--seed", type=int, default=777)
    # parser.add_argument("--device", type=int, default=2)

    # alpha value for Dirichlet split
    parser.add_argument(
        "--alpha", type=float, default=1, help="alpha value for Dirichlet split"
    )

    # parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--amp", action="store_true", help="automatic mixed precision")

    # new for lamb 普通main里置为sgd
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "lamb"])

    args = parser.parse_args()

    args = add_identity(args, dir_path)
    # print(args)
    main(args)
