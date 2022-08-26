import argparse
import logging
import os
import time
from datetime import datetime

import torch
from torch.utils import tensorboard
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.distributed as dist
from LeNet import LeNet5


def logstr(str: str):
    logging.info(str)
    print(str)


def valid(epoch, args, net, valid_data_loader, loss_fun, writer, device):
    """
    :param epoch: 当前验证集epoch，用于writer的保存
    :param args: 超参数，主要用rank参数
    :param net: 模型对象
    :param valid_data_loader : 验证数据集
    :param loss_fun: 损失函数对象
    :param writer: SummaryWriter 对象
    :param device: 设备
    :return: 无
    """

    # 测试步骤开始
    net.eval()
    test_correct_sum = 0  # 每一个epoch轮 总数据中正确识别标签的数量
    test_loss_sum = 0  # 每一个epoch轮 总数据/minibatch个loss的和
    with torch.no_grad():
        # TODO:rank == 0 ，设置tqdm只显示rank 0的进度条
        for data in tqdm(valid_data_loader, desc="epoch:{}-valid: ".format(epoch), colour="BLUE",
                         disable=(args.rank != 0)):
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            # 获得网络输出
            y = net(imgs)
            # 计算精度
            max_idx_list = y.argmax(1)  # 参数为1可以按照【0】【1】每一行，返回一个一维的张量
            correct = (max_idx_list == labels).sum()
            loss = loss_fun(y, labels)
            # TODO:all reduce 计算所有进程loss、acc（all reduce方法是同步的方式计算所有进程，无需rank==0，且是inplace=true的）
            dist.all_reduce(correct)
            dist.all_reduce(loss)
            test_correct_sum += correct
            test_loss_sum += loss
    # TODO:rank == 0记录信息
    if args.rank == 0:
        # len(valid_data_loader)为batch的数目，valid_data_loader.batch_size为batch的大小
        accuracy = test_correct_sum / (len(valid_data_loader) * valid_data_loader.batch_size * args.world_size)
        test_loss_mean = test_loss_sum / (len(valid_data_loader) * args.world_size)  # 求每个mini-batch的平均loss
        writer.add_scalar("test/loss", scalar_value=test_loss_mean.item(), global_step=epoch)
        writer.add_scalar("test/acc", scalar_value=accuracy.item(), global_step=epoch)
        logstr("test/loss:\t{}".format(test_loss_mean.item()))
        logstr("test/acc:\t{}".format(accuracy.item()))


def train(start_epoch, args, train_data_loader, valid_data_loader, train_sampler, model, optimizer, loss_fun, scheduler,
          writer, save_state_path, device):
    """
    :param start_epoch: 训练开始是那个epoch，主要服务于 ”中断后继续训练“
    :param epochs:  总epoch数目
    :param train_data_loader:  训练数据集的dataloader
    :param valid_data_loader:  验证数据集的dataloader
    :param model:  模型
    :param optimizer: 优化器
    :param loss_fun: 损失函数
    :param scheduler: 学习率的scheduler
    :param writer: tensorboard记录
    :param state_dict_root: 模型权重保存的root
    :param device: 训练设备
    :return:
    """

    logstr("训练开始")
    start = time.perf_counter()

    # 开始epoch循环
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        # 训练开始标志，能够开启batch-normalization和dropout
        # TODO:设置train_sampler种子，保证训练每次随机顺序都不同
        train_loss_sum = 0
        # TODO:rank == 0 ，设置tqdm只显示rank 0的进度条
        for data in tqdm(train_data_loader, desc="epoch:{}-train: ".format(epoch), colour="BLUE",
                         disable=(args.rank != 0)):
            # 获得数据
            imgs, labels = data
            imgs = imgs.to(device)
            labels = labels.to(device)
            # 将数据输入
            y = model(imgs)
            # 计算loss
            loss = loss_fun(y, labels)
            # 清空模型梯度
            optimizer.zero_grad()
            # 反向传播求导，更新模型梯度
            loss.backward()
            # 优化器更新模型的权重
            optimizer.step()
            # TODO:all reduce 计算所有进程loss、acc（all reduce方法是同步的方式计算所有进程，无需rank==0，且是inplace=true的）
            dist.all_reduce(loss)  # 求所有进程一轮mini-batch的loss和
            train_loss_sum += loss

        # 更新学习率
        scheduler.step()
        # TODO:rank == 0记录信息
        if args.rank == 0:
            # len(train_data_loader)为batch 的数量
            train_loss_mean = train_loss_sum / (len(train_data_loader) * args.world_size)
            writer.add_scalar("learning_rate", scalar_value=scheduler.get_last_lr()[0], global_step=epoch)
            writer.add_scalar("train/loss", scalar_value=train_loss_mean.item(),
                              global_step=epoch)  # 记录每一百个bitch（640个）后的loss
            logstr("epoch: {}".format(epoch))
            logstr("learning_rate:\t{}".format(scheduler.get_last_lr()[0]))
            logstr("train/loss:\t{}".format(train_loss_mean.item()))

        # 测试
        valid(epoch, args, model, valid_data_loader, loss_fun, writer, device)

        # 保存多种数据以方便继续训练
        # TODO:rank == 0记录信息
        if args.rank == 0:
            state = {
                "start_epoch": epoch,
                # TODO: 保存模型权重
                # TODO: 由于model加载到ddp对象的module参数中，我们只保存模型参数即可
                "model_state_dict": model.module.state_dict(),
                "Optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(state, save_state_path + "/LenetMnist{0}.pt".format(epoch))
            logstr("模型已保存 -> " + save_state_path + "/LenetMnist{0}.pt".format(epoch))
            logstr("---------------------------------")
    # TODO:rank == 0记录信息
    if args.rank == 0:
        end = time.perf_counter()
        logstr("训练结束，训练耗时：{}".format(end - start))


# TODO: 初始化进程组
def init_DDP(args):
    # python 中参数为地址传递
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    print("launch process {}".format(args.local_rank))

    # 初始化进程组
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method,
                            world_size=args.world_size, rank=args.rank)
    # dist.init_process_group(backend=args.dist_backend)


def main(args):
    # TODO:初始化进程组（调用），并生成rank、world_size、local_rank参数
    init_DDP(args)
    print("rank:{},local_rank:{},cuda:{} is ready!".format(args.rank, args.local_rank, args.local_rank))

    # TODO:设置该进程的GPU设备
    device = torch.device("cuda:{}".format(args.local_rank))

    # 1.图片预处理
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(28, 28)),
        transforms.Normalize([0.5], [0.5])  # 要根据通道数设置均值和方差
    ])

    # 2.加载数据集
    train_dataset = datasets.MNIST(root="./datasets", train=True, transform=compose, download=False)
    valid_dataset = datasets.MNIST(root="./datasets", train=False, transform=compose, download=False)

    # TODO:按进程数（GPU数）划分数据集
    train_sampler = data.distributed.DistributedSampler(dataset=train_dataset)
    valid_sampler = data.distributed.DistributedSampler(dataset=valid_dataset)
    # train_sampler = data.distributed.DistributedSampler(dataset=train_dataset, num_replicas=args.world_size,
    #                                                     rank=args.rank)
    # valid_sampler = data.distributed.DistributedSampler(dataset=valid_dataset, num_replicas=args.world_size,
    #                                                     rank=args.rank)
    if args.rank == 0:
        # 数据集长度
        train_len = len(train_dataset)
        test_len = len(valid_dataset)
        print("训练测试集长度：", train_len)
        print("测试数据集长度：", test_len)
    # TODO:划分batch
    train_data_loader = data.DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                                        num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_memory)
    valid_data_loader = data.DataLoader(dataset=valid_dataset, sampler=valid_sampler, batch_size=args.batch_size,
                                        num_workers=args.num_workers, shuffle=False, pin_memory=args.pin_memory)

    # 3.定义要使用的网络LeNet
    model = LeNet5()
    model.to(device)
    # 4.定义损失函数函数（损失函数已经对minibatch求过平均）
    loss_fun = nn.CrossEntropyLoss().to(device)
    # 5.定义优化器损失函数
    sgd_optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(sgd_optimizer, T_max=args.epochs, eta_min=0.0001)
    # 6.判断是否继续训练
    start_epoch = 0
    if os.path.exists(args.resume_path):
        # TODO: map_location 保证加载到local_rank的gpu上
        checkpoint = torch.load(args.resume_path, map_location=device)
        start_epoch = checkpoint["start_epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        sgd_optimizer.load_state_dict(checkpoint["Optimizer_state_dict"])
        # 检查要加载的权重的张量大小和权重是否一致，将一致的收集起来
        load_weights_dict = {k: v for k, v in checkpoint["model_state_dict"].items()
                             if model.state_dict()[k].numel() == v.numel()}
        # 以不严格的方式加载（比如：class的改变常导致最后一层权重无法读取，该方法能够使模型顺利加载）
        # 能加载的设置为加载权重，不能加载的不改变(默认的初始化权重)
        model.load_state_dict(load_weights_dict, strict=False)
        print("在{}文件上继续训练".format(args.resume_path))

    # 是否使用同步的BN层
    if args.syncBN:
        # 使用SyncBatchNorm后训练会更耗时
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # TODO: 将模型包装到数据并行当中（优化器中加载的参数可以是DDP包装的，也可以是包装后的）
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device],
                                                      output_device=device)

    # 7.log 设置
    # 创建tensorboard来记录网络
    writer = None
    save_state_path = None
    # TODO:rank == 0记录信息
    if args.rank == 0:
        time_str = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        log_path = os.path.join(args.log_root, time_str)  # "./log/" + time_str
        save_state_path = os.path.join(log_path, "state")
        tensorboard_dir = os.path.join(log_path, "tensorboard")
        logging_dir = os.path.join(log_path, "logs")

        if not os.path.exists(save_state_path):
            os.makedirs(save_state_path)
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)

        writer = tensorboard.SummaryWriter(log_dir=tensorboard_dir)
        logging.basicConfig(filename=os.path.join(logging_dir, "train_log.txt"), format='%(asctime)s : %(message)s',
                            level=logging.INFO)
        logstr("batch_size: {}, num_workers: {} ,".format(args.batch_size, args.num_workers) +
               "pin_memory: {}, epochs: {}, ".format(args.pin_memory, args.epochs) +
               "learning_rate: {}, log_root: {}, ".format(args.learning_rate, args.log_root) +
               "resume_path: {}, init_method: {}, ".format(args.resume_path, args.init_method) +
               "init_method: {}, dist_backend: {}, ".format(args.init_method, args.dist_backend) +
               "syncBN: {}".format(args.syncBN))

    # 8.开始训练
    train(start_epoch, args, train_data_loader, valid_data_loader, train_sampler, model, sgd_optimizer, loss_fun,
          scheduler, writer, save_state_path, device)
    # TODO:rank == 0记录信息
    if args.rank == 0:
        writer.close()


def parse_add_args():
    parser = argparse.ArgumentParser('Model')

    parser.add_argument('--batch_size', type=int, default=32, help='batch的大小，DDP真实的batch_size应该为16 * world_size（进程数）')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader 的num_works的大小')
    parser.add_argument('--pin_memory', type=bool, default=True, help='dataloader是否使用锁页内存加载数据')
    parser.add_argument('--epochs', default=200, type=int, help='总epoch的数量')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--log_root', type=str, default='./log/distribute_data_parallel', help='保存权重和tensorboard的root')
    parser.add_argument('--resume_path', type=str, default='', help='继续训练的地址，若为 none则重新训练')
    parser.add_argument('--init_method', type=str, default="env://", help='进程组环境变量初始化 或 TCP初始化-主机的IP+端口号')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='分布式主机间通信协议')
    parser.add_argument('--syncBN', type=bool, default=True, help='是否使用同步的BN')
    # DDP的gpu从torch.launch中设定，为local_rank参数
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_add_args()
    main(args)
