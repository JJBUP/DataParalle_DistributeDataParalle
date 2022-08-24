import argparse
import os
from datetime import datetime

import torch
from torch.nn import DataParallel
from torch.utils import tensorboard
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms
from tqdm import tqdm
import collections.abc
from LeNet import LeNet5


def valid(epoch, net, valid_data_loader, loss_fun, writer, device):
    """
    :param net: 模型对象
    :param valid_data_loader : 验证数据集
    :param writer: SummaryWriter 对象
    :param loss_fun: 损失函数对象
    :param epoch: 当前验证集epoch，用于writer的保存
    :param device: 设备
    :return: 无
    """
    # 测试步骤开始
    net.eval()
    test_correct_sum = 0  # 每一个epoch轮 总数据中正确识别标签的数量
    test_loss_sum = 0  # 每一个epoch轮 总数据/minibatch个loss的和
    with torch.no_grad():
        for imgs, labels in tqdm(valid_data_loader, desc="epoch:{}-valid: ".format(epoch), colour="GREEN"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            # 获得网络输出
            y = net(imgs)
            # 计算精度
            max_idx_list = y.argmax(1)  # 参数为1可以按照【0】【1】每一行，返回一个一维的张量
            test_correct_sum += (max_idx_list == labels).sum()
            # 计算损失
            test_loss_sum = loss_fun(y, labels) + test_loss_sum

    accuracy = test_correct_sum / (
            len(valid_data_loader) * valid_data_loader.batch_size)  # len(valid_data_loader) 为batch的数目
    test_loss_mean = test_loss_sum / len(valid_data_loader)  # 因为损失函数自动对每一个minibitch求平均，所以loss的和为每一个minibitch的数量
    print("acc: {} , loss: {}".format(accuracy.item(), test_loss_mean.item()))
    writer.add_scalar("test/acc", scalar_value=accuracy, global_step=epoch)
    writer.add_scalar("test/loss", scalar_value=test_loss_mean, global_step=epoch)


def train(start_epoch, epochs, train_data_loader, valid_data_loader, model, optimizer, loss_fun, scheduler, writer,
          state_dict_root, device):
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

    # 开始epoch循环
    for epoch in range(start_epoch, epochs):
        # 训练开始标志，能够开启batch-normalization和dropout
        batch_num_train = 0  # 一个epoch中batch 的数目
        train_loss_sum = 0
        for data in tqdm(train_data_loader, desc="epoch:{}-train: ".format(epoch), colour="GREEN"):
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
            train_loss_sum += loss
            batch_num_train += 1
        tain_loss_mean = train_loss_sum / batch_num_train
        writer.add_scalar("train/loss", scalar_value=tain_loss_mean, global_step=epoch)  # 记录每一百个bitch（640个）后的loss
        writer.add_scalar("learning_rate", scalar_value=scheduler.get_last_lr()[0], global_step=epoch)

        # 更新学习率
        scheduler.step()

        # 测试
        valid(epoch, model, valid_data_loader, loss_fun, writer, device)

        # 保存多种数据以方便继续训练

        state = {
            "start_epoch": epoch,
            # TODO: 保存模型权重
            # TODO: 由于model加载到DataParallel对象的module参数中，我们只保存模型参数即可
            "model_state_dict": model.module.state_dict(),
            "Optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(state, state_dict_root + "/LenetMnist{0}.pt".format(epoch))
        # print("模型已保存")


def main(args):
    # 定义超参数
    # TODO:设置可使用的GPU环境变量，改变其逻辑顺序
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_list
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 指定训练设备

    # 1.图片预处理
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(28, 28)),
        transforms.Normalize([0.5], [0.5])  # 要根据通道数设置均值和方差
    ])

    # 2.加载数据集
    train_dataset = datasets.MNIST(root="./datasets", train=True, transform=compose, download=False)
    test_dataset = datasets.MNIST(root="./datasets", train=False, transform=compose, download=False)
    # 数据集长度
    train_len = len(train_dataset)
    test_len = len(test_dataset)
    print("训练测试集长度：", train_len)
    print("测试数据集长度：", test_len)
    train_data_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                        shuffle=True, drop_last=True)
    valid_data_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                        shuffle=True, drop_last=True)

    # 3.定义要使用的网络LeNet
    leNet = LeNet5()
    # 4.定义损失函数函数（损失函数已经对minibatch求过平均）
    loss_fun = nn.CrossEntropyLoss().to(device)
    # 5.定义优化器
    sgd_optimizer = optim.SGD(leNet.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(sgd_optimizer, T_max=args.epochs, eta_min=0.0001)
    # 6.判断是否继续训练
    start_epoch = 0
    if args.resume_path is not None and args.resume_path != '':
        # TODO: map_location 保证加载指定gpu上
        checkpoint = torch.load(args.resume_path, map_location=device)
        start_epoch = checkpoint["start_epoch"] + 1
        leNet.load_state_dict(checkpoint["model_state_dict"])
        sgd_optimizer.load_state_dict(checkpoint["Optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # TODO: 将模型包装到数据并行当中
    gpu_num = len(args.GPU_list.split(","))  # 因为设置过环境变量GPU逻辑编号从0开始
    gpu_id_list = [int(gpu_id) for gpu_id in range(gpu_num)]
    model = DataParallel(leNet.to(device), device_ids=gpu_id_list, output_device=gpu_id_list[0])
    # 7.log 设置
    # 创建tensorboard来记录网络
    time_str = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    log_path = os.path.join(args.log_root, time_str)  # "./log/" + time_str
    writer = tensorboard.SummaryWriter(log_path)

    # 8.开始训练
    train(start_epoch, args.epochs, train_data_loader, valid_data_loader, model, sgd_optimizer, loss_fun, scheduler,
          writer,
          log_path, device)
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser('Model')

    parser.add_argument('--batch_size', type=int, default=24 * 4, help='batch的大小')
    parser.add_argument('--num_workers', type=int, default=3, help='dataloader 的num_works的大小')
    parser.add_argument('--epochs', default=100, type=int, help='总epoch的数量')
    parser.add_argument('--GPU_list', type=str, default="0,1,2,3", help='可用gpu编号')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--log_root', type=str, default='./log/data_parallel', help='保存权重和tensorboard的root')
    parser.add_argument('--resume_path', type=str, default='', help='继续训练的地址，若为 none则重新训练')
    return parser.parse_args()


if __name__ == '__main__':
    # 参数
    args = parse_args()
    main(args)
