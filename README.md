# 项目概述

使用Mnist数据集实现DP单机多卡训练和DDP分布式单机单卡/单机多卡训练

# DP训练方法



| 方法                                                         | 参数                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [DataParallel()](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) | 该函数实现了在module级别上的数据并行使用，注意batch size要大于GPU的数量。 |
|                                                              | module, device_ids=None, output_device=None, dim=0           |
|                                                              | **module：**需要多GPU训练的网络模型<br/>**device_ids：list[ int or torch.device]**，GPU的编号（默认全部GPU），手动设定<br/>**output_device： int or torch.device**，GPU的主设备编号，默认是第0块GPU（默认是device_ids[0]) |
|                                                              | 例如：**output_device=gpus[0] **指定的第 0 张卡为主卡，相当于参数服务器，其向其他卡广播其参数，参与训练的 GPU 参数**device_ids=gpus**；<br/>output_device一般不使用，默认为GPU0即可，原因是代码将使用更少的判断兼容单GPU的训练！ |

**（1）DP数据并行处理机制：**

DataParallel系统通过将整个小批次（minibatch）数据加载到主线程上，然后将子小批次（ub-minibatches）数据分散到整个GPU网络。

**（2）DP数据并行处理机制细节：**

1. **划分minibatch数据。**minibatch数据 加载到主 master GPU (GPU 0)，然后将 minibatch 的数据均分成多份，分别送到对应的 GPU （batchsize切分后在每个GPU上变小，这要求batchsize大小为原来倍数）。
2. **在 GPUs 之间复制模型。**与 Module 相关的所有数据也都会复制多份。
3. **在每个GPU之上运行前向传播，计算输出。**PyTorch 使用多线程来并行前向传播，每个 GPU 在单独的线程上将针对各自的输入数据独立并行地进行 forward 计算。
4. **在 master GPU 之上收集（gather）输出，计算损失。**即通过将网络输出与批次中每个元素的真实数据标签进行比较来计算损失函数值。
5. **把损失在 GPUs 之间 scatter，在各个GPU之上运行后向传播，计算参数梯度。**
6. **在 GPU 0 之上归并梯度all-reduece。**
7. **更新梯度参数。**
   - 进行梯度下降，并更新主GPU上的模型参数。
   - 由于模型参数仅在主GPU上更新，而其他从属GPU此时并不是同步更新的，所以需要将更新后的模型参数复制到剩余的从属 GPU 中，以此来实现并行。



**（3）模型权重保存问题：**

> torch.save：正常保存
>
> torch.load：maplocation指定模型权重的加载位置，
>
> ​	首先，若模型在GPU，应将模型加载到GPU中而不是CPU，即maplocation=“cuda"或“cuda”，若加载到 CPU将会再将模型复制到相应GPU，减缓了速度。
>
> ​	其次，应使用DP设置的主GPU加载权重，因为主GPU会将权重复制到其他线程上，
>
> ​	DP的output_device一般不使用，默认为逻辑cuda：0即可，这样maplocation应该设置为cuda或cuda:0,





# DDP使用流程

`Pytorch` 中分布式的基本使用流程如下：

1. 在使用 `distributed` 包的任何其他函数之前，需要使用 `init_process_group` 初始化进程组，同时初始化 `distributed` 包。
2. 创建分布式并行模型 `DDP(model, device_ids=device_ids)`
3. 为数据集创建 `Sampler`
4. 使用启动工具 `torch.distributed.launch` 在每个主机上执行一次脚本，开始训练
5. 使用 `destory_process_group()` 销毁进程组

**该流程即适用于单机多卡，也适用于分布式的多级多卡**，但是单机单卡的使用时某些地方要令作判断（吐槽：单机单卡凑什么热闹，毕竟先完成单卡的再该多卡的保留一份不好嘛，还整的多卡流程不清晰了）

**（1）启动**

```bash
# 单机多卡

python -m torch.distributed.launch --nproc_per_node 4 --use_env train.py

# 若为本地的多GPU测试 则 master_addr / master_port 参数可不设置，系统将自动设置为
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29500'

# nproc_per_node 每台主机的进程数目，设置为可使用GPU数目，即进程GPU是一对一的
# 	 设置环境中GPU命令修改为：CUDA_VISIBLE_DEVICES=0,1 python -m ......
# nnodes 主机个数，若为本地训练设置为1
# node_rank 主机的优先级，rank = 0 为master，本地训练设置为0

# 多机多卡
#机器1：
python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr='192.168.1.105' --master_port='12345' --use_env train.py
#机器2：
python -m torch.distributed.launch --nproc_per_node 4 --nnodes 2 --node_rank 0 --master_addr='192.168.1.105' --master_port='12345' --use_env train.py
```

**（2）初始化进程组**

环境变量的形式初始化

```python
import torch.distributed as dist

# 读取环境参数，便于下一步使用
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"]) # LOCAL_RANK环境需要-use_env参数
# 初始化进程组
dist.init_process_group(backend=args.dist_backend,# nccl
                        # env://环境变量初始化TCP://192.168.1.105:23333TCP初始化
                        init_method=args.init_method,
                        world_size=args.world_size,
                        rank=args.rank)

```

**（3）设置本机训练Device**

设置本机训练Device可分为两种方式

方式1：设置本进程的可见GPU

```python
# 设置本进程可见GPU，即设置该GPU为进程逻辑cuda:0
local_rank = int(os.environ["LOCAL_RANK"])
# 设置进程的GPU环境
troch.cuda.set_device("cuda:{}".format(local_rank))
# 或
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

# 设置model/tesor/DDP等所使用的设备时的指定方式
device = torch.device("cuda"/"cuda:0")
# ......
# model.to(device)/model.cuda()/model/cuda(0)
# DDP(model,device_id=[device],output_device=device)
# 
```

方式2：设置本进程所使用的device

```python
# 直接创建设备对象，保证该进程任何位置使用to(device)完成设备指定
local_rank = int(os.environ["LOCAL_RANK"])

# 设置model/tesor/DDP等所使用的设备时的指定方式
device = torch.device("cuda:{}".format(local_rank))
......
# model/DDP/tensor中要将张量移动到该device，如下：
model.to(device)
DDP(model,device_id=[device],output_device=device) # 或将device 直接替换为 0也可
data = data.to(device)
```



**（4）数据分布式采样**

```PYTHON
train_sampler = torch.utils.data.distributed.DistributedSampler(
			      						   dataset=train_dataset,
    									   num_replicas=world_size,
    									   rank=rank)
# 将样本索引每batch_size个元素组成一个list
train_loader = torch.utils.data.DataLoader(train_sampler,# 设置分布式采样
                                           batch_size=64,
                                           pin_memory=Flase,
                                           num_workers=arg.num_workers,
                                           shuffle = False # 默认为Flase，因为DistributedSampler已经时乱序，所以这里可以设置为Flase
                                           )
```

**（5）模型分布式数据并行**

```python
# 实例化模型
model = resnet34(num_classes=num_classes).to(device)

# 权重加载
if os.path.exists(weights_path):
     # 加载权重，map_location设置 local_rank设备可以防止cuda:0的显存爆炸
     checkpoint = torch.load(args.resume_path，maplocation = device)

     start_epoch = checkpoint["start_epoch"] + 1   #用于继续训练（进度条）
     sgd_optimizer.load_state_dict(checkpoint["Optimizer_state_dict"])
     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
     # 检查要加载的权重的张量大小和权重是否一致，将一致的收集起来 
     load_weights_dict = {k: v for k, v in checkpoint["model_state_dict"].items()
                             if model.state_dict()[k].numel() == v.numel()}
     # 以不严格的方式加载（比如：class的改变常导致最后一层权重无法读取，该方法能够使模型顺利加载）
     # 能加载的设置为加载权重，不能加载的不改变(默认的初始化权重)
     model.load_state_dict(load_weights_dict,strict=False)
     print("在{}文件上继续训练".format(args.resume_path))

# 是否使用同步的BN层
if args.syncBN:
     # 使用SyncBatchNorm后训练会更耗时
     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

# 转为DDP模型
# 注意model参数必须已经在device_ids的GPU上了
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device],output_device=device)

```

**（6）学习率设置（可选）**

​		有研究标明扩大了batch_size就是放缩了学习率，所以学习率应该相应扩大一定倍数（GPU个数）

**（7）模型数据打印和权重的保存加载**

```python
# 模型的权重，梯度经过通信在所有进程中都相同，所以权重都相同，但是每个模型对梯度的反向传播、和梯度的清空，以及梯度优化、学习率调整是在每个进程中独立进行，这样减少数据交换可加大分布式速度。
#数据记录和打印只在rank == 0时执行即可
# 创建tensorboard
if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
	print(args)
	print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(tensorboard_dir)
    if os.path.exists(tensorboard_dir) is False:
       os.makedirs(tensorboard_dir)
# 打印epcoh进度
if rank == 0: 
    for epoch in tqdm(iterable=range(start_epoch, epochs),desc="进度：{}/{} ".format(start_epoch,epochs), smoothing=0.9,colour="BLUE"):

    
    
# ... 前向传播，反向传播，优化，学习率调整...
    
if rank == 0
# 统计所有进程中的loss张量，correct张量
	loss = dist.all_reduce(loss, torch.distributed.AVG)
	acc = dist.all_reduce(acc, torch.distributed.AVG)
    tags =["train/lr","train/loss","train/acc"]
    
#保存tensorboard
if rank == 0:
    print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
    tags = ["loss", "accuracy", "learning_rate"]
    for tag,data in zip(tags,[loss,accuracy,shceduler.get_last_lr()]
    	tb_writer.add_scalar(tag, data,epoch)
# 保存模型
	# 保存多种数据以方便继续训练
    state = {
            "start_epoch": epoch,
            "model_state_dict": model.module.state_dict(),# 注意DDP与DP 一样要保存module中的权重
            "Optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        	}
    torch.save(state,os.path.join(log_dir,"model_{}.pth".format(epoch))
	
```

**（8）进程组销毁**

...我就没看到别人的demo写过



**注意：**

1. lauch可以根据参数自动生成5个环境变量，并启动相应进程
2. init_process_group尽量采用环境变量方式初始化（比较简单）
3. device 两种方法一种是改变进程中可见的GPU，一种保证整个代码的device设置
4. **batch_size的设置DDP不同于DP：**
   - **先打包在划分：**Dataloder已经按正常顺序打包数据，DP数据的划分在DP当中，将minibatch划分为多份后分给多个线程控制的GPU
   - **先划分在打包：**DDP的数据在DistributeSampler中已经划分，相应进程取对应数据，再通过Dataloader指定batchsize打包后直接训练
5. model加载到DDP前确保模型已经在local_rank指定的GPU上
6. 模型保存时保存model的module成员变量
7. loss 是DDP返回的loss，其梯度传递由进程组，DDP控制自动控制，我们只需要loss.backward()即可
8. 同步所有进程张量用dist.all_reduce(tensor,op)
9. 我们只在rank = 0上记录，首先是权重相同，其次是loss，acc数据可同步，
10. 同步的BN层会减慢速度，但是会提高精度

11. **单机单卡区别和多级机多卡（分布式）使用上的区别：**
    - 启动方式：必须指定主机node_rank = 0 的 master_addr和master_port(可在launch中指定，也可以再init_process_grooup中指定)
    - 启动次数：必须在其他主机上按同样方式开启，主机数量=nnodes，并注意修改nproc_per_node 和 node_rank 参数
    - 代码上没有区别！

# 问题

虽然准备工作做的非常充分，但是还是遇到了bug

1.  init_process_group()  认为TCP://127.0.0.1:12345 是单机多卡，这导致程序再init process group时阻塞等待其他进程的加入
2. 其他卡被其他程序用着，只要显存够我们就能使用
