#!/usr/bin/env python
# coding: utf-8

# # MNIST PyTorch DDP
# 
# In this notebook we want to test whether PyTorch DDP can successfully train our CNN on this VM using multiple CPU processes.
# 
# With DDP, each process gets its own copy of the model and trains on a different subset of the training data. Gradients are synchronised between processes.
# 
# For the sanity check, we don't need HDFS or Spark involved yet. If DDP works independently, we know any later problem combining DDP with Spark is a separate issue.

# In[12]:


import os
import time

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CPU threads:", torch.get_num_threads())


# ## Same CNN Architecture

# In[8]:


import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ### Distributed setup

# In[14]:


def setup_distributed():
    # read the process information provided by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # initialise communication between the processes
    dist.init_process_group(
        backend="gloo",   # because Gloo is for CPUs
        rank=rank,
        world_size=world_size
    )

    return rank, local_rank, world_size

def cleanup_distributed():
    # close the distributed process group after training
    dist.destroy_process_group()


# ### Main training function

# In[15]:


def main():
    # initialise the distributed environment
    rank, local_rank, world_size = setup_distributed()

    # only rank 0 prints the main experiment information
    if rank == 0:
        print(f"World size: {world_size}")
        print("Backend: gloo")
        print("Device: CPU")

    # use the same MNIST preprocessing as the baseline experiment
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # divide the training dataset between the DDP processes
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # each process loads only its assigned part of the training data
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        sampler=train_sampler
    )

    # each process can evaluate the complete test dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )

    # create an independent copy of the CNN for this process
    model = MNISTCNN()

    # wrap the model so that gradients are synchronised between processes
    model = DDP(model)

    # use the same loss function and optimizer as the baseline
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train for the same number of epochs as the baseline
    num_epochs = 5

    if rank == 0:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Processes: {world_size}")

    total_start = time.time()

    for epoch in range(num_epochs):

        # change the sampler's shuffle order for this epoch
        train_sampler.set_epoch(epoch)

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start = time.time()

        for images, labels in train_loader:
            optimizer.zero_grad()   # clear gradients from the previous batch

            outputs = model(images)   # perform the forward pass
            loss = criterion(outputs, labels)   # calculate the classification loss
            loss.backward()   # calculate gradients
            optimizer.step()   # update model parameters

            # accumulate training statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # calculate local training metrics
        local_loss = running_loss / total
        local_accuracy = correct / total

        # combine statistics from all processes
        loss_tensor = torch.tensor([running_loss, total], dtype=torch.float64)
        correct_tensor = torch.tensor([correct], dtype=torch.float64)

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)

        global_loss = loss_tensor[0].item() / loss_tensor[1].item()
        global_accuracy = correct_tensor[0].item() / loss_tensor[1].item()

        epoch_time = time.time() - epoch_start

        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train loss: {global_loss:.4f} | "
                f"Train acc: {global_accuracy:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

    total_time = time.time() - total_start

    if rank == 0:
        print(f"\nTotal training time: {total_time:.2f} seconds")
        print(f"Average epoch time: {total_time / num_epochs:.2f} seconds")

    # finish the distributed training session
    cleanup_distributed()


# ### Launch the training

# In[ ]:


if __name__ == "__main__":
    main()


# ### Actual execution is in terminal
# 
# The next step is to execute this notebook through `nbconvert` and launch the resulting Python process with `torchrun --standalone --nproc-per-node=2 my_script.py`
# 
# There are two separate concepts here:
# - number of DDP processes → `--nproc-per-node`
# - number of CPU threads available to each process → controlled separately by PyTorch/environment
# 
# For our first test, we use 2 DDP processes on our VM's 8 CPUs.
# 
# So:
# 
# 8 VM CPUs
#    ↓
# 2 DDP processes
#    ↓
# each process can use multiple CPU threads
# 
# We are not saying `--nproc-per-node=8`. That would create 8 copies of the CNN, which is unnecessary for the initial sanity check and could actually make CPU performance worse due to contention.
# 
# Since our file is `03_MNIST_DDP.ipynb`, we first need to convert it to a temporary Python script with `jupyter nbconvert --to script 03_MNIST_DDP.ipynb`. This should create `03_MNIST_DDP.py`.
# 
# Then, we launch **2 DDP processes** with `torchrun --standalone --nproc-per-node=2 03_MNIST_DDP.py`

# In[ ]:





# In[ ]:




