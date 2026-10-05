#!/usr/bin/env python
# coding: utf-8

# # DermaMNIST + HDFS + Spark + PyTorch DDP
# 
# In this notebook, we extend the Spark + HDFS DermaMNIST pipeline by using PyTorch Distributed Data Parallel (DDP) for CNN training.
# 
# The dataset is stored as individual PNG images in HDFS and loaded through Spark. The reconstructed data is then passed to PyTorch for preprocessing and distributed CNN training.
# 
# With DDP, each process maintains its own copy of the CNN and processes a different subset of the training data. Gradients are synchronised between processes during training.
# 
# The CNN architecture, preprocessing, data augmentation, class-imbalance handling, optimizer, number of epochs, and evaluation procedure are kept consistent with the local and Spark + HDFS experiments. This allows us to compare the effect of distributed training while keeping the underlying machine-learning experiment controlled.

# In[36]:


import io
import os
import sys
import time
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PySpark:", pyspark.__version__)
print("PySpark location:", pyspark.__file__)


# ### 1. Reproducibility
# 
# We keep it identical to the baseline.

# In[4]:


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

g = torch.Generator()
g.manual_seed(seed)


# ### 2. Helper functions
# 
# 

# In[28]:


# convert the PNG bytes to NumPy
def decode_image(image_bytes):
    return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

def sort_by_original_index(rows):
    return sorted(
        rows,
        key=lambda row: int(row.path.rsplit("/", 1)[-1].replace(".png", ""))
    )


# #### 2.1. Dataset class - PyTorch dataset adapter
# 
# The Spark pipeline produces the image data as NumPy arrays, whereas PyTorch's DataLoader expects a dataset implementing the standard Dataset interface. 
# 
# DermaMNISTArrayDataset class provides this interface by retrieving an image and label by index, converting the NumPy image to a PIL image for compatibility with torchvision transformations, applying the same preprocessing as the baseline, and returning the image and label as PyTorch tensors.

# In[29]:


class DermaMNISTArrayDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # convert NumPy image back to PIL for torchvision transforms
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        # keep label shape consistent with the baseline
        return image, torch.tensor([label], dtype=torch.long)


# #### 2.2. CNN architecture definition
# 
# We use the same CNN architecture as in the baseline.

# In[30]:


class CNN_deep(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# #### 2.3. Distributed Training Setup
# 
# The Spark/HDFS data-processing pipeline is completed before we initialize the distributed training.
# 
# PyTorch Distributed Data Parallel (DDP) is then initialized so that multiple CPU processes can train the CNN collaboratively. Each process maintains its own model replica and processes a separate portion of the training data, while gradients are synchronised across processes during backpropagation.

# In[33]:


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


# #### 2.4. DDP-compatible Weighted Sampler
# 
# We need a small DDP-compatible replacement for `WeightedRandomSampler` that preserves exact square-root inverse-frequency weighting.

# In[34]:


class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(self, weights, num_replicas, rank, seed=42):
        self.weights = weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.num_samples = len(weights) // num_replicas
        self.total_size = self.num_samples * num_replicas

        self.epoch = 0

    def __iter__(self):
        # generate one weighted sample list for the whole DDP job
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g)

        # give each process its own subset
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# ### 3. Main distributed experiment

# In[39]:


def main():
    pipeline_start = time.perf_counter()

    # initialise the distributed environment
    rank, local_rank, world_size = setup_distributed()

    # only rank 0 prints the main experiment information
    if rank == 0:
        print(f"DDP world size: {world_size}")
        print("Backend: gloo")
        print("Device: CPU")

    # Spark + HDFS data pipeline
    # only rank 0 performs the Spark/HDFS data retrieval (this avoids starting one Spark application per DDP process)
    if rank == 0:
        # start Spark
        spark_start = time.perf_counter()

        spark = (
            SparkSession.builder
            .appName("DermaMNIST_Spark_HDFS_DDP")
            .master("local[*]")
            .getOrCreate()
        )

        spark_startup_time = time.perf_counter() - spark_start

        # read the image files from HDFS
        hdfs_dataset_path = "hdfs:///ca1/dermamnist_spark"

        spark_read_start = time.perf_counter()

        images_df = (spark.read.format("binaryFile").option("recursiveFileLookup", "true").load(hdfs_dataset_path))

        # extract split and class from the HDFS path
        images_df = (
            images_df
            .withColumn(
                "split",
                regexp_extract("path", r"/(train|val|test)/", 1)
            )
            .withColumn(
                "class_id",
                regexp_extract("path", r"/(?:train|val|test)/(\d+)/", 1).cast("int")
            )
        )

        image_count = images_df.count()

        spark_read_time = time.perf_counter() - spark_read_start

        # transfer Spark data to Python
        spark_to_python_start = time.perf_counter()

        image_rows = (images_df.select("path", "content", "split", "class_id").collect())

        spark_to_python_time = (time.perf_counter() - spark_to_python_start)

        print(f"Records transferred: {len(image_rows)}")

        # convert PNG bytes to NumPy arrays
        conversion_start = time.perf_counter()

        train_rows = sort_by_original_index([row for row in image_rows if row.split == "train"])
        val_rows = sort_by_original_index([row for row in image_rows if row.split == "val"])
        test_rows = sort_by_original_index([row for row in image_rows if row.split == "test"])

        train_images = np.stack([decode_image(row.content) for row in train_rows])
        train_labels = np.array([row.class_id for row in train_rows], dtype=np.int64)

        val_images = np.stack([decode_image(row.content) for row in val_rows])
        val_labels = np.array([row.class_id for row in val_rows], dtype=np.int64)

        test_images = np.stack([decode_image(row.content) for row in test_rows])
        test_labels = np.array([row.class_id for row in test_rows], dtype=np.int64)

        conversion_time = time.perf_counter() - conversion_start

        print(f"Train: {train_images.shape} {train_labels.shape}")
        print(f"Val:   {val_images.shape} {val_labels.shape}")
        print(f"Test:  {test_images.shape} {test_labels.shape}")

        # calculate normalization statistics
        train_images_float = (train_images.astype(np.float32) / 255.0)

        mean_spark = train_images_float.mean(axis=(0, 1, 2))
        std_spark = train_images_float.std(axis=(0, 1, 2))

        print("Training mean:", mean_spark)
        print("Training std:", std_spark)

        # Spark is no longer needed after the HDFS data has been transferred to Python
        spark.stop()


    # broadcast dataset from rank 0 to all DDP processes
    # broadcast dataset sizes
    if rank == 0:
        dataset_sizes = torch.tensor(
            [len(train_images), len(val_images), len(test_images)], dtype=torch.long
        )
    else:
        dataset_sizes = torch.zeros(3, dtype=torch.long)

    dist.broadcast(dataset_sizes, src=0)
    train_size = dataset_sizes[0].item()
    val_size = dataset_sizes[1].item()
    test_size = dataset_sizes[2].item()

    # broadcast images and labels
    broadcast_start = time.perf_counter()

    if rank == 0:
        train_images_tensor = torch.from_numpy(train_images)
        train_labels_tensor = torch.from_numpy(train_labels)

        val_images_tensor = torch.from_numpy(val_images)
        val_labels_tensor = torch.from_numpy(val_labels)

        test_images_tensor = torch.from_numpy(test_images)
        test_labels_tensor = torch.from_numpy(test_labels)

        mean_tensor = torch.tensor(mean_spark, dtype=torch.float32)
        std_tensor = torch.tensor(std_spark, dtype=torch.float32)

    else:
        train_images_tensor = torch.empty((train_size, 64, 64, 3), dtype=torch.uint8)
        train_labels_tensor = torch.empty((train_size,), dtype=torch.long)

        val_images_tensor = torch.empty((val_size, 64, 64, 3), dtype=torch.uint8)
        val_labels_tensor = torch.empty((val_size,), dtype=torch.long)

        test_images_tensor = torch.empty((test_size, 64, 64, 3), dtype=torch.uint8)
        test_labels_tensor = torch.empty((test_size,), dtype=torch.long)

        mean_tensor = torch.empty(3, dtype=torch.float32)
        std_tensor = torch.empty(3, dtype=torch.float32)

    # broadcast the data from rank 0 to every other process
    dist.broadcast(train_images_tensor, src=0)
    dist.broadcast(train_labels_tensor, src=0)

    dist.broadcast(val_images_tensor, src=0)
    dist.broadcast(val_labels_tensor, src=0)

    dist.broadcast(test_images_tensor, src=0)
    dist.broadcast(test_labels_tensor, src=0)

    dist.broadcast(mean_tensor, src=0)
    dist.broadcast(std_tensor, src=0)

    broadcast_time = (time.perf_counter() - broadcast_start)

    # convert back to NumPy arrays for the Dataset class
    train_images = train_images_tensor.numpy()
    train_labels = train_labels_tensor.numpy()

    val_images = val_images_tensor.numpy()
    val_labels = val_labels_tensor.numpy()

    test_images = test_images_tensor.numpy()
    test_labels = test_labels_tensor.numpy()

    mean_spark = mean_tensor.numpy()
    std_spark = std_tensor.numpy()

    # make sure every process has finished receiving the data
    dist.barrier()

    # prepare the data for PyTorch
    pytorch_preparation_start = time.perf_counter()

    # same transforms as the baseline
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_spark.tolist(), std=std_spark.tolist())
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_spark.tolist(), std=std_spark.tolist())
    ])

    train_dataset = DermaMNISTArrayDataset(train_images, train_labels, transform=transform_train)
    val_dataset = DermaMNISTArrayDataset(val_images, val_labels, transform=transform_eval)
    test_dataset = DermaMNISTArrayDataset(test_images, test_labels, transform=transform_eval)

    # create DDP-compatible training sampler
    class_counts = np.bincount(train_labels)

    class_sampling_weights = 1 / np.sqrt(class_counts)

    sample_weights = class_sampling_weights[train_labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    train_sampler = DistributedWeightedSampler(
        weights=sample_weights,
        num_replicas=world_size,
        rank=rank,
        seed=seed
    )

    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    pytorch_preparation_time = (time.perf_counter() - pytorch_preparation_start)

    # initialise the model
    model_start = time.perf_counter()

    model = CNN_deep(num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # wrap the model with DistributedDataParallel
    model = DDP(model)

    model_initialization_time = (time.perf_counter() - model_start)


    # distributed training
    num_epochs = 50

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "epoch_time": []
    }

    best_macro_f1 = -1
    best_epoch = 0
    best_model_state = None

    # synchronize before starting the measured training phase
    dist.barrier()

    training_start = time.perf_counter()

    for epoch in range(num_epochs):
        # make sure all processes start the epoch together
        dist.barrier()

        epoch_start = time.perf_counter()

        # change the sampler's shuffle order for this epoch
        train_sampler.set_epoch(epoch)

        # training
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            labels = labels.squeeze(1)

            optimizer.zero_grad()   # clear gradients from the previous batch

            outputs = model(images)   # perform the forward pass
            loss = criterion(outputs, labels)   # calculate the classification loss
            loss.backward()   # calculate gradients
            optimizer.step()   # update model parameters

            # accumulate training statistics
            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # aggregate training metrics across DDP processes
        train_metrics = torch.tensor([running_loss, correct, total], dtype=torch.float64)
        dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)
        train_loss = train_metrics[0].item() / train_metrics[2].item()
        train_acc = train_metrics[1].item() / train_metrics[2].item()

        # validation
        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        all_val_predictions = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.squeeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += (loss.item() * images.size(0))
                predictions = outputs.argmax(dim=1)
                val_correct += ((predictions == labels).sum().item())
                val_total += labels.size(0)

                all_val_predictions.extend(predictions.numpy())
                all_val_labels.extend(labels.numpy())

        val_loss = (val_running_loss / val_total)
        val_acc = (val_correct / val_total)
        val_macro_f1 = f1_score(all_val_labels, all_val_predictions, average="macro", zero_division=0)

        epoch_time = time.perf_counter() - epoch_start

        # only rank 0 records and prints the results
        if rank == 0:

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_macro_f1"].append(val_macro_f1)
            history["epoch_time"].append(epoch_time)

            if val_macro_f1 > best_macro_f1:
                best_macro_f1 = val_macro_f1
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.module.state_dict())

            print(
                f"Epoch {epoch + 1:2d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val Macro-F1: {val_macro_f1:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

    training_time = time.perf_counter() - training_start

    if rank == 0:

        print(f"\nTraining time: {training_time:.2f}s")
        print(f"Best validation macro-F1: {best_macro_f1:.4f}")
        print(f"Best epoch: {best_epoch}")


    # restore best model and evaluate test set
    dist.barrier()

    # only rank 0 performs the final test evaluation
    if rank == 0:
        # restore the best checkpoint kept in memory
        model.module.load_state_dict(best_model_state)

        test_start = time.perf_counter()

        model.eval()

        test_running_loss = 0.0
        test_correct = 0
        test_total = 0

        all_test_labels = []
        all_test_predictions = []

        with torch.no_grad():
            for images, labels in test_loader:
                labels = labels.squeeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += (loss.item() * images.size(0))
                predictions = outputs.argmax(dim=1)
                test_correct += ((predictions == labels).sum().item())
                test_total += labels.size(0)
                all_test_labels.extend(labels.numpy())
                all_test_predictions.extend(predictions.numpy())

        test_time = (time.perf_counter() - test_start)

        test_loss = (test_running_loss / test_total)
        test_acc = (test_correct / test_total)
        test_macro_f1 = f1_score(all_test_labels, all_test_predictions, average="macro", zero_division=0)

        print(f"Test evaluation time: {test_time:.2f}s")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test macro-F1: {test_macro_f1:.4f}")

        # classification report
        print()
        print("\nClassification report:")
        print(
            classification_report(all_test_labels, all_test_predictions, labels=list(range(7)), zero_division=0)
        )

        # confusion matrix
        cm = confusion_matrix(all_test_labels, all_test_predictions, labels=list(range(7)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(7)))

        fig, ax = plt.subplots(figsize=(7, 7))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title("DermaMNIST DDP Confusion Matrix")
        fig.tight_layout()
        fig.savefig("dermamnist_ddp_confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        print("Confusion matrix saved to: dermamnist_ddp_confusion_matrix.png")

    else:
        test_time = 0.0

    # make sure rank 0 finishes testing before any process destroys the process group
    dist.barrier()

    # final timing summary
    if rank == 0:

        average_epoch_time = np.mean(history["epoch_time"])

        # sum the measured pipeline stages
        measured_pipeline_time = (spark_startup_time + spark_read_time + spark_to_python_time + conversion_time
            + broadcast_time + pytorch_preparation_time + model_initialization_time + training_time + test_time)

        print()
        print(f"Spark startup: {spark_startup_time:.2f}s")
        print(f"Spark data read: {spark_read_time:.2f}s")
        print(f"Spark → Python transfer: {spark_to_python_time:.2f}s")
        print(f"Image conversion: {conversion_time:.2f}s")
        print(f"DDP data broadcast: {broadcast_time:.2f}s")
        print(f"PyTorch data preparation: {pytorch_preparation_time:.2f}s")
        print(f"Model initialization: {model_initialization_time:.2f}s")
        print(f"Training: {training_time:.2f}s")
        print(f"Average epoch: {average_epoch_time:.2f}s")
        print(f"Test evaluation: {test_time:.2f}s")
        print()
        print(f"Measured pipeline time: {measured_pipeline_time:.2f}s")

    # finish the distributed training session
    cleanup_distributed()


# ### 4. Launch the training

# In[ ]:


if __name__ == "__main__":
    main()


# ## Actual execution is in terminal
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
# 
# Since our file is `03_DermaMNIST_Spark_HDFS_DDP.ipynb`, we first need to convert it to a temporary Python script with `jupyter nbconvert --to script 03_MNIST_DDP.ipynb`. This should create `03_DermaMNIST_Spark_HDFS_DDP.py`.
# 
# Then, we launch **2 DDP processes** with `torchrun --standalone --nproc-per-node=2 03_DermaMNIST_Spark_HDFS_DDP.py`

# In[23]:





# 

# ## HDFS + Spark + DDP pipeline conclusion
# 
# The Spark/HDFS pipeline is reproducing the baseline ML experiment correctly: **75.86% test accuracy** and a **macro-F1 of 0.6198**.
# 
# There is an interesting difference in timing:
# - Spark/HDFS measured pipeline: 1550.49 s (~25.84 min)
# - Local baseline: 1564.58 s (~26.08 min)
# - 
# That's only about 14.09 seconds faster overall, despite Spark adding ~14 seconds of data-pipeline overhead. And the training itself is 26.71 seconds faster. But this difference isn't evidence that Spark made training intrinsically faster. It's more likely normal runtime variation between two CPU runs.
# 
# Moving the dataset access to HDFS/Spark did not change model performance, while the additional Spark/HDFS data-processing overhead was relatively small compared with the 50-epoch CNN training time.

# In[ ]:




