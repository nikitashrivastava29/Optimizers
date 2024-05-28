import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
#from torch.nn.parallel import DistributedDataParallel as DDP
from fairscale.experimental.nn.data_parallel import SlowMoDistributedDataParallel as SlowMoDDP
from torch.distributed import init_process_group, destroy_process_group
import os
import math
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision import models 
import math
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)



def prepare_dataloader(dataset: Dataset, batch_size: int, rank: int, world_size: int):
    """
    Prepares a DataLoader with DistributedSampler for distributed training.

    Args:
    - dataset (Dataset): The dataset to load.
    - batch_size (int): The batch size per process.
    - rank (int): The rank of the current process.
    - world_size (int): The total number of processes.

    Returns:
    - DataLoader: A DataLoader configured for distributed training.
    """
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    return dataloader

'''
class SmallNet(torch.nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    def name(self):
        return "SmallNet"
'''


def load_train_objs():
    '''transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform) '''
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))  # Normalizing for CIFAR-10
    ])
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10('./data', train=False, transform=transform)
    #model= SmallNet()
    model= models.resnet18()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    return train_set, test_set, model, optimizer

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = SlowMoDDP(model, slowmo_momentum=0.5, nprocs_per_node=2)

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        self.model.zero_grad(set_to_none=True)
        self.model.perform_slowmo(self.optimizer)

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    
    def _calculate_accuracy(self, data_loader):
        total_correct = 0
        total_samples = 0
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for source, targets in data_loader:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source)
                _, predictions = torch.max(output, 1)
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
        accuracy = total_correct / total_samples * 100
        return accuracy
    
    def train(self, max_epochs: int, test_data: DataLoader):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                train_accuracy = self._calculate_accuracy(self.train_data)
                test_accuracy = self._calculate_accuracy(test_data)
                if self.gpu_id == 0:  # Only print from the first process
                    print(f"Epoch {epoch} | Train Accuracy: {train_accuracy:.2f}% | Test Accuracy: {test_accuracy:.2f}%")
                self._save_checkpoint(epoch)

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    train_dataset, test_dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_dataset, batch_size, rank, world_size)
    test_data = prepare_dataloader(test_dataset, batch_size, rank, world_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs, test_data)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
