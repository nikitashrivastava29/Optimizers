import argparse
from typing import Tuple
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def create_data_loaders(rank: int,
                        world_size: int,
                        batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_loc = './mnist_data'

    train_dataset = datasets.MNIST(dataset_loc,
                                   download=True,
                                   train=True,
                                   transform=transform)
    sampler = DistributedSampler(train_dataset,
                                 num_replicas=world_size,  # Number of GPUs
                                 rank=rank,  # GPU where process is running
                                 shuffle=True,  # Shuffling is done by Sampler
                                 seed=42)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False,  # This is mandatory to set this to False here, shuffling is done by Sampler
                              num_workers=2,  # Number of GPUs in current server
                              sampler=sampler,
                              pin_memory=True)

    test_dataset = datasets.MNIST(dataset_loc,
                                  download=True,
                                  train=False,
                                  transform=transform)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2, # Number of GPUs in current server
                             pin_memory=True)

    return train_loader, test_loader

class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

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

def create_model():
    # create model architecture
    model = SmallNet()
    return model

def main(rank: int,
         epochs: int,
         model: nn.Module,
         train_loader: DataLoader,
         test_loader: DataLoader) -> nn.Module:
    device = torch.device(f'cuda:{rank}')
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # initialize optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # train the model
    for i in range(epochs):
        model.train()
        train_loader.sampler.set_epoch(i)

        epoch_loss = 0
        total_batches = len(train_loader)

        # train the model for one epoch
        pbar = tqdm(train_loader)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x = x.view(x.shape[0], -1)
            optimizer.zero_grad()
            y_hat = model(x)
            batch_loss = loss_fn(y_hat, y)
            batch_loss.backward()
            optimizer.step()

            # Synchronize the loss across all processes
            reduced_loss = torch.tensor(batch_loss.item()).to(device)
            torch.distributed.reduce(reduced_loss, dst=0)
            batch_loss_scalar = reduced_loss.item() / world_size

            epoch_loss += batch_loss_scalar
            pbar.set_description(f'training batch_loss={batch_loss_scalar:.4f}')

        # calculate validation loss
        with torch.no_grad():
            model.eval()
            val_loss = 0

            # Synchronize the loss across all processes
            for x, y in tqdm(test_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x = x.view(x.shape[0], -1)
                y_hat = model(x)
                batch_loss = loss_fn(y_hat, y)

                reduced_loss = torch.tensor(batch_loss.item()).to(device)
                torch.distributed.reduce(reduced_loss, dst=0)
                batch_loss_scalar = reduced_loss.item() / world_size

                val_loss += batch_loss_scalar

            val_loss /= len(test_loader)
            print(f"Epoch={i}, train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    batch_size = 128
    epochs = 10

    rank = args.local_rank
    world_size = torch.cuda.device_count()

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=Backend.NCCL,
                                         init_method='env://')

    train_loader, test_loader = create_data_loaders(rank, world_size, batch_size)
    model = main(rank=rank,
                 epochs=epochs,
                 model=create_model(),
                 train_loader=train_loader,
                 test_loader=test_loader)

    if rank == 0:
        torch.save(model.state_dict(), 'model.pt')