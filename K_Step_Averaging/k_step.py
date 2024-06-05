# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from torchvision import datasets, transforms
# import numpy as np
# import copy 



# class DistributedKAVG:
#     def __init__(self, model, num_processors, learning_rate, num_epochs, batch_size, k_steps):
#         self.model = model
#         self.num_processors = num_processors
#         self.learning_rate = learning_rate
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.k_steps = k_steps

#     def train(self, train_data):
#         # Creating a DataLoader for each processor
#         data_loaders = [DataLoader(train_data, batch_size=self.batch_size, shuffle=True) for _ in range(self.num_processors)]
        
#         # Initialize model weights for each processor
#         models = [copy.deepcopy(self.model.state_dict()) for _ in range(self.num_processors)]
        
#         # Training loop
#         for epoch in range(self.num_epochs):
#             for p in range(self.num_processors):
#                 local_model = copy.deepcopy(self.model).to('cuda')  # Create a new instance of the model and move it to GPU
#                 local_model.load_state_dict(models[p])  # Load the model state
#                 optimizer = torch.optim.SGD(local_model.parameters(), lr=self.learning_rate)
                
#                 for k in range(self.k_steps):
#                     for data, target in data_loaders[p]:
#                         data, target = data.to('cuda'), target.to('cuda')  # Move data and target to GPU
#                         optimizer.zero_grad()
#                         output = local_model(data)
#                         loss = torch.nn.functional.cross_entropy(output, target)
#                         loss.backward()
#                         optimizer.step()

#                 # Store the updated weights
#                 models[p] = local_model.state_dict()
                
#             # Synchronize weights
#             self.synchronize(models)
    
#     def synchronize(self, models):
#         # Average the parameters
#         with torch.no_grad():
#             for name in models[0]:
#                 average_param = torch.mean(torch.stack([models[p][name] for p in range(self.num_processors)]), dim=0)
#                 for p in range(self.num_processors):
#                     models[p][name].copy_(average_param)

# # Example use-case
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # Load your data here
# print("Dataset Load")
# train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
# model = torch.nn.Sequential(
#     torch.nn.Flatten(),
#     torch.nn.Linear(784, 128),
#     torch.nn.ReLU(),
#     torch.nn.Linear(128, 10)
# )

# print("Calling KAVG!")
# kavg = DistributedKAVG(model, num_processors=4, learning_rate=0.01, num_epochs=5, batch_size=64, k_steps=2)
# kavg.train(train_dataset)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import os
import torch.multiprocessing as mp
import torch.nn.functional as F

class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12365'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def evaluate(model, data_loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def train(rank, world_size, epochs, k_steps):
    setup(rank, world_size)
    
    model = SimpleNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Train dataset and loader
    train_dataset = datasets.MNIST('.', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,))
                                   ]))
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

    # Test dataset and loader
    test_dataset = datasets.MNIST('.', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
                                  ]))
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        local_steps = 0
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)
            output = ddp_model(data)
            loss = F.cross_entropy(output, target)
            
            loss.backward()
            if (local_steps + 1) % k_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                average_parameters(ddp_model.module)
            
            local_steps += 1

        acc = evaluate(ddp_model.module, test_loader, rank)
        print(f"Rank {rank}, Epoch {epoch}, Loss {loss.item()}, Test Accuracy {acc}%")

    cleanup()

def average_parameters(model):
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= dist.get_world_size()

def main():
    world_size = torch.cuda.device_count()

    for i in range(0, 7):
        k= math.pow(2,i)
        print("Result for K=",k) 
        epochs = 10
        mp.spawn(train, args=(world_size, epochs, k), nprocs=world_size, join=True)
        
if __name__ == "__main__":
    main()
