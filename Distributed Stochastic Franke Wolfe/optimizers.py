import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class SFW(torch.optim.Optimizer):
    """Stochastic Frank Wolfe Algorithm
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        learning_rate (float): learning rate between 0.0 and 1.0
        rescale (string or None): Type of learning_rate rescaling. Must be 'diameter', 'gradient' or None
        momentum (float): momentum factor, 0 for no momentum
    """

    def __init__(self, params, learning_rate=0.1, rescale='diameter', momentum=0.9):
        if not (0.0 <= learning_rate <= 1.0):
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between [0, 1].")
        if not (rescale in ['diameter', 'gradient', None]):
            raise ValueError("Rescale type must be either 'diameter', 'gradient' or None.")

        # Parameters
        self.rescale = rescale

        defaults = dict(lr=learning_rate, momentum=momentum)
        super(SFW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # Add momentum
                momentum = group['momentum']
                if momentum > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - momentum)
                        d_p = param_state['momentum_buffer']

                v = constraints[idx].lmo(d_p)  # LMO optimal solution

                if self.rescale == 'diameter':
                    # Rescale lr by diameter
                    factor = 1. / constraints[idx].get_diameter()
                elif self.rescale == 'gradient':
                    # Rescale lr by gradient
                    factor = torch.norm(d_p, p=2) / torch.norm(p - v, p=2)
                else:
                    # No rescaling
                    factor = 1

                lr = max(0.0, min(factor * group['lr'], 1.0))  # Clamp between [0, 1]

                p.mul_(1 - lr)
                p.add_(v, alpha=lr)
                idx += 1
        return loss


class AdaGradSFW(torch.optim.Optimizer):
    """AdaGrad Stochastic Frank-Wolfe algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        inner_steps (integer, optional): number of inner iterations (default: 2)
        learning_rate (float, optional): learning rate (default: 1e-2)
        delta (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        momentum (float, optional): momentum factor
    """

    def __init__(self, params, inner_steps=2, learning_rate=1e-2, delta=1e-8, momentum=0.9):
        if not 0.0 <= learning_rate:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("Momentum must be between [0, 1].")
        if not 0.0 <= delta:
            raise ValueError("Invalid delta value: {}".format(delta))
        if not int(inner_steps) == inner_steps and not 0.0 <= inner_steps:
            raise ValueError("Number of inner iterations needs to be a positive integer: {}".format(inner_steps))

        self.K = inner_steps

        defaults = dict(lr=learning_rate, delta=delta, momentum=momentum)
        super(AdaGradSFW, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                param_state = self.state[p]
                momentum = group['momentum']

                if momentum > 0:
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = d_p.detach().clone()
                    else:
                        param_state['momentum_buffer'].mul_(momentum).add_(d_p, alpha=1 - momentum)
                        d_p = param_state['momentum_buffer']

                param_state['sum'].addcmul_(d_p, d_p, value=1)  # Holds the cumulative sum
                H = torch.sqrt(param_state['sum']).add(group['delta'])

                y = p.detach().clone()
                for _ in range(self.K):
                    d_Q = d_p.addcmul(H, y - p, value=1. / group['lr'])
                    y_v_diff = y - constraints[idx].lmo(d_Q)
                    gamma = group['lr'] * torch.div(torch.sum(torch.mul(d_Q, y_v_diff)),
                                                    torch.sum(H.mul(torch.mul(y_v_diff, y_v_diff))))
                    gamma = max(0.0, min(gamma, 1.0))  # Clamp between [0, 1]

                    y.add_(y_v_diff, alpha=-gamma)  # -gamma needed as we want to add v-y, not y-v
                p.copy_(y)
                idx += 1
        return loss


class SGD(torch.optim.Optimizer):
    """Modified SGD which allows projection via Constraint class"""

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if momentum is None:
            momentum = 0
        if weight_decay is None:
            weight_decay = 0
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between [0, 1].")
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, constraints, closure=None):
        """Performs a single optimization step.
        Args:
            constraints (iterable): list of constraints, where each is an initialization of Constraint subclasses
            parameter groups
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        idx = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

                # Project if necessary
                if not constraints[idx].is_unconstrained():
                    p.copy_(constraints[idx].euclidean_project(p))
                idx += 1

        return loss

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

def load_train_objs():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    '''model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28*28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)  # 10 output classes for MNIST digits
    )'''
    model= SmallNet()
    optimizer = SFW(model.parameters(), learning_rate=0.1, rescale='diameter', momentum=0.9)
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
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

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
