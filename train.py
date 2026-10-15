# Code is based on the PyTorch MNIST example: https://github.com/pytorch/examples/tree/main/mnist

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from checkpoint_utils import TrainStateSaver
import writer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set performance: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    writer.put_scalar("test_loss", test_loss, step=epoch)
    writer.put_scalar("test_accuracy", 100. * correct / len(test_loader.dataset), step=epoch)


def main():
    parser = argparse.ArgumentParser(description='Train a CNN on MNIST')
    parser.add_argument('--output', help="Path to folder for output data.", required=True)

    parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train. Default: 100000.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size. Default: 64.')
    
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers. Default: 4.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed. Default: 1.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA training.')

    parser.add_argument('--no_tensorboard', action='store_true', help='Do not log to tensorboard. Default: do log.')
    parser.add_argument('--wandb', action='store_true', help='Log to weights and biases. Default: Do not log.')
    parser.add_argument('--exp_id', type=str, help="Experiment ID. Used for wandb logging.")
    parser.add_argument('--comment', type=str, help="Comment for the experiment. Used for wandb logging.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    finished_epochs = 0

    # Set up output directories and logging:
    checkpoint_dir = os.path.join(args.output, "checkpoints")
    wandb_logs_dir = os.path.join(args.output, "wandb_logs")
    tensorboard_logs_dir = os.path.join(args.output, "tensorboard_logs")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(wandb_logs_dir, exist_ok=True)
    os.makedirs(tensorboard_logs_dir, exist_ok=True)

    print(f"Output directory: {args.output}\n")

    writer.setup_writers(log_tensorboard=not args.no_tensorboard, 
                         log_wandb=args.wandb, 
                         tensorboard_logs_dir=tensorboard_logs_dir, 
                         wandb_logs_dir=wandb_logs_dir,
                         exp_id=args.exp_id,
                         comment=args.comment,)

    # Set up data:
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
    dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000, num_workers=args.num_workers, pin_memory=True)

    # Set up model:
    model = Net().to(device)

    # Set up optimizer and scheduler:
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    # Set up checkpoint saver:
    saver = TrainStateSaver(model=model, optim=optimizer, scheduler=scheduler, base_path=args.output, max_to_keep=2)

    # Resume training from latest checkpoint if available:
    all_checkpoints = sorted(saver.get_checkpoints(include_epoch=True))
    if len(all_checkpoints) > 0:
        finished_epochs, newest_checkpoint = all_checkpoints[-1]
        print(f"Restoring training state from checkpoint {newest_checkpoint}.")
        saver.load(full_path=newest_checkpoint)

    # Training loop:
    for epoch in range(finished_epochs + 1, args.epochs + 1):
        writer.put_scalar("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        writer.write_out_storage()

        scheduler.step()

        save_path = saver.save(epoch=epoch)
        print(f"Saved checkpoint to {save_path}.\n")


if __name__ == '__main__':
    main()