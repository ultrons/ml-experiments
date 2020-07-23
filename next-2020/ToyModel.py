import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms


class ToyModel (nn.Module):
    """ Toy Classifier """
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.mp1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1440, 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.mp1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.Softmax()(x)
        return x

FLAGS = {
    'batch_size': 32,
    'world_size': 4,
    'epochs': 2,
    'log_steps': 10
}

def train(rank, FLAGS):
    dist.init_process_group(backend='nccl', world_size=FLAGS['world_size'],
                            rank=rank)
    model = ToyModel()
    torch.cuda.set_device(rank)
    model.cuda(rank)
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    transform = transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = torchvision.datasets.MNIST('/tmp/', train=True, download=True,
                             transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=FLAGS['world_size'], rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                             batch_size=FLAGS['batch_size'],
                                                   shuffle=False,
                                                   num_worker=0,
                                                   sampler=train_sampler)
    for epoch in range(FLAGS['epochs']):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not i % FLAGS['log_steps']:
                print('Epoch: {}/{}, Loss:{}'.format(epoch + 1, FLAGS['epochs'],
                                                     loss.item()))

mp.spawn(train, nprocs=FLAGS['world_size'], args=(FLAGS,))


model = ToyModel()
from torchsummary import summary
summary(model, (1, 28, 28))
