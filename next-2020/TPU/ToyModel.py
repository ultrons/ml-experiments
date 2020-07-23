import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.debug.metrics as met


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
        x = nn.Softmax(dim=-1)(x)
        return x

FLAGS = {
    'batch_size': 32,
    'world_size': 8,
    'epochs': 2,
    'log_steps': 10,
    'metrics_debug': True
}
SERIAL_EXEC = xmp.MpSerialExecutor()
WRAPPED_MODEL = xmp.MpModelWrapper(ToyModel())

def train(rank, FLAGS):
    print("Starting train method on rank: {}".format(rank))
    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    def get_dataset():
        transform = transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    
        train_dataset = torchvision.datasets.MNIST('/tmp/', train=True, download=True,
                                 transform=transform)
        return train_dataset
    
    train_dataset = SERIAL_EXEC.run(get_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=FLAGS['world_size'], rank=rank)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                             batch_size=FLAGS['batch_size'],
                                                   shuffle=False,
                                                   num_workers=0,
                                                   sampler=train_sampler)
    para_loader = pl.ParallelLoader(train_loader, [device])
    for epoch in range(FLAGS['epochs']):
        for i, (images, labels) in enumerate(para_loader.per_device_loader(device)):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)
            if not i % FLAGS['log_steps']:
                xm.master_print('Epoch: {}/{}, Loss:{}'.format(epoch + 1, FLAGS['epochs'],
                                                     loss.item()))
        if FLAGS['metrics_debug']: 
            xm.master_print(met.metrics_report())

if __name__ == '__main__': 
    xmp.spawn(train, nprocs=FLAGS['world_size'], args=(FLAGS,), start_method='fork')

