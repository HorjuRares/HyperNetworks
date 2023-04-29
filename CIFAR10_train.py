import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR

from ResNetFunctional import ResnetHypernet


transform = T.Compose([T.Resize((320, 320)),
                       T.ToTensor(),
                       T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
train_dataset = CIFAR10(train=True, root='./data', download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=128, num_workers=0)

net = ResnetHypernet()
loss_fn = nn.CrossEntropyLoss()

epochs = 1000
lr = 0.003

optimizer = optim.Adam(params=net.parameters(), lr=lr)
lr_scheduler = PolynomialLR(optimizer=optimizer, total_iters=100000, power=0.9)


for epoch in range(epochs):
    training_loss = 0
    validation_loss = 0

    total = 0
    correct = 0

    # training
    print('Training epoch {}'.format(epoch))

    net.train()

    # before
    before = list(torch.clone(p.data) for p in net.hypernet.parameters())

    for batch_idx, batch_data in enumerate(train_dataloader):
        inputs = batch_data[0]
        labels = batch_data[1]

        outputs = net(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        running_loss = loss.detach().cpu().numpy()
        training_loss += running_loss


        print('{}/{}: {}'.format(batch_idx + 1, len(train_dataloader), running_loss))

    # after
    after = list(p.data for p in net.hyp_net.parameters())

    for idx_p, _ in enumerate(after):
        print(torch.equal(after[idx_p], before[idx_p]))

    print('Training loss: {}'.format(training_loss))

    # validation
    print('Validation loop...')

    net.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(train_dataloader):
            inputs = batch_data[0]
            labels = batch_data[1]

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)

            running_loss = loss.detach().cpu().numpy()
            validation_loss += running_loss

            total += labels.shape[0]
            preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = labels.detach().cpu().numpy()
            correct += np.count_nonzero(preds == labels)

        print('Finished validation')
        print('Validation loss: {}; validation accuracy: {}%'.format(validation_loss, (correct / total) * 100.))
