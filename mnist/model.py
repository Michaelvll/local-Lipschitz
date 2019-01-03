import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 20, 5, 1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2),
                                  nn.Conv2d(20, 50, 5, 1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2, 2),
                                  )
        self.hidden2label = nn.Sequential(nn.Linear(4*4*50, 500),
                                          nn.ReLU(),
                                          nn.Linear(500, 10),
                                          )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 4*4*50)
        x = self.hidden2label(x)
        return self.softmax(x), x


def get_loss(args, model, data, target):
    data.requires_grad_(True)
    prob, scores = model(data)
    gradient = torch.autograd.grad(
        outputs=scores, inputs=data, grad_outputs=torch.ones_like(scores, device=data.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_loss = gradient.norm(2) * args.lip_lambda
    target_loss = F.nll_loss(prob, target)
    return target_loss + gradient_loss, scores


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), ncols=50, total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss, _ = get_loss(args, model, data, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() / len(train_loader)

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # output, _ = model(data)
        # sum up batch loss
        # test_loss += F.nll_loss(output, target, reduction='sum').item()
        loss, scores = get_loss(args, model, data, target)
        test_loss += loss.item()
        # get the index of the max log-probability
        pred = scores.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
