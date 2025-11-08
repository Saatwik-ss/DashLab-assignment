# fedavg_cifar.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import copy



# ----------------------------
class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.num_classes = num_classes

        self._init_fc_layers()  # build linear layers dynamically


    def _init_fc_layers(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32)          # CIFAR-10 image size
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = torch.flatten(x, 1)
            flat_size = x.numel()                   # number of features
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ----------------------------
def local_train(model, dataloader, epochs, lr, device):
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for i in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
    return model.state_dict()


# ----------------------------
def fedavg(global_w, local_ws, sizes):
    total = sum(sizes)
    new_state = copy.deepcopy(global_w)
    for k in global_w.keys():
        new_state[k] = sum(local_ws[i][k] * (sizes[i] / total)
                           for i in range(len(local_ws)))
    return new_state

# ----------------------------
def random_partition(dataset, num_clients):
    lengths = [len(dataset)//num_clients] * num_clients
    lengths[-1] += len(dataset) - sum(lengths)
    subsets = random_split(dataset, lengths)
    return subsets

# ----------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    num_clients = 5
    num_rounds = 1
    local_epochs = 1
    batch_size = 64
    lr = 0.01

    # CIFAR-10 setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    client_sets = random_partition(trainset, num_clients)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)

    # Global model
    global_model = CNN().to(device)
    global_weights = global_model.state_dict()

    for r in range(num_rounds):
        print(f"Global Round {r+1}/{num_rounds}")
        local_ws, sizes = [], []

        for cid in range(num_clients):
            print(f": Client {cid+1}")
            local_model = CNN().to(device)
            local_model.load_state_dict(global_weights)

            loader = DataLoader(client_sets[cid], batch_size=batch_size, shuffle=True)
            updated = local_train(local_model, loader, local_epochs, lr, device)
            local_ws.append(copy.deepcopy(updated))
            sizes.append(len(loader.dataset))

        global_weights = fedavg(global_weights, local_ws, sizes)
        global_model.load_state_dict(global_weights)

        # Evaluate global model
        global_model.eval()
        total, correct, loss_sum = 0, 0, 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = global_model(x)
                loss_sum += criterion(out, y).item() * x.size(0)
                pred = out.argmax(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total
        print(f"Global Test Accuracy: {acc:.2f}% \n Loss: {loss_sum/total:.4f}")

if __name__ == "__main__":
    main()
