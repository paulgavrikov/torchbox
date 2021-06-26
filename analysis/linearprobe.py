import torch
from torch.nn import Flatten, LazyLinear, Softmax


class LinearProbe:

    def __init__(self, device="cpu", verbose=False):
        self.device = device
        self.max_epochs = 10
        self.verbose = verbose

    rep = None

    def _hook(model, inp, out):
        rep = out

    def fit_all(self):
        pass

    def _fit_probe(self, model, probe, train_loader, optimizer, criterion):
        for epoch in range(self.max_epochs):
            # acc = torchmetrics.Accuracy().to(self.device)
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with torch.no_grad():
                     _ = model(inputs)
                optimizer.zero_grad()
                outputs = probe(rep)
                loss = criterion(outputs, targets)

                # acc(outputs, targets)
                # total_loss += loss.item()

                loss.backward()
                optimizer.step()
            if self.verbose:
                print(f"epoch: {epoch}/{self.max_epochs} train-loss: {total_loss / len(train_loader)}")

    def fit_layer(model, trainloader, valloader, classes, layer, epochs=10, device=-1,
                  criterion=torch.nn.CrossEntropyLoss(), optimizer=torch.optim.Adam) -> float:
        global rep

        model.eval()

        # def probe
        probe = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyLinear(classes),
            torch.nn.Softmax()
        ).to(device)

        criterion = criterion.to(device)
        optimizer = optimizer(probe.parameters(), lr=1e-4)

        # register hook on layer
        for name, module in model.named_modules():
            module._forward_hooks.clear()

        handle = layer.register_forward_hook(_hook)

        # dequeue trainloader through model and train probe
        self._

        # dequeue valloader through probe and measure loss / acc

        probe.eval()

        acc = torchmetrics.Accuracy().to(device)
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                _ = model(inputs)
                outputs = probe(rep)
                loss = criterion(outputs, targets)
                acc(outputs, targets)
                total_loss += loss.item()

        handle.remove()

        return total_loss / len(trainloader), acc.compute().item()
