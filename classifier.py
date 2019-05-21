import torch
from collections import OrderedDict
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
import pickle


class Classifier:
    def __init__(self, net, batch_size, ds_path):
        """
        The classifier used for training and launching predictions
        Args:
            net (nn.Module): The neural net module containing the definition of your model
        """
        self.net = net
        self.epochs = None
        self.epoch_counter = 0
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.writer = SummaryWriter()

    def _train_epoch(self, train_loader, optimizer, criterion):
        losses = []
        it_count = len(train_loader)
        with tqdm(total=it_count,
                  desc="Epochs {}/{}".format(self.epoch_counter + 1, self.epochs),
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
                  ) as pbar:
            for iteration, (inputs, targets) in enumerate(train_loader):
                if self.use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                
                inputs, targets = Variable(inputs), Variable(targets)
                # forward

                logits = self.net.forward(inputs)
                optimizer.zero_grad()
#                 print (targets)
          
#                 print (logits.size(), targets.size())
                loss = criterion(logits, torch.squeeze(targets,dim=1))

                loss.backward()
                optimizer.step()
                # print statistics
#                 losses.append(loss.item())
                self.writer.add_scalar('loss', loss.item(), iteration)
                _, preds = torch.max(logits, 1)

                if self.use_cuda:
                    preds = preds.cpu()
                    targets = targets.cpu()

                acc = accuracy_score(targets.numpy(), preds.numpy())

                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss.item()),
                                             acc='{0:1.5f}'.format(acc)))
                pbar.update(1)
        return loss.item(), acc, losses

    def _validate_epoch(self, val_loader, criterion):
        it_count = len(val_loader)
        losses = []
        with tqdm(total=it_count, desc="Validating", leave=False) as pbar:
            for inputs, targets in val_loader:
                if self.use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    
                inputs, targets = Variable(inputs), Variable(targets)
                # forward
                #                 print (self.net)
                logits = self.net(inputs)
                loss = criterion(logits, targets)


            
                if self.use_cuda:
                    preds = preds.cpu()
                    targets = target.cpu()
                acc = accuracy_score(target.numpy(), preds.numpy())

                losses.append(loss.item())

                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss.item()),
                                             acc='{0:1.5f}'.format(acc)))
                pbar.update(1)
        return loss.item(), acc, losses

    def _run_epoch(self, train_loader, val_loader,
                   optimizer, criterion, lr_scheduler):
        # switch to train mode
        self.net.train()

        # Run a train pass on the current epoch
        train_loss, train_acc, train_losses = self._train_epoch(train_loader, optimizer, criterion)

        # switch to evaluate mode
        self.net.eval()

        # Run the validation pass
        # val_loss, val_acc, val_losses = self._validate_epoch(val_loader, criterion)

        # Reduce learning rate if needed
        # lr_scheduler.step(val_loss, self.epoch_counter)
        with open("train_loss.txt", "a+") as f:
            for item in train_losses:
                f.write("%s\n" % item)

        # with open("val_loss.txt", "a+") as f:
        #     for item in val_losses:
        #         f.write("%s\n" % item)
        #
        # print("train_loss = {:03f}, train_acc = {:03f}\n"
        #       "val_loss   = {:03f}, val_acc   = {:03f}"
        #       .format(train_loss, train_acc, val_loss, val_acc))
        self.epoch_counter += 1

    def train(self, train_loader, val_loader, epochs):
        if self.use_cuda:
            self.net.cuda()

        if self.epochs != None:
            self.epochs += epochs
        else:
            self.epochs = epochs
            self.epoch_counter = 0
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.03)
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, min_lr=1e-7)

        for epoch in range(epochs):
            self._run_epoch(train_loader, val_loader, optimizer, criterion, lr_scheduler)