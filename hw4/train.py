import torch
import torch.nn as nn
from datasets.dataloader import make_train_dataloader
from models.model import ExampleCNN

import os
import copy
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse

# argumant
parser = argparse.ArgumentParser(description='Train a CNN model')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument("--batch-size", type=int, default=8, help="batch size")
parser.add_argument("--learning-rate", type=float, default=0.01, help="learning rate")
parser.add_argument("--loss", type=str, choices=["cross_entropy", "label_smoothing_lossS", "focal_loss"], default="cross_entropy", help="loss function to use")
parser.add_argument("--weight-file", type=str, default="weight.pth", help="path to save the weights")
parser.add_argument("--img-desc", type=str, default="", help="image suffix")
args = parser.parse_args()

# training parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = args.epochs
learning_rate = args.learning_rate
batch_size = args.batch_size

# data path and weight path
base_path = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(base_path, "data", "train")
weight_path = os.path.join(base_path, "weights", args.weight_file)
# if args.loss != "cross_entropy":
#     weight_path = os.path.join(base_path, "weights", f"weight_e{epochs}_bs{batch_size}_lr{learning_rate}_loss_{args.loss}.pth")

# make dataloader for train data
train_loader, valid_loader = make_train_dataloader(train_data_path, batch_size=batch_size)

# set cnn model
criterion = nn.CrossEntropyLoss()
if args.loss == "focal_loss":
    from models.loss.focal import FocalLoss
    criterion = FocalLoss()
elif args.loss == "label_smoothing_loss":
    from models.loss.LabelSmoothingLoss import LabelSmoothingLoss
    criterion = LabelSmoothingLoss()

model = ExampleCNN()
model = model.to(device)

# set optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# train
train_loss_list = list()
valid_loss_list = list()
train_accuracy_list = list()
valid_accuracy_list = list()
best = 100
best_model_wts = copy.deepcopy(model.state_dict())
for epoch in range(epochs):
    print(f'\nEpoch: {epoch+1}/{epochs}')
    print('-' * len(f'Epoch: {epoch+1}/{epochs}'))
    train_loss, valid_loss = 0.0, 0.0
    train_correct, valid_correct = 0, 0
    train_accuracy, valid_accuracy = 0.0, 0.0

    model.train()
    for data, target in tqdm(train_loader, desc="Training", bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        data, target = data.to(device), target.to(device)

        # forward + backward + optimize
        output  = model(data)
        _, preds = torch.max(output.data, 1)
        loss = criterion(output, target)
        optimizer.zero_grad()   # zero the parameter gradients
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        train_correct += torch.sum(preds == target.data)
    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    train_accuracy = float(train_correct) / len(train_loader.dataset)
    train_accuracy_list.append((train_accuracy))

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(valid_loader, desc="Validation",bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            _, preds = torch.max(output.data, 1)

            valid_loss += loss.item() * data.size(0)
            valid_correct += torch.sum(preds == target.data)
        valid_loss /= len(valid_loader.dataset)
        valid_loss_list.append(valid_loss)
        valid_accuracy = float(valid_correct) / len(valid_loader.dataset)
        valid_accuracy_list.append((valid_accuracy))
    
    # print loss and accuracy in one epoch
    print(f'Training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}')
    print(f'Training accuracy: {train_accuracy:.4f}, validation accuracy: {valid_accuracy:.4f}')

    # record best weight so far
    if valid_loss < best :
        best = valid_loss
        best_model_wts = copy.deepcopy(model.state_dict())
# save the best weight
torch.save(best_model_wts, weight_path)

# plot the loss curve for training and validation
print("\nFinished Training")

pd.DataFrame({
    "train-loss": train_loss_list,
    "valid-loss": valid_loss_list
}).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(1,epoch+1)
plt.xlabel("Epoch"),plt.ylabel("Loss")
plt.savefig(os.path.join(base_path, "result", f"Loss_curve_{args.img_desc}.png"))

# plot the accuracy curve for training and validation
pd.DataFrame({
    "train-accuracy": train_accuracy_list,
    "valid-accuracy": valid_accuracy_list
}).plot()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(1,epoch+1)
plt.xlabel("Epoch"),plt.ylabel("Accuracy")
plt.savefig(os.path.join(base_path, "result", f"Training_accuracy_{args.img_desc}.png"))

