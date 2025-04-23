import torch
from models.model import ExampleCNN
from datasets.dataloader import make_test_dataloader
import argparse

import os
from tqdm import tqdm
import json

# argumant
parser = argparse.ArgumentParser(description='Test a CNN model')
parser.add_argument("--weight-file", type=str, required=True, help="path to the weight file")
parser.add_argument("--log", type=str, required=True, help="path to the log file")
args= parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

base_path = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(base_path, "data", "test")
weight_path = os.path.join(base_path, "weights", args.weight_file)

# load model and use weights we saved before
model = ExampleCNN()
model.load_state_dict(torch.load(weight_path, weights_only=True))
model = model.to(device)

# make dataloader for test data
test_loader = make_test_dataloader(test_data_path)

predict_correct = 0
model.eval()
with torch.no_grad():
    for data, target in tqdm(test_loader, desc="Testing",bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
        data, target = data.to(device), target.to(device)

        output = model(data)
        predict_correct += (output.data.max(1)[1] == target.data).sum()
        
    accuracy = 100. * predict_correct / len(test_loader.dataset)
print(f'Test accuracy: {accuracy:.4f}%')


with open(args.log, "a") as f:
    f.write(json.dumps(
        {
            "weight_path": weight_path,
            "accuracy": float(f"{accuracy:.4f}")
        }
    ) + "\n")