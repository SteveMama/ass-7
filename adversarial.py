import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import *
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Palette images with Transparency expressed in bytes should be converted to RGBA images")


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)


# define CNN for a 3-class problem with input size 160x160 images
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.final_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = x.view(-1, 256 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.final_activation(x)
        return x


# Load dataset
train_dir = "./data/train"
test_dir = "./data/test"
image_size = 160
batch_size = 16
workers = 0


class CropToSmallerDimension(transforms.Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)

    def __call__(self, img):
        # Get the original image size
        width, height = img.size

        # Determine the smaller dimension
        smaller_dimension = min(width, height)

        # Crop the image to the smaller dimension
        return transforms.CenterCrop(smaller_dimension)(img)


train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=transforms.Compose(
        [
            CropToSmallerDimension(256),
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
        ]
    ),
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers
)

test_dataset = datasets.ImageFolder(
    root=test_dir,
    transform=transforms.Compose(
        [
            CropToSmallerDimension(256),
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
        ]
    ),
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers
)

print("Number of training images: {}".format(len(train_dataset)))
print("Number of test images: {}".format(len(test_dataset)))
print(
    "Detected Classes are: ", train_dataset.classes
)  # classes are detected by folder structure


# Define the attack
def FGSM(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def PGD(model, image, label, epsilon, iterations, alpha):
    orig_image = image.detach().clone().to(device)

    # Initialize perturbed image as a clone of the original image
    perturbed_image = orig_image.clone()

    for i in range(iterations):
        perturbed_image.requires_grad = True

        output = model(perturbed_image)
        model.zero_grad()

        loss = F.nll_loss(output, label)
        loss.backward()

        # Add perturbation
        perturbation = alpha * perturbed_image.grad.data.sign()
        perturbed_image = perturbed_image + perturbation

        # Project the perturbation to make sure it's within the epsilon ball
        perturbation = torch.clamp(perturbed_image - orig_image, min=-epsilon, max=epsilon)
        perturbed_image = orig_image + perturbation

        # Clamp the image to [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()

    return perturbed_image


net = Net()
net.to(device)

# Train the network

# criterion = nn.NLLLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.001)
# epochs = 100
# running_loss = 0
# train_losses, test_losses = [], []
# i=0

# for epoch in tqdm(range(epochs)):
# 	for inputs, labels in train_dataloader:
# 		inputs, labels = inputs.to(device), labels.to(device)
# 		optimizer.zero_grad()
# 		logps = net(inputs)
# 		loss = criterion(logps, labels)
# 		loss.backward()
# 		optimizer.step()
# 		running_loss += loss.item()

# # Save the model
# torch.save(net.state_dict(), 'model.pth')


# Test the model
net.load_state_dict(torch.load("model.pth", map_location="cpu"))
net.to(device)

correct = []

net.eval()
accuracy = 0
for inputs, labels in test_dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    accuracy += (predicted == labels).sum().item()
    correct.append((predicted == labels).tolist())

print("="*50)
print(
    "Accuracy of the network on the test images: {:.2f} %".format(100 * accuracy / len(test_dataset))
)
print("="*50)

# Test the model with adversarial examples
# Save adversarial examples for each class using FGSM with (eps = 0.001, 0.01, 0.1)
# Save one adversarial example for each class using PGD with (eps = 0.01, 0.05, 0.1, alpha = 0.001, 0.005, 0.01 respectively, iterations = 20)

# Adversarial Testing 
epsilon_values_fgms = [0.001, 0.01, 0.1]
epsilon_values_pgd = [0.01, 0.05, 0.1]

saved_adversarials_FGMS = {
    eps: {cls: None for cls in train_dataset.classes} for eps in epsilon_values_fgms
}
saved_adversarials_PGD = {
    eps: {cls: None for cls in train_dataset.classes} for eps in epsilon_values_pgd
}
saved_originals = {cls: False for cls in train_dataset.classes}

for idx, _ in enumerate(epsilon_values_pgd):
    adversarial_accuracy_FGMS, adversarial_accuracy_PGD = 0, 0
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True

        # Forward pass
        outputs = net(inputs)
        loss = F.nll_loss(outputs, labels)
        net.zero_grad()
        loss.backward()
        data_grad = inputs.grad.data

        # FGSM Attack
        perturbed_data_FGSM = FGSM(inputs, epsilon_values_fgms[idx], data_grad)

        # PGD Attack
        perturbed_data_PGD = PGD(net, inputs, labels, epsilon_values_pgd[idx], 50, 2/225)

        # Save one example per class (both original and adversarial)
        for i in range(inputs.size(0)):
            class_name = train_dataset.classes[labels[i].item()]
            if not saved_originals[class_name]:
                vutils.save_image(inputs[i], f"original_{class_name}.png")
                saved_originals[class_name] = True
            if not saved_adversarials_FGMS[ epsilon_values_fgms[idx]][class_name]:
                vutils.save_image(
                    perturbed_data_FGSM[i], f"adv_FGSM_{class_name}_eps{epsilon_values_fgms[idx]}.png"
                )
                saved_adversarials_FGMS[ epsilon_values_fgms[idx]][class_name] = True
            if not saved_adversarials_PGD[epsilon_values_pgd[idx]][class_name]:
                vutils.save_image(
                    perturbed_data_PGD[i], f"adv_PGD_{class_name}_eps{epsilon_values_pgd[idx]}.png"
                )
                saved_adversarials_PGD[epsilon_values_pgd[idx]][class_name] = True

        # Re-classify the perturbed images
        outputs_FGSM = net(perturbed_data_FGSM)
        _, predicted_FGMS = torch.max(outputs_FGSM.data, 1)
        adversarial_accuracy_FGMS += (predicted_FGMS == labels).sum().item()
        
        outputs_PGD = net(perturbed_data_PGD)
        _, predicted_PGD = torch.max(outputs_PGD, 1)
        adversarial_accuracy_PGD += (predicted_PGD == labels).sum().item()

    print(
        "Adversarial Test Accuracy Using FGSM for Epsilon {}: {:.2f}%".format(
            epsilon_values_fgms[idx], 100 * adversarial_accuracy_FGMS / len(test_dataset)
        )
    )
    
    print(
        "Adversarial Test Accuracy Using PGD for Epsilon {}: {:.2f}%".format(
            epsilon_values_pgd[idx], 100 * adversarial_accuracy_PGD / len(test_dataset)
        )
    )
    
    print("="*50)


# updated training loop with adverserial training
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
additional_epochs = 10
epsilon = 0.075
alpha = 2/255
iterations = 50

for epoch in tqdm(range(additional_epochs)):
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Generate adversarial images using PGD
        adversarial_inputs = PGD(net, inputs, labels, epsilon, iterations, alpha)

        # Training on adversarial images
        optimizer.zero_grad()
        logps = net(adversarial_inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

# Saving the updated model weights
torch.save(net.state_dict(), 'model_adversarially_trained.pth')

# Testing the updated model
net.load_state_dict(torch.load("model.pth", map_location="cpu"))
net.to(device)

correct = []

# Test the model
net.load_state_dict(torch.load("model_adversarially_trained.pth", map_location="cpu"))
net.to(device)

net.eval()
accuracy = 0
for inputs, labels in test_dataloader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    accuracy += (predicted == labels).sum().item()
    correct.append((predicted == labels).tolist())

print("="*50)
print(
    "Accuracy of the adversarially trained network on the clean test images: {:.2f} %".format(100 * accuracy / len(test_dataset))
)
print("="*50)

for epsilon in epsilon_values_pgd:
    adversarial_accuracy_PGD = 0
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True

        # PGD Attack
        perturbed_data_PGD = PGD(net, inputs, labels, epsilon, 50, 2/225)

        outputs_PGD = net(perturbed_data_PGD)
        _, predicted_PGD = torch.max(outputs_PGD, 1)
        adversarial_accuracy_PGD += (predicted_PGD == labels).sum().item()

    print(
        "Adversarial Test Accuracy on the Adverarially trained model Using PGD for Epsilon {}: {:.2f}%".format(
            epsilon, 100 * adversarial_accuracy_PGD / len(test_dataset)
        )
    )
    
print("="*50)