# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Write your own steps

### STEP 2: 

Load the Fashion MNIST dataset, apply image preprocessing, and create training and testing data loaders.

### STEP 3:
Design and initialize the Convolutional Neural Network (CNN) architecture with convolution, pooling, and fully connected layers.

### STEP 4:
Define the loss function and optimizer, then train the CNN model using the training dataset.

### STEP 5:
Evaluate the trained model using the testing dataset and calculate accuracy, confusion matrix, and classification report.

### STEP 6:
Test the model with a new image and display the actual and predicted class labels. 





## PROGRAM

### Name:S.R.AMIRTHA

### Register Number:212224220009

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from torchsummary import summary

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

im, lbl = train_set[0]

print("Image Shape:", im.shape)
print("Training Dataset:", len(train_set))
print("Testing Dataset:", len(test_set))


trl = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True
)

tstl = DataLoader(
    test_set,
    batch_size=32,
    shuffle=False
)



class CNNclassifier1(nn.Module):

    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.c2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.c3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.l1 = nn.Linear(128 * 3 * 3, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 10)

    def forward(self, x):

        x = self.pool(torch.relu(self.c1(x)))

        x = self.pool(torch.relu(self.c2(x)))

        x = self.pool(torch.relu(self.c3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.l1(x))

        x = torch.relu(self.l2(x))

        x = self.l3(x)

        return x


model = CNNclassifier1()

criterion = nn.CrossEntropyLoss()

op = optim.Adam(
    model.parameters(),
    lr=0.001
)



device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print("Device:", device)

model.to(device)

summary(
    model,
    input_size=(1, 28, 28),
    device=str(device)
)

epochs = 3

for i in range(epochs):

    model.train()

    running_loss = 0.0

    for a, b in trl:

        a = a.to(device)
        b = b.to(device)

        op.zero_grad()

        pred = model(a)

        loss = criterion(pred, b)

        loss.backward()

        op.step()

        running_loss += loss.item()

    print(
        f"Epoch {i+1}, Loss: "
        f"{running_loss/len(trl):.4f}"
    )


t = 0
c = 0

act = []
pre = []

model.eval()

with torch.no_grad():

    for img, labels in tstl:

        img = img.to(device)
        labels = labels.to(device)

        output = model(img)

        _, predicted = torch.max(output, 1)

        t += labels.size(0)

        c += (predicted == labels).sum().item()

        pre.extend(predicted.cpu().numpy())

        act.extend(labels.cpu().numpy())


accuracy = c / t * 100

print("Accuracy Score:", accuracy)

conf_matrix = confusion_matrix(act, pre)

class_report = classification_report(
    act,
    pre,
    target_names=test_set.classes
)

print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 8))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=test_set.classes,
    yticklabels=test_set.classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

with torch.no_grad():

    img1, label = test_set[0]

    output = model(
        img1.unsqueeze(0).to(device)
    )

    _, predicted = torch.max(output, 1)

    classes = test_set.classes

    img1 = img1 * 0.5 + 0.5

    plt.imshow(
        img1.squeeze(),
        cmap="gray"
    )

    plt.title("Predicted Image")

    plt.axis('off')

    plt.show()

    print("Actual:", classes[label])

    print("Predicted:", classes[predicted.item()])


original_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=None
)

image, label = original_dataset[0]

plt.imshow(image, cmap='gray')

plt.title(original_dataset.classes[label])

plt.axis('off')

plt.show()

print("Original Class:", original_dataset.classes[label])


```

### OUTPUT

## Training Loss per Epoch

<img width="718" height="640" alt="image" src="https://github.com/user-attachments/assets/13506cff-3dd4-4f3c-8ed2-34e191385fe4" />


## Confusion Matrix

<img width="917" height="788" alt="image" src="https://github.com/user-attachments/assets/31c34017-67ee-4d0d-ad5a-55ab6e70da94" />


## Classification Report

<img width="863" height="407" alt="image" src="https://github.com/user-attachments/assets/2773b4af-5f66-4ca2-bfe3-e8333aab4fce" />


### New Sample Data Prediction

<img width="600" height="538" alt="image" src="https://github.com/user-attachments/assets/52cdba25-8756-4062-a044-66627a2f8c23" />


## RESULT

The CNN model was successfully trained and tested on the Fashion MNIST dataset, achieving accurate image classification with performance evaluation metrics.
