# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Mohamed Hameem Sajith J

### Register Number: 212223240090

```
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/admin/Desktop/dl/.venv/exp1.csv")
df
x=df[["Input"]].values
y=df[["Output"]].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=52)
scale1=MinMaxScaler()
xt=scale1.fit_transform(xt)
xst=scale1.transform(xst)
xt1=torch.FloatTensor(xt)
xst1=torch.FloatTensor(xst)
yt1=torch.FloatTensor(yt)
yst1=torch.FloatTensor(yst)
class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1))
    def forward(self,x):
        return self.network(x)
model=neuralnet()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
epochs=1000
losses=[]
for i in range(epochs):
    optimizer.zero_grad()
    pred=model(xt1)
    loss=criterion(pred,yt1)
    loss.backward()
    optimizer.step()

    if(i%50==0):
        print(f"{i}/{epochs}:",loss.item())   
        losses.append(loss.item())
x11=np.array([[16]])
ip=torch.FloatTensor(scale1.transform(x11))
pred=model(ip)
print(pred.item())
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


```

### Dataset Information
Include screenshot of the generated data

<img width="1358" height="793" alt="image" src="https://github.com/user-attachments/assets/87b36696-3eb3-49c6-829b-5e359c6803ff" />


### OUTPUT

### Training Loss Vs Iteration Plot
Include your plot here

<img width="1747" height="937" alt="image" src="https://github.com/user-attachments/assets/4e995b5d-df62-40e0-890f-497632d2b5ce" />


### New Sample Data Prediction
Include your sample input and output here

<img width="682" height="617" alt="image" src="https://github.com/user-attachments/assets/4f914537-4419-4b93-916c-2e0a6a407a6b" />

#### predicted loss
<img width="1672" height="268" alt="image" src="https://github.com/user-attachments/assets/a363fcd2-0002-45d8-b18e-d3b11740d190" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
