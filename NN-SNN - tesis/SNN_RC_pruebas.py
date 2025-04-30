import snntorch as snn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torchvision import datasets, transforms #
from snntorch import utils
from torch.utils.data import TensorDataset, DataLoader
from snntorch import spikegen
import numpy as np
import random
import itertools
import csv
import torch.optim as optim

# Definir nombres de archivos para los datos
X_train_file = "X_train.pt"
y_train_file = "y_train.pt"
X_test_file = "X_test.pt"
y_test_file = "y_test.pt"

# Cargar datos desde los archivos .pt
print("Cargando datos desde archivos .pt...")
X_train = torch.load(X_train_file)
y_train = torch.load(y_train_file)
X_test = torch.load(X_test_file)
y_test = torch.load(y_test_file) 

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparámetros
num_inputs = 28*28  
num_outputs = 10
learning_rate = 0.01
batch_size=  500
num_trials = 100

# Temporal Dynamics           
beta = 0.95 

# Verificar dimensiones y transformar si es necesario
X_train = X_train.view(-1, 28 * 28).float()
X_test = X_test.view(-1, 28 * 28).float()


# Crear datasets y DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

csv_filename = "resultados_experimentos_SNNRC_ganadora280_200_ts.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["hidden_sizes", "num_epochs", "num_steps", "optimizer", "num_params", "mean_accuracy", "std_accuracy", "accuracies"])


# Torch Variables
dtype = torch.float       
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))]) 

  

class Net(nn.Module):
    def __init__(self, num_inputs, hidden_sizes, num_outputs):
        super().__init__()

        self.num_layers = len(hidden_sizes)
        self.lif_layers = nn.ModuleList()   
        self.fc_layers = nn.ModuleList()    

        # Input layer
        self.fc_layers.append(nn.Linear(num_inputs, hidden_sizes[0]))
        self.lif_layers.append(snn.Leaky(beta=beta))

        # Hidden layers
        for i in range(1, self.num_layers):
            self.fc_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.lif_layers.append(snn.Leaky(beta=beta))

        # Output layer
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], num_outputs))
        self.lif_layers.append(snn.Leaky(beta=beta))  
    

    def forward(self, x):
        # Initialize hidden states at t=0
        mem_states = [layer.init_leaky() for layer in self.lif_layers]

        # Record the final layer
        spk_rec = []
        mem_rec = []

        for step in range(num_steps):
            cur = x[step]                
            for i in range(len(self.fc_layers)):
                    cur = self.fc_layers[i](cur)  
                    spk, mem_states[i] = self.lif_layers[i](cur, mem_states[i])  
                    cur = spk  # Pasamos el spike a la siguiente capa
            spk_rec.append(cur)
            mem_rec.append(mem_states[-1])  
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)    

def train_and_evaluate(num_epochs, num_steps, optimi, hidden_sizes):
    net = Net(num_inputs, hidden_sizes, num_outputs).to(device)    

    # nums of parameters
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    CE_loss = nn.CrossEntropyLoss()
    if optimi == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    elif optimi == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # train
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        net.train()
        for i, (data, targets) in enumerate(train_loader): #for data, targets in train_loader:
            spike_data = spikegen.rate(data, num_steps=num_steps)
            spike_data = spike_data.to(device)                                 #.to(device) mueve el tensor data al dispositivo especificado para. Data: [batch_size, num_inputs], targets: 
            targets = targets.to(device) 
            spike_data = spike_data.view(num_steps, data.size(0), -1)
            
            optimizer.zero_grad()

            outputs, mem_rec = net(spike_data)
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += CE_loss(mem_rec[step], targets)          
                
            loss_val.backward()
            optimizer.step() 

        if (i+1) % 100 == 0:      #Si se quiere imprimir esta parte se debe cambiar el 100 por algo más pequño ya que se cambió el batch_size a 500, y el el train loader puede ser menor que 100
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss_val.item():.4f}')

    # Test the model
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:

            targets = targets.to(device)

            spike_data = spikegen.rate(data, num_steps=num_steps)
            spike_data = spike_data.to(device)
            spike_data = spike_data.view(num_steps, data.size(0), -1)

            outputs, mem = net(spike_data)
            _, pred = outputs.sum(dim=0).max(1)

            correct += (pred == targets).sum().item()
            total += targets.size(0)
            

    print(f"Accuracy: {100 * correct / total:.2f}%")

    return 100 * correct / total, num_params

# Generar listas de capas ocultas
one_layer = [[480]]
#two_layers = [[128, 128]]
#three_layers = [[128, 128, 128]]
#five_layers = [[128, 128, 128, 128, 128]]

hidden_layers_list = one_layer # + two_layers + three_layers + five_layers

num_time_steps_list = list(range(65, 200, 5))  # 5, 10, ..., 30
optimizer = ["adam"]
num_epochs = [1, 2, 5, 10]


configurations = list(itertools.product(hidden_layers_list, num_time_steps_list, optimizer, num_epochs))

print(f"🔹 Total de configuraciones a probar: {len(configurations)}")


for config in configurations:  # configurations ya tiene todas las combinaciones hidden_layers_list, num_time_steps_list, optimizer, num_epochs
    hidden_sizes, num_steps, optimizer, num_epochs = config  
    accuracies = []
    for _ in range(num_trials):
        acc, num_params = train_and_evaluate(num_epochs, num_steps, optimizer, hidden_sizes)
        accuracies.append(acc)
        print(f"Accuracy en prueba {_+1}: {acc:.4f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([hidden_sizes, num_epochs, num_steps, optimizer, num_params, mean_acc, std_acc, accuracies])

print(f"\n🔹 Resultados guardados en {csv_filename}")



print(accuracies)
print(f"\n🔹 Promedio de accuracy en {num_trials} pruebas: {mean_acc:.4f}")
print(f"🔹 Desviación estándar: {std_acc:.4f}") 