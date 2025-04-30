# Experimento para variar más el numero de capas y epocas
import torch
import numpy as np
import csv
import itertools
from itertools import combinations
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
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
num_inputs = 784  
num_outputs = 10
learning_rate = 0.01
batch_size = 500
num_trials = 100  # Número de pruebas

# Verificar dimensiones y transformar si es necesario
X_train = X_train.view(-1, 28 * 28).float()
X_test = X_test.view(-1, 28 * 28).float()

# Crear datasets y DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

csv_filename = "resultados_experimentos_mas_capas_NN.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["hidden_sizes", "num_epochs", "optimizer","activation_list", "num_params", "mean_accuracy", "std_accuracy", "accuracies"])

class NeuralNet(nn.Module):
    def __init__(self, num_inputs, hidden_sizes, num_outputs, activation_list):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(num_inputs, hidden_sizes[0]))
        self.layers.append(self.get_activation_function(activation_list[0]))

        # Hidden Layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(self.get_activation_function(activation_list[i]))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], num_outputs))

    def get_activation_function(self, activation):
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())  

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_and_evaluate(num_epochs, optimi, hidden_sizes, activation_list):
    model = NeuralNet(num_inputs, hidden_sizes, num_outputs, activation_list).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss()

    if optimi == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimi == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Entrenamiento
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluación
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total, num_params

# Rango de valores para el número de neuronas en cada capa
optimizer = ["adam", "sgd"] 
num_epochs = [5, 10]
activation = ["relu", "tanh", "leaky_relu", "sigmoide"] #

# Generar activaciones repetidas
activation_combinations = (
    [[a] for a in activation] +
    [[a] * 2 for a in activation] +
    [[a] * 3 for a in activation] +
    [[a] * 5 for a in activation]
)
# Generar listas de capas ocultas
one_layer = [[480]]
two_layers = [[128, 128]]
three_layers = [[128, 128, 128]]
five_layers = [[128, 128, 128, 128, 128]]

hidden_layers_list = one_layer + two_layers + three_layers + five_layers
# Filtrar configuraciones válidas (número de activaciones = número de capas ocultas)
valid_configurations = []
for hidden_layers in hidden_layers_list:
    num_hidden_layers = len(hidden_layers)
    valid_activations = [act for act in activation_combinations if len(act) == num_hidden_layers]
    for activation_set in valid_activations:
        for opt, epochs in itertools.product(optimizer, num_epochs):
            valid_configurations.append((hidden_layers, opt, epochs, activation_set))
            #print(valid_configurations)
# Filtrar configuraciones válidas (número de activaciones = número de capas ocultas)

print(f" Total de configuraciones a probar: {len(valid_configurations)}")


for config in valid_configurations:
    hidden_sizes, opt, epochs, act_list = config
    results = [train_and_evaluate(epochs, opt, hidden_sizes, act_list) for _ in range(num_trials)]
    
    # Obtener la precisión promedio y la desviación estándar
    accuracies = [result[0] for result in results]
    mean_acc, std_acc = np.mean(accuracies), np.std(accuracies)
    
    # Obtener num_params de cualquiera de los resultados (todos deberían ser iguales)
    num_params = results[0][1]

    with open(csv_filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([hidden_sizes, epochs, opt, act_list, num_params, mean_acc, std_acc, accuracies])

print(f"\n🔹 Resultados guardados en {csv_filename}")