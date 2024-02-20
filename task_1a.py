'''
*****************************************************************************************
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*****************************************************************************************
'''
####################### IMPORT MODULES #######################
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler 

def data_preprocessing(task_1a_dataframe):
    

    categorical_columns = ['Education', 'City', 'Gender', 'EverBenched']
    label_encoder = preprocessing.LabelEncoder()
    for col in categorical_columns:
        task_1a_dataframe[col] = label_encoder.fit_transform(task_1a_dataframe[col])

    numerical_columns = ['JoiningYear', 'Age', 'ExperienceInCurrentDomain']
    scaler = StandardScaler()
    task_1a_dataframe[numerical_columns] = scaler.fit_transform(task_1a_dataframe[numerical_columns])

    encoded_dataframe = task_1a_dataframe
    return encoded_dataframe
    
def identify_features_and_targets(encoded_dataframe):
    features = encoded_dataframe.drop(columns=['LeaveOrNot'])
    target = encoded_dataframe['LeaveOrNot']
    features_and_targets = [features, target]
    return features_and_targets

def load_as_tensors(features_and_targets):
    features, target = features_and_targets
    
    X_tensor = torch.tensor(features.values, dtype=torch.float32)
    y_tensor = torch.tensor(target.values, dtype=torch.float32)
    split_ratio = 0.8  
    train_size = int(split_ratio * len(features))
    test_size = len(features) - train_size
    X_train_tensor, X_test_tensor = X_tensor[:train_size], X_tensor[train_size:]
    y_train_tensor, y_test_tensor = y_tensor[:train_size], y_tensor[train_size:]
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    tensors_and_iterable_training_data = [X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader]
    return tensors_and_iterable_training_data

class Salary_Predictor(nn.Module):
    def __init__(self):
        super(Salary_Predictor, self).__init__()
        input_size = tensors_and_iterable_training_data[0].shape[1]

        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)  
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x) 
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(x)  
        x = self.fc4(x)
        x = self.sigmoid(x)

        predicted_output = x
        return predicted_output

def model_loss_function():
    loss_function = nn.BCELoss()
    return loss_function  

def model_optimizer(model):
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    return optimizer

# Number of Epochs
def model_number_of_epochs():
    number_of_epochs = 100
    return number_of_epochs


def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
    X_train_tensor, _, y_train_tensor, _, train_loader = tensors_and_iterable_training_data

    for epoch in range(number_of_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{number_of_epochs}], Loss: {total_loss:.4f}')

    trained_model = model

    return trained_model

def validation_function(trained_model, tensors_and_iterable_training_data):
    _, X_val_tensor, _, y_val_tensor, _ = tensors_and_iterable_training_data
    trained_model.eval()
    with torch.no_grad():
        y_val_pred = trained_model(X_val_tensor)
        y_val_pred = (y_val_pred > 0.5).float() 

    correct = (y_val_pred == y_val_tensor.view(-1, 1)).sum().item()
    total = y_val_tensor.size(0)
    model_accuracy = correct / total * 100.0
    return model_accuracy


if __name__ == "__main__":
    # Reading the provided dataset csv file using pandas library and
    # converting it to a pandas Dataframe
    task_1a_dataframe = pandas.read_csv("task_1a_dataset.csv")

	# data preprocessing and obtaining encoded data
    encoded_dataframe = data_preprocessing(task_1a_dataframe)
    
	# selecting required features and targets
    features_and_targets = identify_features_and_targets(encoded_dataframe)
    
	# obtaining training and validation data tensors and the iterable
    # training data object
    tensors_and_iterable_training_data = load_as_tensors(features_and_targets)

	# model is an instance of the class that defines the architecture of the model
    model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
    loss_function = model_loss_function()
    optimizer = model_optimizer(model)
    number_of_epochs = model_number_of_epochs()
    
	# training the model
    trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer)

    # validating and obtaining accuracy
    model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
    print(f"Accuracy on the test set = {model_accuracy}")
    # Saving the model 
    X_train_tensor = tensors_and_iterable_training_data[0]
    x = X_train_tensor[0]
    jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")