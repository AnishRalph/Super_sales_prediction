# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Setting path to input files
file = "Super_Sales.csv"

# Creating dataframes
data = pd.read_csv(file)
df = pd.DataFrame(data)

# Load your DataFrame
df = pd.DataFrame(data)

# Preprocess the data
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.month
df['Order Day'] = df['Order Date'].dt.day

# Encode categorical features
label_encoders = {}
for column in ['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target
features = ['Order Year', 'Order Month', 'Order Day', 'Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category']
X = df[features]
y = df['Sales']

# Include 'Order Date' in the test set
X['Order Date'] = df['Order Date']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate 'Order Date' for plotting
X_test_dates = X_test['Order Date']
X_train = X_train.drop(columns='Order Date')
X_test = X_test.drop(columns='Order Date')

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define the neural network with dropout for regularization
class SalesPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SalesPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
model = SalesPredictor(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping
epochs = 300
patience = 20
best_loss = float('inf')
counter = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
        best_model = model.state_dict()
    else:
        counter += 1
        
    if counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Load the best model
model.load_state_dict(best_model)

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    mse = criterion(predictions, y_test)
    rmse = torch.sqrt(mse)
    pytorch_rmse_4 = rmse.item()
    print(f'Root Mean Squared Error on Test Data: {pytorch_rmse_4:.4f}')



# Prepare data for plotting
predictions = predictions.numpy().flatten()
y_test = y_test.numpy().flatten()
results_df = pd.DataFrame({'Order Date': X_test_dates, 'Actual Sales': y_test, 'Predicted Sales': predictions})
results_df['Order Year'] = results_df['Order Date'].dt.year

# Group by year and sum sales
grouped_results = results_df.groupby('Order Year').sum().reset_index()

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(grouped_results['Order Year'], grouped_results['Actual Sales'], label='Actual Sales', marker='o')
plt.plot(grouped_results['Order Year'], grouped_results['Predicted Sales'], label='Predicted Sales', marker='x')
plt.xlabel('Order Year')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
