import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer

class FraudClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FraudClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    # Load the data
    data = pd.read_csv("../data/fraud_detector_data.csv")
    train_data = data[data.Split == 'Train']
    X_train = train_data['Body']
    y_train = train_data['Label']
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    y_train_numerical = y_train.astype(int)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define batch size
    batch_size = 4

    # Instantiate the model and move it to GPU
    model = FraudClassifier(input_size=X_train_vectorized.shape[1], hidden_size=16, output_size=2)
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Create DataLoader and move it to GPU
    train_dataset = TensorDataset(torch.Tensor(X_train_vectorized.toarray()), torch.LongTensor(y_train_numerical.values))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = [(X.to(device), y.to(device)) for X, y in train_dataloader]

    # Define PrivacyEngine parameters
    DELTA = 1e-5
    EPSILON = 8.0
    EPOCHS = 10
    MAX_GRAD_NORM = 1.0

    # Wrap the model with PrivacyEngine
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        target_delta=DELTA,
        target_epsilon=EPSILON, 
        epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
    )

    # Training loop
    num_epochs = 3

    for epoch in range(num_epochs):
        for X_batch, y_batch in train_dataloader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
