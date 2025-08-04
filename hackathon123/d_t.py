import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths
DATA_PATH = 'diabetes.csv'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Create PyTorch Dataset class
class DiabetesDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
            self.has_targets = True
        else:
            self.has_targets = False
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.has_targets:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]

# Define Neural Network Model
class DiabetesNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super(DiabetesNN, self).__init__()
        
        # Input layer
        layers = [nn.Linear(input_dim, hidden_dims[0])]
        
        # Hidden layers with Batch Normalization and Dropout
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.BatchNorm1d(hidden_dims[i]),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dims[i], hidden_dims[i+1])
            ])
        
        # Output layer
        layers.extend([
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Data Preprocessing Class
class DiabetesPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        
    def fit(self, X):
        # Replace zeros with NaN for specific columns where zero doesn't make sense
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        X_processed = X.copy()
        for col in cols_with_zeros:
            X_processed.loc[X_processed[col] == 0, col] = np.nan
        
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X_processed)
        
        # Apply power transform to make data more Gaussian-like
        X_transformed = self.power_transformer.fit_transform(X_imputed)
        
        # Scale the features
        self.scaler.fit(X_transformed)
        
        return X_processed
        
    def transform(self, X):
        # Replace zeros with NaN for specific columns
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        X_processed = X.copy()
        for col in cols_with_zeros:
            X_processed.loc[X_processed[col] == 0, col] = np.nan
        
        # Apply the same preprocessing as during fit
        X_imputed = self.imputer.transform(X_processed)
        X_transformed = self.power_transformer.transform(X_imputed)
        X_scaled = self.scaler.transform(X_transformed)
        
        return X_scaled
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Function to create derived features
def create_derived_features(df):
    df_derived = df.copy()
    
    # Create BMI categories
    df_derived['BMI_Category'] = pd.cut(df_derived['BMI'], 
                                       bins=[0, 18.5, 25, 30, 100],
                                       labels=[0, 1, 2, 3])
    
    # Create Age groups
    df_derived['Age_Group'] = pd.cut(df_derived['Age'],
                                    bins=[20, 30, 40, 50, 60, 100],
                                    labels=[0, 1, 2, 3, 4])
    
    # Calculate Glucose to Insulin ratio (with safe division)
    df_derived['Glucose_Insulin_Ratio'] = df_derived['Glucose'] / df_derived['Insulin'].replace(0, 1)
    
    # Interaction between Age and BMI
    df_derived['Age_BMI_Interaction'] = df_derived['Age'] * df_derived['BMI'] / 100
    
    # Interaction between Pregnancies and Age
    df_derived['Preg_Age_Interaction'] = df_derived['Pregnancies'] * df_derived['Age'] / 10
    
    # Convert categorical columns to int (handle NaN values that might be generated for edge cases)
    df_derived['BMI_Category'] = df_derived['BMI_Category'].fillna(0).astype(int)
    df_derived['Age_Group'] = df_derived['Age_Group'].fillna(0).astype(int)
    
    return df_derived

# Function for early stopping
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100, patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(MODEL_DIR, 'best_model.pt'))
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Learning rate adjustment
        scheduler.step(val_loss)
        
        # Early stopping check
        early_stopping(val_loss, model)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))
    return model, train_losses, val_losses

# Evaluation function
def evaluate_model(model, test_loader, threshold=0.5):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = outputs.cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    all_probs = np.array(all_probs).flatten()
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    roc_auc = roc_auc_score(all_targets, all_probs)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix
    }
    
    return metrics, all_probs

# Main function
def main():
    # Load dataset
    print("Loading and exploring data...")
    df = pd.read_csv(DATA_PATH)
    
    # Data exploration and visualization
    print("\nDataset overview:")
    print(df.describe())
    
    # Check for missing values (implicit zeros)
    print("\nColumns with zero values (potentially missing data):")
    for col in df.columns:
        if col != 'Outcome':
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                print(f"{col}: {zero_count} zeros ({zero_count/len(df)*100:.1f}%)")
    
    # Analyze target distribution
    print("\nTarget distribution:")
    print(df['Outcome'].value_counts(normalize=True))
    
    # Create derived features
    print("\nCreating derived features...")
    df_derived = create_derived_features(df)
    
    # Correlation analysis
    plt.figure(figsize=(12, 10))
    correlation = df_derived.corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', mask=mask)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'correlation_matrix.png'))
    
    # Calculate correlation with target
    target_corr = correlation['Outcome'].sort_values(ascending=False)
    print("\nFeature correlation with target (Outcome):")
    print(target_corr)
    
    # Feature importance analysis will be done after model training
    
    # Split into features and target
    X = df_derived.drop('Outcome', axis=1)
    y = df_derived['Outcome']
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocess the data
    print("\nPreprocessing data...")
    preprocessor = DiabetesPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.pkl'))
    
    # Create the PyTorch datasets and dataloaders
    batch_size = 32
    
    # Create a validation set from the training set
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    train_dataset = DiabetesDataset(X_train_final, y_train_final.values)
    val_dataset = DiabetesDataset(X_val, y_val.values)
    test_dataset = DiabetesDataset(X_test_processed, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize the model
    input_dim = X_train_processed.shape[1]
    model = DiabetesNN(input_dim=input_dim, hidden_dims=[64, 32, 16])
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Train the model
    print("\nTraining the model...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=200, patience=15
    )
    
    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(MODEL_DIR, 'training_loss.png'))
    
    # Evaluate the model
    print("\nEvaluating the model...")
    metrics, _ = evaluate_model(model, test_loader)
    
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save the model
    print("\nSaving the model...")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'diabetes_model.pt'))
    
    # Save model architecture separately
    torch.save({
        'input_dim': input_dim,
        'hidden_dims': [64, 32, 16]
    }, os.path.join(MODEL_DIR, 'model_architecture.pt'))
    
    # Save column names for future use
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'feature_names.pkl'))
    
    print(f"\nModel and preprocessing tools saved to {MODEL_DIR}/")
    
    # Feature importance using permutation importance
    print("\nCalculating feature importance...")
    
    # Create a scikit-learn compatible wrapper for our PyTorch model
    class ModelWrapper:
        def __init__(self, model):
            self.model = model
            
        def fit(self, X, y):
            # This is a dummy method to satisfy scikit-learn's API
            return self
            
        def predict(self, X):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            self.model.eval()
            with torch.no_grad():
                probs = self.model(X_tensor).cpu().numpy()
                return (probs >= 0.5).astype(int)
                
        def predict_proba(self, X):
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            self.model.eval()
            with torch.no_grad():
                probs = self.model(X_tensor).cpu().numpy()
                return np.hstack([1-probs, probs])  # Return probabilities for both classes
                
        def score(self, X, y):
            # Calculate accuracy score
            y_pred = self.predict(X)
            return np.mean(y_pred.flatten() == y)
    
    # Create the wrapper instance
    model_wrapper = ModelWrapper(model)
    
    # Calculate feature importance using a specific scoring metric to avoid relying on .score()
    perm_importance = permutation_importance(model_wrapper, X_test_processed, y_test, 
                                           scoring='accuracy', n_repeats=10, 
                                           random_state=42, n_jobs=-1)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'))
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()
