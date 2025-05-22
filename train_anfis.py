import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import logging
import matplotlib.pyplot as plt
from anfis_model import ANFIS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/anfis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """Load and preprocess UNSW-NB15 dataset"""
    try:
        train_df = pd.read_csv('dataset/UNSW_NB15_training-set.csv')
        test_df = pd.read_csv('dataset/UNSW_NB15_testing-set.csv')
        df = pd.concat([train_df, test_df], ignore_index=True)
        logger.info("Dataset loaded successfully")
    except FileNotFoundError:
        logger.error("UNSW-NB15 dataset not found. Please download from https://research.unsw.edu.au/projects/unsw-nb15-dataset")
        raise

    # Handle categorical features
    categorical_cols = ['proto', 'service', 'state']
    df = pd.get_dummies(df, columns=categorical_cols)
    logger.info("Categorical features encoded")

    # Select features and target
    X = df.drop(columns=['id', 'attack_cat', 'label'])
    y = df['attack_cat']

    # Feature selection using Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
    selected_features = feature_importance.nlargest(10).index
    X = X[selected_features]
    logger.info(f"Selected top 10 features: {selected_features.tolist()}")

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Features scaled")

    # Encode target labels
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
    logger.info("Target labels encoded")

    # Save scaler, encoder, and selected features
    with open('savedmodel/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('savedmodel/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open('savedmodel/selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    logger.info("Scaler, encoder, and selected features saved")

    return X_scaled, y_encoded, selected_features

def plot_metrics(anfis):
    """Plot accuracy and loss over epochs"""
    plt.figure(figsize=(10, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(anfis.accuracy_history, label='Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(anfis.loss_history, label='Loss', color='orange')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_metrics.png')
    logger.info("Training metrics plot saved as training_metrics.png")
    plt.close()

def train_model(X, y):
    """Train ANFIS model and save it"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Data split into train and test sets")

    anfis = ANFIS(
        n_membership=2,
        learning_rate=0.01,
        epochs=300,
        max_rules=16,
        n_classes=10
    )
    anfis.fit(X_train, y_train)

    # Plot training metrics
    plot_metrics(anfis)

    # Save the model
    model_params = {
        'centers': anfis.centers,
        'spreads': anfis.spreads,
        'consequent_params': anfis.consequent_params,
        'n_membership': anfis.n_membership,
        'learning_rate': anfis.learning_rate,
        'epochs': anfis.epochs,
        'max_rules': anfis.max_rules,
        'n_classes': anfis.n_classes
    }
    with open('savedmodel/anfis_model.pkl', 'wb') as f:
        pickle.dump(model_params, f)
    logger.info("Model saved to anfis_model.pkl")

    return X_train, X_test, y_train, y_test

def main():
    """Main function to run training"""
    logger.info("Starting training process")
    X, y, selected_features = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_model(X, y)
    logger.info("Training process completed")

if __name__ == "__main__":
    main()