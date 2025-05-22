import numpy as np
import logging

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

class ANFIS:
    def __init__(self, n_membership=2, learning_rate=0.01, epochs=300, max_rules=16, n_classes=10):
        self.n_membership = n_membership
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_rules = max_rules
        self.n_classes = n_classes
        self.centers = None
        self.spreads = None
        self.consequent_params = None
        self.accuracy_history = []
        self.loss_history = []
        logger.info(f"ANFIS initialized with {n_membership} memberships, {max_rules} rules, {n_classes} classes")

    def triangular_membership(self, x, center, spread):
        """Triangular membership function"""
        left = center - spread
        right = center + spread
        return np.maximum(0, np.minimum(
            (x - left) / (spread + 1e-10),
            (right - x) / (spread + 1e-10)
        ))

    def layer1(self, X):
        """Fuzzification layer"""
        n_samples, n_features = X.shape
        memberships = np.zeros((n_samples, n_features, self.n_membership))
        
        for i in range(n_features):
            for j in range(self.n_membership):
                memberships[:, i, j] = self.triangular_membership(
                    X[:, i],
                    self.centers[i, j],
                    self.spreads[i, j]
                )
        logger.debug("Fuzzification layer completed")
        return np.clip(memberships, 1e-10, 1.0)

    def layer2_3(self, memberships):
        """Simplified rule layer with limited rules"""
        n_samples = memberships.shape[0]
        n_features = memberships.shape[1]
        
        feature_memberships = np.mean(memberships, axis=2)
        
        firing_strengths = np.zeros((n_samples, self.max_rules))
        
        for i in range(self.max_rules):
            selected_features = np.random.choice(n_features, size=min(3, n_features), replace=False)
            firing_strengths[:, i] = np.mean(feature_memberships[:, selected_features], axis=1)
        
        sum_firing = np.sum(firing_strengths, axis=1, keepdims=True) + 1e-10
        logger.debug("Rule layer completed")
        return firing_strengths / sum_firing

    def softmax(self, x):
        """Softmax activation for multi-class"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)

    def forward_pass(self, X):
        """Forward pass with softmax activation"""
        memberships = self.layer1(X)
        normalized_firing_strengths = self.layer2_3(memberships)
        
        X_expanded = np.column_stack([X, np.ones(X.shape[0])])
        consequent_outputs = np.zeros((X.shape[0], self.max_rules, self.n_classes))
        
        for i in range(self.max_rules):
            consequent_outputs[:, i, :] = np.dot(X_expanded, self.consequent_params[i].T)
        
        weighted_sum = np.sum(normalized_firing_strengths[:, :, np.newaxis] * consequent_outputs, axis=1)
        
        logger.debug("Forward pass completed")
        return self.softmax(weighted_sum)

    def initialize_parameters(self, X):
        """Initialize parameters"""
        n_features = X.shape[1]
        
        self.centers = np.zeros((n_features, self.n_membership))
        self.spreads = np.zeros((n_features, self.n_membership))
        
        for i in range(n_features):
            feature_min, feature_max = np.min(X[:, i]), np.max(X[:, i])
            feature_range = feature_max - feature_min
            
            self.centers[i] = np.linspace(feature_min, feature_max, self.n_membership)
            self.spreads[i] = np.ones(self.n_membership) * (feature_range / (self.n_membership - 0.5))
        
        self.consequent_params = np.random.normal(0, 0.1, (self.max_rules, self.n_classes, n_features + 1))
        logger.info("Parameters initialized")

    def fit(self, X, y):
        """Train the model"""
        from sklearn.metrics import accuracy_score
        self.initialize_parameters(X)
        
        best_accuracy = 0
        best_params = None
        
        for epoch in range(self.epochs):
            predictions = self.forward_pass(X)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1)
            accuracy = accuracy_score(true_classes, predicted_classes)
            
            # Compute categorical cross-entropy loss
            loss = -np.mean(np.sum(y * np.log(predictions + 1e-10), axis=1))
            
            self.accuracy_history.append(accuracy)
            self.loss_history.append(loss)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (self.centers.copy(), self.spreads.copy(), self.consequent_params.copy())
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            
            error = predictions - y
            
            X_expanded = np.column_stack([X, np.ones(X.shape[0])])
            memberships = self.layer1(X)
            normalized_firing_strengths = self.layer2_3(memberships)
            
            for i in range(self.max_rules):
                for j in range(self.n_classes):
                    gradient = np.dot(
                        error[:, j] * normalized_firing_strengths[:, i],
                        X_expanded
                    )
                    self.consequent_params[i, j] -= self.learning_rate * gradient
            
            for i in range(X.shape[1]):
                for j in range(self.n_membership):
                    center_gradient = np.mean(error[:, 0] * normalized_firing_strengths[:, 0] * 
                                           (X[:, i] - self.centers[i, j]))
                    self.centers[i, j] -= self.learning_rate * center_gradient
                    
                    spread_gradient = np.mean(error[:, 0] * normalized_firing_strengths[:, 0] * 
                                           np.abs(X[:, i] - self.centers[i, j]))
                    self.spreads[i, j] -= self.learning_rate * spread_gradient
                    self.spreads[i, j] = max(self.spreads[i, j], 1e-5)
        
        self.centers, self.spreads, self.consequent_params = best_params
        logger.info(f"Training completed with best accuracy: {best_accuracy:.4f}")

    def predict(self, X):
        """Predict class labels"""
        predictions = self.forward_pass(X)
        logger.debug("Prediction completed")
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        """Predict class probabilities"""
        probabilities = self.forward_pass(X)
        logger.debug("Probability prediction completed")
        return probabilities