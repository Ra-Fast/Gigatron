import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix,auc
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from scipy.stats import mutual_info_score

class ModelGenerator:
    def __init__(self, num_dimensions, num_classes, bins=10, drop_fraction=10,num_samples=1000, random_state=42):
        self.num_dimensions = num_dimensions
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.random_state = random_state
        self.bins=bins
        self.drop_fraction=drop_fraction
        self.df = None  # Internal variable to store the DataFrame
        self.model = None  # Internal variable to store the TensorFlow model
        self.history = None  # Internal variable to store training history
        self.y_pred = None  # Internal variable to store predicted probabilities
        self.y_pred_binary = None  # Internal variable to store binary predictions
        self.y_test_binary = None  # Internal variable to store binary ground truth labels
        self.X = None  # Internal variable to store feature matrix
        self.y = None  # Internal variable to store target vector
        self.X_init = None  # Internal variable to store feature matrix - Initial values
        self.y_init = None  # Internal variable to store target vector - Initial values

        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    def generate_synthetic_data(self, plot_scatter=False):
        # Generate synthetic data with specified dimensions and classes
        X, y = make_classification(
            n_samples=self.num_samples,
            n_features=self.num_dimensions,
            n_informative=self.num_dimensions,
            n_redundant=0,
            n_clusters_per_class=1,  # Use 1 cluster per class
            n_classes=self.num_classes,
            random_state=self.random_state
        )

        # Scale values between 0 and 1
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Create a DataFrame
        columns = [f"feature_{i}" for i in range(self.num_dimensions)]
        self.df = pd.DataFrame(X_scaled, columns=columns)
        self.df['target'] = y

        # Store feature matrix and target vector
        self.X = X_scaled
        self.y = y

        # Store initial values
        self.X_init=self.X
        self.y_init=self.y
        # Display scatter plot if plot_scatter is True
        if plot_scatter:
            self.plot_scatter()

        return self.X, self.y, self.X_init, self.y_init

    def drop_rows_randomly(self):
        if self.X is not None and self.y is not None:
            num_rows_to_drop = int(self.X.shape[0] * self.drop_fraction)
            indices_to_drop = np.random.choice(self.X.shape[0], num_rows_to_drop, replace=False)

            self.X = np.delete(self.X, indices_to_drop, axis=0)
            self.y = np.delete(self.y, indices_to_drop, axis=0)

            print(f"Dropped {num_rows_to_drop} rows randomly.")

        else:
            print("Feature matrix (X) or target vector (y) not available. Call generate_synthetic_data first.")

    def calculate_entropy(self, bins):
        entropy_values = []
        for column in self.X.columns:
            hist, bin_edges = np.histogram(self.X[column], bins=bins, range=(0, 1), density=True)
            entropy_values.append(entropy(hist, base=2))
        return entropy_values
        
    def calculate_mutual_information(self):
        mutual_info_values = []
        for i in range(len(self.X.columns)):
            for j in range(i + 1, len(self.X.columns)):
                mi = mutual_info_score(self.X.iloc[:, i], self.X.iloc[:, j], bins=self.bins)
                mutual_info_values.append(mi)
        return mutual_info_values
    
    def calculate_relative_entropy(self):
        relative_entropy_values = []
        for i in range(len(self.X_init.columns)):
            kl_distance = entropy(self.X_init.iloc[:, i], qk=self.X.iloc[:, i], base=2)
            relative_entropy_values.append(kl_distance)
        return relative_entropy_values

    def plot_histogram(self, num_bins=10):
        if self.df is not None:
            # Plot histograms in separate graphs for each feature
            num_features = self.num_dimensions
            num_rows = num_features // 2 if num_features % 2 == 0 else num_features // 2 + 1
            #fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))
            fig, axes = plt.subplots(num_rows, 2, figsize=(12, 12))
            #fig, axes = plt.subplots(ncols=2, nrows=num_rows, figsize=(15,15))
            fig.suptitle('Feature Value Distribution', y=1.02)

            for i, column in enumerate(self.df.columns[:-1]):  # Exclude the target column
                row, col = divmod(i, 2)
                # Data distribution
                if num_features==2:
                    sns.histplot(ax=axes[row+col],data=self.df,x=column,hue='target',kde=True,bins=num_bins)
                    axes[row+col].set_title(column)
                    axes[row+col].set_xlabel('Feature Value')
                    axes[row+col].set_ylabel('Frequency')
                else:
                    sns.histplot(ax=axes[row, col],data=self.df,x=column,hue='target',kde=True, bins=num_bins)
                    axes[row, col].set_title(column)
                    axes[row, col].set_xlabel('Feature Value')
                    axes[row, col].set_ylabel('Frequency')

            plt.tight_layout()
            plt.show()
        else:
            print("DataFrame not generated. Call generate_synthetic_data first.")

    def plot_scatter(self):
        if self.df is not None and self.num_dimensions >= 2:
            # Plot scatter plot using Seaborn
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='feature_0', y='feature_1', hue='target', data=self.df, palette='viridis')
            plt.title('Scatter Plot of Synthetic Data')
            plt.xlabel('Feature 0')
            plt.ylabel('Feature 1')
            plt.show()
        else:
            print("DataFrame not generated or insufficient dimensions for scatter plot.")

    def build_neural_network(self, num_layers=2, num_neurons_per_layer=10):
        if self.df is not None:
            # Build a neural network using TensorFlow
            model = Sequential()
            model.add(Dense(num_neurons_per_layer, input_dim=self.num_dimensions, activation='relu'))

            for _ in range(num_layers - 1):
                model.add(Dense(num_neurons_per_layer, activation='relu'))

            if self.num_classes == 2:
                # For binary classification, use sigmoid activation
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                model.summary()
            else:
                # For multi-class classification, use softmax activation
                model.add(Dense(self.num_classes, activation='softmax'))
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.summary()

            self.model = model
        else:
            print("DataFrame not generated. Call generate_synthetic_data first.")

    def train_neural_network(self, num_epochs=10, batch_size=32):
        if self.df is not None and self.model is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                self.df.drop('target', axis=1), self.df['target'], test_size=0.2, random_state=self.random_state)

            history = self.model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0,
                                     validation_data=(X_test, y_test))

            self.history = history.history

            # Get predicted probabilities and binary predictions
            self.y_pred = self.model.predict(X_test)
            self.y_pred_binary = (self.y_pred > 0.5) if self.num_classes == 2 else self.y_pred.argmax(axis=1)
            self.y_test_binary = (y_test > 0.5) if self.num_classes == 2 else y_test

            # Return X_train, X_test, y_train, and y_pred
            return X_train, X_test, y_train, y_test,self.y_pred,self.y_pred_binary

        else:
            print("DataFrame or model not generated. Call generate_synthetic_data and build_neural_network first.")
            return None, None, None, None

    def plot_confusion_matrix(self):
        if self.y_test_binary is not None and self.y_pred_binary is not None:
            cm = confusion_matrix(self.y_test_binary, self.y_pred_binary)

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Predicted 0', 'Predicted 1'],
                        yticklabels=['Actual 0', 'Actual 1'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
        else:
            print("Ground truth or predictions not available. Call train_neural_network first.")

    def plot_roc_curve(self):
        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(self.y_test_binary, self.y_pred)
            auc_value = roc_auc_score(self.y_test_binary, self.y_pred)

            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_value:.2f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.show()
        else:
            print("ROC curve and AUC are applicable for binary classification only.")

    def performance_measurement(self):
        # Performance
        print('accuracy:', np.round(accuracy_score(self.y_test_binary, self.y_pred_binary),5))
        print('precision:', np.round(precision_score(self.y_test_binary, self.y_pred_binary),5))
        print('recall:', np.round(recall_score(self.y_test_binary, self.y_pred_binary),5))

        # auc 
        fpr, tpr, _ = roc_curve(self.y_test_binary, self.y_pred_binary)  # obtención de las tasas de falsos y verdaderos positivos
        print('auc:', np.round(auc(fpr, tpr),5))

    def plot_training_history(self):
        if self.history is not None:
            plt.figure(figsize=(12, 6))

            # Plot training & validation accuracy values
            plt.subplot(1, 2, 1)
            plt.plot(self.history['accuracy'])
            if 'val_accuracy' in self.history:
                plt.plot(self.history['val_accuracy'])
                plt.legend(['Train', 'Validation'], loc='upper left')
            else:
                plt.legend(['Train'], loc='upper left')
            plt.title('Model accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')

            # Plot training & validation loss values
            plt.subplot(1, 2, 2)
            plt.plot(self.history['loss'])
            if 'val_loss' in self.history:
                plt.plot(self.history['val_loss'])
                plt.legend(['Train', 'Validation'], loc='upper left')
            else:
                plt.legend(['Train'], loc='upper left')
            plt.title('Model loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.tight_layout()
            plt.show()
        else:
            print("Training history not available. Call train_neural_network first.")
'''
# Example usage
num_dimensions = 5        # Change this to your desired number of dimensions
num_classes = 2           # Change this to your desired number of classes (2 for binary classification)
num_layers = 2            # Change this to your desired number of layers
num_neurons_per_layer = 10  # Change this to your desired number of neurons per layer
num_epochs = 10           # Change this to your desired number of epochs
batch_size = 32           # Change this to your desired batch size
drop_fraction = 0.2       # Change this to the desired fraction of rows to drop randomly

model_instance = ModelGenerator(num_dimensions, num_classes)
X, y = model_instance.generate_synthetic_data(plot_scatter=True)
model_instance.drop_rows_randomly(drop_fraction)
model_instance.plot_histogram(num_bins=20)
model_instance.build_neural_network(num_layers, num_neurons_per_layer)
X_train, X_test, y_train, y_pred = model_instance.train_neural_network(num_epochs, batch_size)

# Now X_train, X_test, y_train, and y_pred are accessible for further analysis or evaluation.
'''
