import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix,auc, f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression

class ModelGenerator:
    def __init__(self, num_dimensions, num_classes, bins=20,num_samples=1000, random_state=42):
        self.num_dimensions = num_dimensions
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.random_state = random_state
        self.bins=bins
        #self.drop_fraction=drop_fraction
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
            #plt.scatter(self.X[:,0],self.X[:,1], c=self.y)

        return self.X, self.y

    def drop_rows_randomly(self,drop_fraction):
        if self.X_init is not None and self.y is not None:
            num_rows_to_drop = int(self.X_init.shape[0] * drop_fraction)
            indices_to_drop = np.random.choice(self.X_init.shape[0], num_rows_to_drop, replace=False)

            self.X = np.delete(self.X_init, indices_to_drop, axis=0)
            self.y = np.delete(self.y_init, indices_to_drop, axis=0)
            
            # Update DataFrame
            self.df.drop(indices_to_drop,inplace=True)

            print(f"Dropped {num_rows_to_drop} rows randomly.")

        else:
            print("Feature matrix (X) or target vector (y) not available. Call generate_synthetic_data first.")

    def calculate_entropy(self):
        entropy_values = []
        for column in self.X.T:
            hist, bin_edges = np.histogram(column, bins=self.bins, range=(0, 1), density=True)
            entropy_values.append(entropy(hist, base=2))
        return sum(entropy_values)
        
    def calculate_mutual_information(self):
        tam=self.X.shape[1]
        mutual_info_values = np.zeros((tam,tam))
        for i in range(tam):
            for j in range(i + 1, tam):
                    mutual_info_values[i,j]= mutual_info_regression(self.X[:, i:i+1], self.X[:, j])[0]
        return mutual_info_values
    
    def calculate_relative_entropy(self):
        relative_entropy_values = []
        epsilon=0.000001
        for i in range(self.X_init.shape[1]):
            # Calculate probs
            prob_1,_=np.histogram(self.X_init[:,i],bins=self.bins,range=(0, 1), density=True)
            prob_2,_=np.histogram(self.X[:,i],bins=self.bins,range=(0, 1), density=True)


            #kl_distance = entropy(prob_1, prob_2, base=2)
            # Check for null values of prob_2
            for i in range(prob_2.shape[0]):
                if prob_2[i]==0:
                    prob_2[i]=epsilon

            kl_distance=np.sum(prob_1*np.log2(prob_1/prob_2))


            relative_entropy_values.append(kl_distance)
        return relative_entropy_values
    

    def plot_histogram(self):
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
                    sns.histplot(ax=axes[row+col],data=self.df,x=column,hue='target',kde=True,bins=self.bins)
                    axes[row+col].set_title(column)
                    axes[row+col].set_xlabel('Feature Value')
                    axes[row+col].set_ylabel('Frequency')
                else:
                    sns.histplot(ax=axes[row, col],data=self.df,x=column,hue='target',kde=True, bins=self.bins)
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

    def build_neural_network(self, num_layers=2, num_neurons_per_layer=10,model_summary=False):
        model=None
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
                if model_summary:
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
        accuracy=np.round(accuracy_score(self.y_test_binary, self.y_pred_binary),5)
        precision=np.round(precision_score(self.y_test_binary, self.y_pred_binary),5)
        recall=np.round(recall_score(self.y_test_binary, self.y_pred_binary),5)
        print('accuracy:', accuracy)
        print('precision:', precision)
        print('recall:', recall)

        # auc 
        fpr, tpr, _ = roc_curve(self.y_test_binary, self.y_pred_binary)
        # obtención de las tasas de falsos y verdaderos positivos
        auc_value=np.round(auc(fpr, tpr),5)
        print('auc:', auc_value)

        # F1 Score
        f1score=np.round(f1_score(self.y_test_binary, self.y_pred_binary),5)
        print('F1_Score:', f1score)

        return accuracy,precision,recall,auc_value, f1score

    def performance_measurement_ground_truth(self):
        # Get predicted probabilities and binary predictions
        self.y_pred_init = self.model.predict(self.X_init)
        self.y_pred_init_binary = (self.y_pred_init > 0.5) if self.num_classes == 2 else self.y_pred_init.argmax(axis=1)
        # Performance
        accuracy=np.round(accuracy_score(self.y_pred_init_binary , self.y_init),5)
        precision=np.round(precision_score(self.y_pred_init_binary , self.y_init),5)
        recall=np.round(recall_score(self.y_pred_init_binary , self.y_init),5)
        print('accuracy:', accuracy)
        print('precision:', precision)
        print('recall:', recall)

        # auc 
        fpr, tpr, _ = roc_curve(self.y_pred_init_binary , self.y_init)  
        # obtención de las tasas de falsos y verdaderos positivos
        auc_value=np.round(auc(fpr, tpr),5)
        print('auc:', auc_value)

        # F1 Score
        f1score=np.round(f1_score(self.y_pred_init_binary , self.y_init),5)
        print('F1_Score:', f1score)


        return accuracy,precision,recall,auc_value,f1score


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

if __name__ == "__main__":
    # Example usage
    num_dimensions = 5        # Change this to your desired number of dimensions
    num_classes = 2           # Change this to your desired number of classes (2 for binary classification)
    num_layers = 2            # Change this to your desired number of layers
    num_neurons_per_layer = 10  # Change this to your desired number of neurons per layer
    num_epochs = 5           # Change this to your desired number of epochs
    batch_size = 32           # Change this to your desired batch size
    drop_fraction = 0.2       # Change this to the desired fraction of rows to drop randomly
    num_samples=100000

    model_instance = ModelGenerator(num_dimensions, num_classes)
    # Now X_train, X_test, y_train, and y_pred are accessible for further analysis or evaluation.
    X, y = model_instance.generate_synthetic_data(plot_scatter=True)
    model_instance.plot_histogram()
    
    # Drop 10%
    model_instance_10=ModelGenerator(num_dimensions, num_classes)
    _,_=model_instance_10.generate_synthetic_data(plot_scatter=False)
    model_instance_10.drop_rows_randomly(drop_fraction)
    model_instance_10.plot_histogram()

    # Amount of information
    print(model_instance.calculate_entropy())
    print(model_instance.calculate_mutual_information())
    print(model_instance.calculate_relative_entropy())
    
    # Train neural network
    model_instance.build_neural_network(num_layers, num_neurons_per_layer)
    X_train, X_test, y_train, y_test,y_pred,y_pred_binary = model_instance.train_neural_network(num_epochs, batch_size)
    model_instance.performance_measurement()

    # Amount of information
    print(model_instance.calculate_entropy())
    print(model_instance.calculate_mutual_information())
    print(model_instance.calculate_relative_entropy())
    # Train neural network
    model_instance.build_neural_network(num_layers, num_neurons_per_layer)
    X_train, X_test, y_train, y_test,y_pred,y_pred_binary = model_instance.train_neural_network(num_epochs, batch_size)
    model_instance.performance_measurement()
    model_instance.performance_measurement_ground_truth()


