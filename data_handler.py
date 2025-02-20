import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class DataHandler:
    @staticmethod
    def load_batch(file):
        """Carga un batch del dataset en formato pickle."""
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        return batch
    
    @staticmethod
    def load_data(path="cifar-10-batches-py"):
        """Carga CIFAR-10 desde archivos locales en lugar de descargarlo con TensorFlow."""
        x_train, y_train = [], []
        for i in range(1, 6):  # Cargar los 5 lotes de entrenamiento
            batch = DataHandler.load_batch(os.path.join(path, f"data_batch_{i}"))
            x_train.append(batch[b"data"])
            y_train += batch[b"labels"]
        
        x_train = np.vstack(x_train)
        y_train = np.array(y_train)
        
        # Cargar el batch de prueba
        test_batch = DataHandler.load_batch(os.path.join(path, "test_batch"))
        x_test = np.array(test_batch[b"data"])
        y_test = np.array(test_batch[b"labels"])
        
        # Dividir datos en entrenamiento y prueba (80-20)
        x_train, x_test, y_train, y_test = train_test_split(
            x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
        return x_train, y_train, x_test, y_test
    
    @staticmethod
    def preprocess_data(x_train, x_test, n_components=150):
        pca = PCA(n_components=n_components)
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca = pca.transform(x_test)
        return x_train_pca, x_test_pca
