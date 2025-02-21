import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler

class DataHandler:
    @staticmethod
    def load_batch(file):
        """Carga un batch del dataset en formato pickle."""
        if not os.path.exists(file):
            raise FileNotFoundError(f"El archivo {file} no se encontró.")
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        return batch

    @staticmethod
    def load_data(path="cifar-10-batches-py", use_existing=False):
        """
        Carga el dataset CIFAR-10 desde archivos locales. Permite cargar datos existentes.
        """
        if use_existing and os.path.exists("cached_data.pkl"):
            with open("cached_data.pkl", 'rb') as file:
                x_train, y_train, x_test, y_test = pickle.load(file)
                print("Datos cargados desde caché.")
                return x_train, y_train, x_test, y_test

        x_train, y_train = [], []
        try:
            for i in range(1, 6):
                batch = DataHandler.load_batch(os.path.join(path, f"data_batch_{i}"))
                x_train.append(batch[b"data"])
                y_train += batch[b"labels"]

            x_train = np.vstack(x_train)
            y_train = np.array(y_train)

            test_batch = DataHandler.load_batch(os.path.join(path, "test_batch"))
            x_test = np.array(test_batch[b"data"])
            y_test = np.array(test_batch[b"labels"])

            x_train, x_test, y_train, y_test = train_test_split(
                x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )

            with open("cached_data.pkl", 'wb') as file:
                pickle.dump((x_train, y_train, x_test, y_test), file)

            print("Datos cargados correctamente.")
            return x_train, y_train, x_test, y_test

        except Exception as e:
            print(f"Error al cargar los datos: {e}")
            return None, None, None, None

    @staticmethod
    def preprocess_data(x_train, x_test, n_components=500, scale_method="robust"):
        """
        Preprocesa los datos aplicando PCA y escalado robusto o MinMax.
        """
        if scale_method == "robust":
            scaler = RobustScaler()
        elif scale_method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Escalador no soportado. Usa 'robust' o 'minmax'.")

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        pca = PCA(n_components=n_components)
        x_train_pca = pca.fit_transform(x_train)
        x_test_pca = pca.transform(x_test)

        explained_variance = np.sum(pca.explained_variance_ratio_) * 100
        print(f"PCA aplicado. Variabilidad explicada: {explained_variance:.2f}%")

        return x_train_pca, x_test_pca
