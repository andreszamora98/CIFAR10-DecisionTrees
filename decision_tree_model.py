import os
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import seaborn as sns

class DecisionTreeModel:
    def __init__(self, max_depth=15, use_random_forest=False, use_xgboost=False, model_path=None):
        """
        Inicializa el modelo (Árbol de Decisión, Random Forest o XGBoost).
        """
        self.use_random_forest = use_random_forest
        self.use_xgboost = use_xgboost
        self.classes = ["airplane", "automobile", "bird", "cat", "deer", 
                        "dog", "frog", "horse", "ship", "truck"]
        self.model = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            if self.use_random_forest:
                self.model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=max_depth,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
            elif self.use_xgboost:
                self.model = XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=max_depth,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            else:
                self.model = DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=max_depth,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )

    def train(self, x_train, y_train):
        """Entrena el modelo."""
        self.model.fit(x_train, y_train)
        print(f"{'XGBoost' if self.use_xgboost else 'Random Forest' if self.use_random_forest else 'Árbol de decisión'} entrenado con éxito.")

    def evaluate(self, x_test, y_test):
        """Evalúa el modelo y muestra métricas clave."""
        y_pred = self.model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", round(acc * 100, 2), "%")
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=self.classes))

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        print(f"Matriz de confusión guardada en {output_dir}/confusion_matrix.png")
        plt.close()

    def plot_tree(self):
        """Genera y guarda el gráfico del árbol de decisión (solo si no es Random Forest o XGBoost)."""
        if self.use_random_forest or self.use_xgboost:
            print("La visualización del árbol no está disponible para Random Forest o XGBoost.")
            return
        
        plt.figure(figsize=(12, 8))
        plot_tree(self.model, max_depth=3, feature_names=[f"PC{i}" for i in range(1, 151)],
                  class_names=self.classes, filled=True)

        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)

        image_path = os.path.join(output_dir, "decision_tree.png")
        plt.savefig(image_path, dpi=300)
        print(f"Árbol de decisión guardado en: {image_path}")
        plt.close()

    def optimize_hyperparameters(self, x_train, y_train, use_random_search=False):
        """Usa GridSearchCV o RandomizedSearchCV para optimizar los hiperparámetros."""
        if self.use_random_forest:
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [20, 50, 70, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        elif self.use_xgboost:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        else:
            param_grid = {
                'max_depth': [10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }

        if use_random_search:
            search = RandomizedSearchCV(self.model, param_distributions=param_grid, n_iter=10, cv=5, n_jobs=-1, verbose=2)
        else:
            search = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1, verbose=2)

        search.fit(x_train, y_train)
        print("Mejores parámetros encontrados:", search.best_params_)
        self.model = search.best_estimator_

    def cross_validate(self, x_train, y_train, cv=5):
        """Aplica validación cruzada para evaluar la estabilidad del modelo."""
        scores = cross_val_score(self.model, x_train, y_train, cv=cv)
        print(f"Precisión media tras {cv}-fold Cross-Validation: {scores.mean() * 100:.2f}%")

    def save_model(self, filename="trained_model.pkl"):
        """Guarda el modelo entrenado en un archivo."""
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"Modelo guardado en {filename}")

    def load_model(self, filename):
        """Carga un modelo previamente guardado."""
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.model = pickle.load(file)
            print(f"Modelo cargado desde {filename}")
        else:
            print(f"El archivo {filename} no existe.")
