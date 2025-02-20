import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

class DecisionTreeModel:
    def __init__(self, max_depth=15):
        self.model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=None
        )
        self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def evaluate(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", acc)
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=self.classes))
    
    def plot_tree(self):
        plt.figure(figsize=(12, 8))
        plot_tree(self.model, max_depth=3, feature_names=[f"PC{i}" for i in range(1, 101)], class_names=self.classes, filled=True)
        
        # Crear una carpeta para guardar la imagen
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar la imagen en la carpeta creada
        image_path = os.path.join(output_dir, "decision_tree.png")
        plt.savefig(image_path, dpi=300)
        print(f"El árbol de decisión ha sido guardado en: {image_path}")