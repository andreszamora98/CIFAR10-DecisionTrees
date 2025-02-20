from data_handler import DataHandler
from decision_tree_model import DecisionTreeModel

if __name__ == "__main__":
    # Cargar y preprocesar los datos
    x_train, y_train, x_test, y_test = DataHandler.load_data()
    x_train, x_test = DataHandler.preprocess_data(x_train, x_test, n_components=100)
    
    # Crear y entrenar el modelo
    model = DecisionTreeModel(max_depth=15)
    model.train(x_train, y_train)
    
    # Evaluar y graficar el árbol de decisión
    model.evaluate(x_test, y_test)
    model.plot_tree()
