import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron

class PerceptronApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Perceptron Trainer")

        # Learning rate
        self.lr_label = tk.Label(master, text="Tasa de Aprendizaje:")
        self.lr_label.grid(row=0, column=0)
        self.learning_rate = tk.Entry(master)
        self.learning_rate.grid(row=0, column=1)

        # Tolerance
        self.tolerance_label = tk.Label(master, text="Tolerancia de Error:")
        self.tolerance_label.grid(row=1, column=0)
        self.tolerance = tk.Entry(master)
        self.tolerance.grid(row=1, column=1)

        # Number of epochs
        self.epochs_label = tk.Label(master, text="Número de Épocas:")
        self.epochs_label.grid(row=2, column=0)
        self.epochs = tk.Entry(master)
        self.epochs.grid(row=2, column=1)

        # Dataset selection
        self.dataset_label = tk.Label(master, text="Seleccionar Dataset:")
        self.dataset_label.grid(row=3, column=0)
        self.dataset_combo = ttk.Combobox(master, values=['c:/Users/angel/Documents/IA/Neurona/Propio/221233_nuevo.xlsx', 'C:/Users/angel/Documents/IA/Neurona/Propio/203140.csv', 'dataset3.xlsx'])
        self.dataset_combo.grid(row=3, column=1)

        # Activation function selection
        self.activation_label = tk.Label(master, text="Función de Activación:")
        self.activation_label.grid(row=4, column=0)
        self.activation_combo = ttk.Combobox(master, values=['Step Function', 'Sigmoid', 'Tanh', 'Linear'])
        self.activation_combo.grid(row=4, column=1)

        # Normalization selection
        self.normalization_label = tk.Label(master, text="Normalizar:")
        self.normalization_label.grid(row=5, column=0)
        self.normalization_combo = ttk.Combobox(master, values=['Sí', 'No'])
        self.normalization_combo.grid(row=5, column=1)

        # Start button
        self.start_button = tk.Button(master, text="Iniciar Entrenamiento", command=self.start_training)
        self.start_button.grid(row=6, columnspan=2)

        # Perceptron instance
        self.perceptron = Perceptron()

    def start_training(self):
        learning_rate = float(self.learning_rate.get())
        tolerance = float(self.tolerance.get())
        epochs = int(self.epochs.get())
        dataset_path = self.dataset_combo.get()

        # Load dataset based on file extension
        if dataset_path.endswith('.xlsx'):
            dataset = pd.read_excel(dataset_path)
        elif dataset_path.endswith('.csv'):
            dataset = pd.read_csv(dataset_path, delimiter=';')
        else:
            print("Unsupported file format")
            return

        # Elimina espacios en blanco de los nombres de las columnas
        dataset.columns = dataset.columns.str.strip()
        
        x_values = dataset.iloc[:, :-1].values
        y_values = dataset.iloc[:, -1].values

        # Select activation function
        activation_function = self.activation_combo.get()
        if activation_function == 'Step Function':
            self.perceptron.activation_function = self.perceptron._fx_escalon
        elif activation_function == 'Sigmoid':
            self.perceptron.activation_function = self.perceptron._sigmoid
        elif activation_function == 'Tanh':
            self.perceptron.activation_function = self.perceptron._fx
        elif activation_function == 'Linear':
            self.perceptron.activation_function = self.perceptron._fx_linear

        # Determine normalization
        normalize = self.normalization_combo.get() == 'Sí'

        # Train perceptron
        self.perceptron.perceptron_train(x_values, y_values, learning_rate, epochs, tolerance, self.perceptron.activation_function, normalize)

if __name__ == "__main__":
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()
