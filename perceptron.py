import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandastable import Table, TableModel
import tkinter as tk

class Perceptron:
    def __init__(self):
        self.error_by_execution = []
        self.activation_function = self._fx  # Default activation function

    def perceptron_train(self, x_values, y_values, learning_rate, epochs, tolerance, activation, normalize):
        if normalize:
            # Normalización Yd
            min_yd = np.min(y_values)
            range_yd = np.max(y_values) - min_yd
            y_values = (y_values - min_yd) / range_yd

            # Normalización de X
            x_min = np.min(x_values, axis=0)
            range_x = np.max(x_values, axis=0) - x_min
            x_values = -1 + 2 * (x_values - x_min) / range_x

        error_by_epoch = []
        weights_by_epoch = []
        W = np.random.rand(x_values.shape[1] + 1)  # Inicializa los pesos
        bias = np.random.uniform(-1, 1)  # Initializing bias correctly as a single value
        self.activation_function = activation

        for epoch in range(epochs):
            X = np.column_stack(([1] * x_values.shape[0], x_values))
            u = np.dot(X, W) + bias
            yc_n = self.activation_function(u)  # Use the selected activation function
            training_error = y_values - yc_n
            delta_w = learning_rate * np.dot(X.T, training_error)
            W += delta_w
            bias += learning_rate * np.sum(training_error)  # Updating bias
            weights_by_epoch.append(W.copy())
            norma_error = np.linalg.norm(training_error)
            error_by_epoch.append(norma_error)
            if norma_error <= tolerance:
                print(f"Epoch: {epoch}, Error: {norma_error}")
                break

        if normalize:
            y_values = self._denormalize(y_values, range_yd, min_yd)
            yc_n = self._denormalize(yc_n, range_yd, min_yd)

        self.error_by_execution.append(error_by_epoch)
        self._draw_results(y_values, yc_n, error_by_epoch, weights_by_epoch)
        self.draw_table(weights_by_epoch, norma_error, yc_n, y_values)

    def _fx(self, u):
        return np.tanh(u)
    
    def _fx_escalon(self, u):
        return np.where(u >= 0, 1, 0)

    def _fx_linear(self, u):
        return u

    def _sigmoid(self, u):
        return 1 / (1 + np.exp(-u))

    def _denormalize(self, values, range_yd, min_yd):
        return values * range_yd + min_yd

    def _draw_results(self, y_values, yc, error_by_epoch, weights_by_epoch):
        self._draw_ys(y_values, yc)
        self._draw_error(error_by_epoch)
        self._draw_weights(weights_by_epoch)

    def _draw_ys(self, y_values, yc):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(y_values)), y_values, color="green", label="Y deseada")
        plt.plot(range(len(y_values)), yc, color="purple", label="Cálculo de la salida")
        plt.ylabel("Y")
        plt.xlabel("ID de Muestra")
        plt.legend()
        plt.show(block=False)

    def _draw_error(self, error_by_epoch):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(error_by_epoch)), error_by_epoch, color='red', label="Error")
        plt.ylabel("Error")
        plt.xlabel("Épocas")
        plt.legend()
        plt.show(block=False)

    def _draw_weights(self, weights_by_epoch):
        plt.figure(figsize=(10, 5))
        epochs = range(len(weights_by_epoch))
        for i in range(len(weights_by_epoch[0])):
            weight_i_by_epoch = [weights[i] for weights in weights_by_epoch]
            plt.plot(epochs, weight_i_by_epoch, label=f"W{i}")
        plt.ylabel("W")
        plt.xlabel("Época")
        plt.title("Evolución de los Pesos por Época")
        plt.legend()
        plt.show()

    def draw_table(self, weights_by_epoch, norma_error, yc_values, yd_values):
        data = {
            'W': [', '.join(map(str, weights_by_epoch[-2]))] + [''] * (len(yc_values) - 1),
            'Error': [f"{norma_error}"] + [''] * (len(yc_values) - 1),
            'Yc': yc_values,
            'Yd': yd_values
        }
        df = pd.DataFrame(data)
        
        table_window = tk.Toplevel()
        table_window.title("Resultados del Perceptron")
        
        frame = tk.Frame(table_window)
        frame.pack(fill='both', expand=True)
        
        pt = Table(frame, dataframe=df)
        pt.show()
