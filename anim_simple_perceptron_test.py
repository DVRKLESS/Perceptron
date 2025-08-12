from random import uniform, shuffle
import matplotlib.pyplot as plt
import os
import pickle

# =============== ГЕНЕРАТОР ДАННЫХ ===============
def generate_data(func=None, n_points=500, x_range=(0, 100), y_range=(-10, 10)):
    data = []
    for _ in range(n_points):
        a = uniform(*x_range)
        b = uniform(*y_range)
        if func is None:
            label = 0 if a < (x_range[0] + x_range[1]) / 2 else 1
        else:
            label = 0 if b < func(a) else 1
        data.append((a, b, label))
    shuffle(data)
    return data

# Пример с функцией — закомментируй если не надо
data = generate_data(func=lambda x: 0.5*x - 10)
# Пример без функции (классы просто по X)
# data = generate_data()

x = [[a, b] for a, b, _ in data]
y = [label for _, _, label in data]

# =============== ПЕРСЕПТРОН ===============
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = [uniform(-1, 1) for _ in range(input_size)]
        self.bias = uniform(-1, 1)
        self.lr = learning_rate

    def train(self, x, y, epochs=25, save_frames=True):
        if save_frames and not os.path.exists("frames"):
            os.makedirs("frames")

        for epoch in range(epochs):
            for inputs, label in zip(x, y):
                prediction = self.predict(inputs)
                error = label - prediction
                for i in range(len(self.weights)):
                    self.weights[i] += self.lr * error * inputs[i]
                self.bias += self.lr * error

            if save_frames:
                self.plot_decision_boundary(x, y, epoch + 1, save_path=f"frames/epoch_{epoch+1:03}.png")

        self.save_model("perceptron.pkl")

    def predict(self, inputs):
        total = sum(w * xi for w, xi in zip(self.weights, inputs)) + self.bias
        return 1 if total >= 0 else 0

    def plot_decision_boundary(self, x, y, epoch, save_path=None):
        plt.clf()
        for (xi, yi), label in zip(x, y):
            plt.scatter(xi, yi, color='red' if label == 0 else 'blue', marker='o' if label == 0 else 'x')

        if self.weights[1] != 0:
            x_vals = [min(pt[0] for pt in x)-10, max(pt[0] for pt in x)+10]
            y_vals = [-(self.weights[0]*xv + self.bias)/self.weights[1] for xv in x_vals]
            plt.plot(x_vals, y_vals, 'k-')
            eq_text = f"y = {-self.weights[0]/self.weights[1]:.2f}x + {-self.bias/self.weights[1]:.2f}"
            plt.text(x_vals[0], y_vals[0], eq_text, fontsize=9, color='green')

        plt.title(f"Epoch {epoch}")
        plt.xlim(-10, 110)
        plt.ylim(-20, 20)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.pause(0.001)

    def save_model(self, filename):
        with open(filename, "wb") as f:
            pickle.dump((self.weights, self.bias), f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.weights, self.bias = pickle.load(f)

# =============== РАБОТА ===============
plt.ion()
p = Perceptron(input_size=2, learning_rate=0.01)
p.train(x, y, epochs=50)
plt.ioff()

# =============== ПРОВЕРКА ТОЧЕК ПОСЛЕ ОБУЧЕНИЯ ===============
while True:
    try:
        inp = input("Введите точку X Y (или 'exit'): ")
        if inp.strip().lower() == 'exit':
            break
        px, py = map(float, inp.split())
        pred = p.predict([px, py])
        print(f"Класс: {pred}")
        plt.scatter(px, py, color='green' if pred == 0 else 'purple', marker='D', s=100)
        plt.draw()
    except Exception as e:
        print("Ошибка ввода:", e)

plt.show()
