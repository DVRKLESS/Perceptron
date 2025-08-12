from random import uniform, shuffle
import matplotlib.pyplot as plt

# =============== ДАННЫЕ ===============
x = []
y = []
data = []
for _ in range(500):  # класс 0
	a = uniform(0, 50)
	b = uniform(-10, 11)
	data.append((a, b, 0))

for _ in range(500):  # класс 1
	a = uniform(50, 101)
	b = uniform(-10, 11)
	data.append((a, b, 1))

shuffle(data)

for a, b, label in data:
	x.append([a, b])
	y.append(label)


# =============== ПЕРСЕПТРОН ===============
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = [uniform(-1, 1) for _ in range(input_size)]
        self.bias = uniform(-1, 1)
        self.lr = learning_rate

    def train(self, x, y, epochs=25):
        for epoch in range(epochs):
            for inputs, label in zip(x, y):
                prediction = self.predict(inputs)
                error = label - prediction
                for i in range(len(self.weights)):
                    self.weights[i] += self.lr * error * inputs[i]
                self.bias += self.lr * error

            # Визуализация после каждой эпохи
            self.plot_decision_boundary(x, y, epoch + 1)

    def predict(self, inputs):
        total = sum(w * xi for w, xi in zip(self.weights, inputs)) + self.bias
        return 1 if total >= 0 else 0

    def plot_decision_boundary(self, x, y, epoch):
        plt.clf()
        # Рисуем точки
        for (xi, yi), label in zip(x, y):
            if label == 0:
                plt.scatter(xi, yi, color='red', marker='o')
            else:
                plt.scatter(xi, yi, color='blue', marker='x')

        # Рисуем разделяющую линию
        if self.weights[1] != 0:
            x_vals = [0, 200]
            y_vals = [-(self.weights[0] * xv + self.bias) / self.weights[1] for xv in x_vals]
            plt.plot(x_vals, y_vals, 'k-')

        plt.title(f"Epoch {epoch}")
        plt.xlim(-10, 300)
        plt.ylim(-150, 150)
        plt.pause(10**(-3))


# =============== РАБОТА ===============
plt.ion()  # включаем интерактивный режим
p = Perceptron(input_size=2, learning_rate=100)
p.train(x, y, epochs=200)
print(p.predict([50, 20]))
plt.ioff()
plt.show()