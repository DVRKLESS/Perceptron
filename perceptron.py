from random import uniform, shuffle
# import resource


# resource.setrlimit(resource.RLIMIT_CPU, (50, 50))


# ===============GENERATOR===============
x = []
y = []
data = []
for _ in range(500000):  # класс 0
	a = uniform(0, 50)
	b = uniform(-10, 11)
	data.append((a, b, 0))

for _ in range(500000):  # класс 1
	a = uniform(50, 101)
	b = uniform(-10, 11)
	data.append((a, b, 1))

shuffle(data)

for a, b, label in data:
	x.append([a, b])
	y.append(label)


# ===============PERCEPTRON===============
class Perceptron:
	def __init__(self, input_size, learning_rate=0.1):
		self.weights = [uniform(-1, 1) for _ in range(input_size)]
		self.bias = uniform(-1, 1)
		self.lr = learning_rate

	def train(self, x, y, epochs=25):
		for _ in range(epochs):
			for inputs, label in zip(x, y):
				prediction = self.predict(inputs)
				error = label - prediction
				for i in range(len(self.weights)):
					self.weights[i] += self.lr * error * inputs[i]
				self.bias += self.lr * error

	def predict(self, inputs):
		total = 0
		for w, x in zip(self.weights, inputs):
			total += w * x
		total += self.bias
		return 1 if total >= 0 else 0

# ===============WORK_PART===============
p = Perceptron(input_size=2, learning_rate=10000)
p.train(x, y, epochs=10000)

cnt = 0
for _ in range(2000000):
	a = uniform(0, 101)
	b = uniform(-20, 20)
	pr = p.predict([a, b])
	ans = 0
	if a >= 50:
		ans = 1
	if pr != ans:
		cnt += 1
		print("Error")
print(cnt)