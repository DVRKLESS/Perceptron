from random import uniform
from math import ceil
from anim_simple_perceptron import Perceptron, generate_data
import matplotlib.pyplot as plt
import subprocess

# def git_commit_and_push(commit_message="Автоматический коммит после работы скрипта"):
# 	try:
# 		subprocess.run(["git", "add", "."], check=True)

# 		subprocess.run(["git", "commit", "-m", commit_message], check=True)

# 		subprocess.run(["git", "push", "origin", "main"], check=True)

# 		print("Git commit и push выполнены успешно.")
# 	except subprocess.CalledProcessError as e:
# 		print(f"Ошибка при выполнении git команды: {e}")
	
# if __name__ == "__main__":
	# =============== TRAIN AND SAVE ===============
# for percept in range(1, 31):
# k = ceil(uniform(-10, 10))
# c = ceil(uniform(-20, 20))
# # ep = ceil(uniform(50, 300))
# tempo = (uniform(0.01, 100))
data = generate_data(func=lambda x: 25 * x - 68, n_points=1000, x_range=(-100, 100), y_range=(-500, 500), gap = 100)
x = [[a, b] for a, b, _ in data]
y = [label for _, _, label in data]
p = Perceptron(input_size=2, learning_rate=100)
p.train(x, y, epochs=200, dump_name=f"5", save_mode="off")
# git_commit_and_push()


# =============== LOADING_DUMP ===============
# p2 = Perceptron(input_size=2)
# p2.load_model("/Users/dvrkless/Documents/SYNC/VScode/Python/Perceptron/01,-5,14,55,41.86668605837094.pkl")

# plt.ion()
# p2.plot_decision_boundary(p2.train_x, p2.train_y, epoch="Loaded model")
# plt.ioff()

# =============== TESTING BY ADDING ===============
# while True:
# 	try:
# 		inp = input("Введите точку X Y (или 'exit'): ")
# 		if inp.strip().lower() == 'exit':
# 			break
# 		px, py = map(float, inp.split())
# 		pred = p.predict([px, py])
# 		print(f"Класс: {pred}")
# 		plt.scatter(px, py, color='green' if pred == 0 else 'purple', marker='D', s=100)
# 		plt.draw()
# 	except Exception as e:
# 		print("Ошибка ввода:", e)
# plt.show()

