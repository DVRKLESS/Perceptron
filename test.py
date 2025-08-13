from anim_simple_perceptron import Perceptron, generate_data
import matplotlib.pyplot as plt
# =============== TRAIN AND SAVE ===============
# data = generate_data(func=lambda x: 0.5 * x - 10, n_points=1000, x_range=(-100, 100), y_range=(-30, 30))
# x = [[a, b] for a, b, _ in data]
# y = [label for _, _, label in data]

# p = Perceptron(input_size=2, learning_rate=0.01)
# p.train(x, y, epochs=50, dump_name="01", save_mode="on")

# =============== LOADING_DUMP ===============
# p2 = Perceptron(input_size=2)
# p2.load_model("dumps\\01.pkl")

# plt.ion()
# p2.plot_decision_boundary(p2.train_x, p2.train_y, epoch="Loaded model")
# plt.ioff()

# =============== TESTING BY ADDING ===============
# while True:
#     try:
#         inp = input("Введите точку X Y (или 'exit'): ")
#         if inp.strip().lower() == 'exit':
#             break
#         px, py = map(float, inp.split())
#         pred = p2.predict([px, py])
#         print(f"Класс: {pred}")
#         plt.scatter(px, py, color='green' if pred == 0 else 'purple', marker='D', s=100)
#         plt.draw()
#     except Exception as e:
#         print("Ошибка ввода:", e)

# plt.show()
