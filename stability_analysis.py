import matplotlib.pyplot as plt

generations = [0, 100, 200, 300, 399]

fitness_data = {
    "boltzmann": [0.168039, 0.196266, 0.203138, 0.207721, 0.210268],
    "tournament": [0.168491, 0.194351, 0.201305, 0.205402, 0.207327],
    "ranking": [0.168538, 0.194430, 0.200603, 0.204231, 0.206658],
    "roulette": [0.168000, 0.192244, 0.198798, 0.202851, 0.205662]  # You can estimate missing 0th gen
}

variance_data = {
    "boltzmann": [0.000038, 0.000132, 0.000209, 0.000236, 0.000248],
    "tournament": [0.000036, 0.000131, 0.000220, 0.000244, 0.000264],
    "ranking": [0.000034, 0.000139, 0.000194, 0.000207, 0.000219],
    "roulette": [0.000041, 0.000118, 0.000170, 0.000229, 0.000244]
}

# for method, values in fitness_data.items():
#     plt.plot(generations, values, label=method)
# plt.title("Average Fitness Over Generations")
# plt.xlabel("Generation")
# plt.ylabel("Avg Fitness")
# plt.legend()
# plt.grid(True)
# plt.show()

for method, values in variance_data.items():
    plt.plot(generations, values, label=method)
plt.title("Fitness Variance Over Generations (Diversity Proxy)")
plt.xlabel("Generation")
plt.ylabel("Fitness Variance")
plt.legend()
plt.grid(True)
plt.show()