import matplotlib.pyplot as plt

fitness_ranking = {
    10: [0.189299, 0.224864, 0.241099, 0.251912, 0.259834],
    25: [0.188658, 0.243020, 0.265400, 0.279213, 0.288775],
    40: [0.188861, 0.258513, 0.280915, 0.294992, 0.303395]
}

fitness_tournament = {
    10: [0.187268, 0.233014, 0.250793, 0.261879, 0.269172],
    25: [0.189801, 0.252251, 0.269078, 0.283634, 0.291899],
    40: [0.194497, 0.265094, 0.286313, 0.301048, 0.307810]
}

fitness_roulette = {
    10: [0.187584, 0.218702, 0.234123, 0.244789, 0.253667],
    25: [0.186066, 0.236038, 0.258225, 0.270694, 0.281313],
    40: [0.188628, 0.250393, 0.274598, 0.288527, 0.297586]
}

var_ranking = {
    10: [0.000129, 0.000257, 0.000333, 0.000322, 0.000380],
    25: [0.000153, 0.000502, 0.000622, 0.000722, 0.000799],
    40: [0.000112, 0.000709, 0.000930, 0.001132, 0.001199]
}

var_tournament = {
    10: [0.000153, 0.000476, 0.000623, 0.000721, 0.000714],
    25: [0.000140, 0.000579, 0.000596, 0.000687, 0.000784],
    40: [0.000281, 0.000321, 0.000607, 0.000731, 0.000826]
}

var_roulette = {
    10: [0.000145, 0.000362, 0.000488, 0.000434, 0.000453],
    25: [0.000086, 0.000296, 0.000309, 0.000358, 0.000461],
    40: [0.000110, 0.000440, 0.000507, 0.000573, 0.000661]
}

# Shared X-axis
generations = [0, 50, 100, 150, 199]

# # Plot average fitness for each method
# plt.figure(figsize=(10, 6))

# # RANKING
# plt.plot(generations, fitness_ranking[10], label="Ranking k=10", linestyle="--", marker="o")
# plt.plot(generations, fitness_ranking[25], label="Ranking k=25", linestyle="-.", marker="o")
# plt.plot(generations, fitness_ranking[40], label="Ranking k=40", linestyle="-", marker="o")

# # TOURNAMENT
# plt.plot(generations, fitness_tournament[10], label="Tournament k=10", linestyle="--", marker="s")
# plt.plot(generations, fitness_tournament[25], label="Tournament k=25", linestyle="-.", marker="s")
# plt.plot(generations, fitness_tournament[40], label="Tournament k=40", linestyle="-", marker="s")

# # ROULETTE
# plt.plot(generations, fitness_roulette[10], label="Roulette k=10", linestyle="--", marker="^")
# plt.plot(generations, fitness_roulette[25], label="Roulette k=25", linestyle="-.", marker="^")
# plt.plot(generations, fitness_roulette[40], label="Roulette k=40", linestyle="-", marker="^")

# plt.title("Evolución del Fitness Promedio según Método de Selección y k")
# plt.xlabel("Generación")
# plt.ylabel("Fitness Promedio")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("fitness_avg_vs_k.png")
# plt.show()

# -------------------------------------------------------------------------------------------
# VARIANZA:
# -------------------------------------------------------------------------------------------

plt.figure(figsize=(10, 6))

# RANKING
plt.plot(generations, var_ranking[10], label="Ranking k=10", linestyle="--", marker="o")
plt.plot(generations, var_ranking[25], label="Ranking k=25", linestyle="-.", marker="o")
plt.plot(generations, var_ranking[40], label="Ranking k=40", linestyle="-", marker="o")

# TOURNAMENT
plt.plot(generations, var_tournament[10], label="Tournament k=10", linestyle="--", marker="s")
plt.plot(generations, var_tournament[25], label="Tournament k=25", linestyle="-.", marker="s")
plt.plot(generations, var_tournament[40], label="Tournament k=40", linestyle="-", marker="s")

# ROULETTE
plt.plot(generations, var_roulette[10], label="Roulette k=10", linestyle="--", marker="^")
plt.plot(generations, var_roulette[25], label="Roulette k=25", linestyle="-.", marker="^")
plt.plot(generations, var_roulette[40], label="Roulette k=40", linestyle="-", marker="^")

plt.title("Varianza del Fitness (Diversidad) según Método y k")
plt.xlabel("Generación")
plt.ylabel("Varianza del Fitness")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fitness_var_vs_k.png")
plt.show()