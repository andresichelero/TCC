# src/bda.py
import gc
import math
import numpy as np
from tqdm import tqdm

try:
    from .fitness_function import evaluate_fitness
except ImportError:
    from fitness_function import evaluate_fitness


class BinaryDragonflyAlgorithm:
    def __init__(
        self,
        N,
        T,
        dim,
        fitness_func,
        X_train_feat,
        y_train,
        s=0.1,
        a=0.1,
        c_cohesion=0.7,
        f_food=1.0,
        e_enemy=1.0,
        w_inertia=0.85,
        tau_min=0.01,
        tau_max=4.0,
        clip_step_min=-6.0,
        clip_step_max=6.0,
        alpha_fitness=0.99,
        beta_fitness=0.01,
        seed=None,
        verbose_optimizer_level=0,
    ):

        self.N = N
        self.T = T
        self.dim = dim
        self.fitness_func = fitness_func
        self.X_train_feat = X_train_feat
        self.y_train = y_train
        self.alpha_fitness = alpha_fitness
        self.beta_fitness = beta_fitness

        self.s = s
        self.a = a
        self.c_cohesion = c_cohesion
        self.f_food = f_food
        self.e_enemy = e_enemy
        self.w_inertia = w_inertia

        self.tau_min = tau_min
        self.tau_max = tau_max
        self.clip_step_min = clip_step_min
        self.clip_step_max = clip_step_max
        self.verbose_optimizer_level = verbose_optimizer_level

        if seed is not None:
            np.random.seed(seed)

        self.positions = np.random.randint(0, 2, size=(self.N, self.dim))
        self.steps = np.random.uniform(-1, 1, size=(self.N, self.dim)) * 0.1
        self.fitness_values = np.full(self.N, np.inf)

        self.food_pos = np.zeros(self.dim, dtype=int)
        self.food_fitness = np.inf
        self.enemy_pos = np.zeros(self.dim, dtype=int)
        self.enemy_fitness = -np.inf

        self.convergence_curve = np.zeros(self.T)

    def _initialize_population_fitness(self):
        if self.verbose_optimizer_level > 0:
            print("BDA: Inicializando população e calculando fitness inicial...")
        for i in tqdm(
            range(self.N),
            desc="BDA Init Fitness",
            disable=self.verbose_optimizer_level == 0,
        ):
            self.fitness_values[i] = self.fitness_func(
                self.positions[i, :],
                self.X_train_feat,
                self.y_train,
                alpha=self.alpha_fitness,
                beta=self.beta_fitness,
                verbose_level=1,
            )
            if self.fitness_values[i] < self.food_fitness:
                self.food_fitness = self.fitness_values[i]
                self.food_pos = self.positions[i, :].copy()
            if self.fitness_values[i] > self.enemy_fitness:
                self.enemy_fitness = self.fitness_values[i]
                self.enemy_pos = self.positions[i, :].copy()
        if np.isinf(self.food_fitness) and self.verbose_optimizer_level > 0:
            print(
                "ALERTA BDA: Nenhuma solução inicial válida encontrada, food_fitness é infinito!"
            )
        if self.verbose_optimizer_level > 0:
            print(f"BDA: Melhor fitness inicial (Food): {self.food_fitness:.4f}")
            print(f"BDA: Pior fitness inicial (Enemy): {self.enemy_fitness:.4f}")

    def run(self):
        self._initialize_population_fitness()

        if np.isinf(self.food_fitness) and self.N > 0:
            if self.verbose_optimizer_level > 0:
                print(
                    "BDA: Otimização não pode prosseguir pois o fitness inicial é infinito."
                )
            if np.sum(self.food_pos) == 0:
                self.food_pos = self.positions[0, :].copy()
            return self.food_pos, self.food_fitness, self.convergence_curve
        elif self.N == 0:
            if self.verbose_optimizer_level > 0:
                print("BDA: Tamanho da população é 0. Não é possível executar.")
            return np.array([]), np.inf, self.convergence_curve

        if self.verbose_optimizer_level > 0:
            print(f"\nIniciando otimização BDA por {self.T} iterações...")

        for t in tqdm(
            range(self.T),
            desc="BDA Iterations",
            disable=self.verbose_optimizer_level == 0,
        ):
            gc.collect()
            if self.T > 1:
                ratio = t / (self.T - 1)
            else:
                ratio = 1.0
            current_tau = (1.0 - ratio) * self.tau_max + ratio * self.tau_min
            current_tau = max(current_tau, 1e-5)

            current_w = self.w_inertia

            # plot_first_agent_in_iter = (t % 2 == 0 or t == self.T -1 ) # Removido, era para plot de DNN da fitness

            for i in range(self.N):
                S_i = np.zeros(self.dim)
                A_i = np.zeros(self.dim)
                C_sum_Xj = np.zeros(self.dim)
                num_neighbors_for_A_C = 0

                for j in range(self.N):
                    if i == j:
                        continue
                    S_i += self.positions[j, :] - self.positions[i, :]
                    A_i += self.steps[j, :]
                    C_sum_Xj += self.positions[j, :]
                    num_neighbors_for_A_C += 1

                if num_neighbors_for_A_C > 0:
                    A_i /= num_neighbors_for_A_C
                    C_i = (C_sum_Xj / num_neighbors_for_A_C) - self.positions[i, :]
                else:
                    A_i = np.zeros(self.dim)
                    C_i = np.zeros(self.dim)

                Fi = self.food_pos - self.positions[i, :]
                Ei = self.enemy_pos + self.positions[i, :]

                behavioral_sum = (
                    self.s * S_i
                    + self.a * A_i
                    + self.c_cohesion * C_i
                    + self.f_food * Fi
                    + self.e_enemy * Ei
                )

                current_step_velocity = behavioral_sum + current_w * self.steps[i, :]
                current_step_velocity = np.clip(
                    current_step_velocity, self.clip_step_min, self.clip_step_max
                )
                self.steps[i, :] = current_step_velocity

                new_position_i = self.positions[i, :].copy()
                for d in range(self.dim):
                    delta_x_component = current_step_velocity[d]
                    x_param_transfer = delta_x_component / current_tau

                    try:
                        if x_param_transfer <= 0:
                            prob_value = 1 - (2 / (1 + math.exp(x_param_transfer)))
                        else:
                            prob_value = (2 / (1 + math.exp(-x_param_transfer))) - 1
                    except OverflowError:
                        # Se exp(grande_positivo) -> inf, então 1 - (2/inf) = 1 para x_param <=0
                        # Se exp(-grande_negativo) -> inf, então (2/inf) - 1 = -1 para x_param > 0 (mas clipado para 0)
                        # Se exp(próximo_zero) -> 1
                        prob_value = (
                            1.0 if x_param_transfer > 0 else 0.0
                        )  # Aproximação segura para overflow

                    prob_value = np.clip(prob_value, 0, 1)

                    if np.random.rand() < prob_value:
                        new_position_i[d] = 1
                    else:
                        new_position_i[d] = 0

                self.positions[i, :] = new_position_i

                current_fitness = self.fitness_func(
                    self.positions[i, :],
                    self.X_train_feat,
                    self.y_train,
                    alpha=self.alpha_fitness,
                    beta=self.beta_fitness,
                    verbose_level=0,  # verbose_level para a função de fitness (KNN CV)
                )
                self.fitness_values[i] = current_fitness

                if current_fitness < self.food_fitness:
                    self.food_fitness = current_fitness
                    self.food_pos = self.positions[i, :].copy()
                if current_fitness > self.enemy_fitness:
                    self.enemy_fitness = current_fitness
                    self.enemy_pos = self.positions[i, :].copy()

            # if plot_first_agent_in_iter: reset_fitness_call_count() # Removido

            self.convergence_curve[t] = self.food_fitness
            if (
                self.verbose_optimizer_level > 0 and (t + 1) % 10 == 0
            ):  # Log a cada 10 iterações
                print(
                    f"BDA Iter {t+1}/{self.T} - Melhor Fitness (Food): {self.food_fitness:.4f}, "
                    f"Pior Fitness (Enemy): {self.enemy_fitness:.4f}, Tau: {current_tau:.2f}"
                )

        if self.verbose_optimizer_level > 0:
            print(
                f"\nBDA Otimização Concluída. Melhor fitness encontrado: {self.food_fitness:.4f}"
            )
            num_selected_bda = np.sum(self.food_pos)
            print(
                f"Número de features selecionadas pelo BDA: {num_selected_bda} de {self.dim}"
            )
        return self.food_pos, self.food_fitness, self.convergence_curve


if __name__ == "__main__":
    print("--- Testando Binary Dragonfly Algorithm (BDA) com fitness KNN ---")

    N_AGENTS_BDA_TEST = 5  # Reduzido para teste rápido
    MAX_ITER_BDA_TEST = 10  # Reduzido para teste rápido
    DIM_FEATURES_BDA_TEST = 45
    N_CLASSES_BDA_TEST = 3
    N_TRAIN_SAMPLES_BDA_TEST = (
        60  # Mínimo para CV com 3 classes e 10 folds (2 por classe por fold)
    )

    y_train_bda_test_dummy = np.concatenate(
        [
            np.zeros(N_TRAIN_SAMPLES_BDA_TEST // N_CLASSES_BDA_TEST, dtype=int),
            np.ones(N_TRAIN_SAMPLES_BDA_TEST // N_CLASSES_BDA_TEST, dtype=int),
            np.full(
                N_TRAIN_SAMPLES_BDA_TEST
                - 2 * (N_TRAIN_SAMPLES_BDA_TEST // N_CLASSES_BDA_TEST),
                2,
                dtype=int,
            ),
        ]
    )
    np.random.shuffle(y_train_bda_test_dummy)
    X_train_bda_test_dummy = np.random.rand(
        N_TRAIN_SAMPLES_BDA_TEST, DIM_FEATURES_BDA_TEST
    )

    bda_params_from_article_test = {
        "s": 0.1,
        "a": 0.1,
        "c_cohesion": 0.7,
        "f_food": 1.0,
        "e_enemy": 1.0,
        "w_inertia": 0.85,
        "tau_min": 0.01,
        "tau_max": 4.0,
    }

    bda_optimizer_test_obj = BinaryDragonflyAlgorithm(
        N=N_AGENTS_BDA_TEST,
        T=MAX_ITER_BDA_TEST,
        dim=DIM_FEATURES_BDA_TEST,
        fitness_func=evaluate_fitness,
        X_train_feat=X_train_bda_test_dummy,
        y_train=y_train_bda_test_dummy,
        # X_val_feat, y_val, dnn_params, verbose_fitness removidos
        **bda_params_from_article_test,
        alpha_fitness=0.99,
        beta_fitness=0.01,
        seed=42,
        verbose_optimizer_level=1,
    )

    best_solution_bda, best_fitness_bda, convergence_bda = bda_optimizer_test_obj.run()

    print(
        f"\nMelhor solução BDA (vetor binário): {''.join(map(str,best_solution_bda.astype(int)[:20]))}..."
    )
    print(f"Melhor fitness BDA: {best_fitness_bda:.4f}")
    print(f"Número de features selecionadas BDA: {np.sum(best_solution_bda)}")
    # print(f"Curva de convergência BDA: {convergence_bda}") # Pode ser longa

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(convergence_bda)
    plt.title("Curva de Convergência BDA (Teste com KNN Fitness)")
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Fitness")
    plt.grid(True)
    plt.show()