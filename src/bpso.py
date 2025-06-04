# src/bpso.py
import gc
import numpy as np
from tqdm import tqdm

try:
    from .fitness_function import evaluate_fitness
except ImportError:
    from fitness_function import evaluate_fitness

class BinaryPSO:
    def __init__(
        self,
        N,
        T,
        dim,
        fitness_func,
        X_train_feat,
        y_train,
        w_max=0.9,
        w_min=0.4,
        c1=2.0,
        c2=2.0,
        Vmax=4.0,
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
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.Vmax = Vmax

        self.verbose_optimizer_level = verbose_optimizer_level

        if seed is not None:
            np.random.seed(seed)

        self.positions = np.random.randint(0, 2, size=(self.N, self.dim))
        self.velocities = np.random.uniform(-1, 1, size=(self.N, self.dim)) * 0.1

        self.pbest_pos = self.positions.copy()
        self.pbest_fitness = np.full(self.N, np.inf)

        self.gbest_pos = np.zeros(
            self.dim, dtype=int
        )
        self.gbest_fitness = np.inf

        self.convergence_curve = np.zeros(self.T)

    def _initialize_population(self):
        if self.verbose_optimizer_level > 0:
            print("BPSO: Inicializando população e calculando fitness inicial...")
        for i in tqdm(
            range(self.N),
            desc="BPSO Init Fitness",
            disable=self.verbose_optimizer_level == 0,
        ):
            fitness_val = self.fitness_func(
                self.positions[i, :],
                self.X_train_feat,
                self.y_train,
                alpha=self.alpha_fitness,
                beta=self.beta_fitness,
                verbose_level=1,
            )
            self.pbest_fitness[i] = fitness_val

            if fitness_val < self.gbest_fitness:
                self.gbest_fitness = fitness_val
                self.gbest_pos = self.positions[i, :].copy()
        if self.verbose_optimizer_level > 0:
            print(f"BPSO: Melhor fitness inicial (gBest): {self.gbest_fitness:.4f}")

    def _sigmoid_transfer_function(self, v):
        return 1 / (1 + np.exp(-v))

    def run(self):
        self._initialize_population()
        if np.isinf(self.gbest_fitness) and self.N > 0:
            if self.verbose_optimizer_level > 0:
                print(
                    "BPSO: Otimização não pode prosseguir pois o fitness inicial é infinito."
                )
            if np.sum(self.gbest_pos) == 0:  # Se gbest_pos não foi atualizado
                self.gbest_pos = self.positions[0, :].copy()
            return self.gbest_pos, self.gbest_fitness, self.convergence_curve
        elif self.N == 0:
            if self.verbose_optimizer_level > 0:
                print("BPSO: Tamanho da população é 0. Não é possível executar.")
            return np.array([]), np.inf, self.convergence_curve

        if self.verbose_optimizer_level > 0:
            print(f"\nIniciando otimização BPSO por {self.T} iterações...")

        for t in tqdm(
            range(self.T),
            desc="BPSO Iterations",
            disable=self.verbose_optimizer_level == 0,
        ):
            gc.collect()
            current_w = self.w_max - (self.w_max - self.w_min) * (t / self.T)

            for i in range(self.N):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive_comp = (
                    self.c1 * r1 * (self.pbest_pos[i, :] - self.positions[i, :])
                )
                social_comp = self.c2 * r2 * (self.gbest_pos - self.positions[i, :])

                self.velocities[i, :] = (
                    current_w * self.velocities[i, :] + cognitive_comp + social_comp
                )

                if self.Vmax is not None:
                    self.velocities[i, :] = np.clip(
                        self.velocities[i, :], -self.Vmax, self.Vmax
                    )

                prob_to_be_1 = self._sigmoid_transfer_function(self.velocities[i, :])
                new_position_i = (np.random.rand(self.dim) < prob_to_be_1).astype(int)
                self.positions[i, :] = new_position_i

                current_fitness = self.fitness_func(
                    self.positions[i, :],
                    self.X_train_feat,
                    self.y_train,
                    alpha=self.alpha_fitness,
                    beta=self.beta_fitness,
                    verbose_level=1,
                )
                if current_fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = current_fitness
                    self.pbest_pos[i, :] = self.positions[i, :].copy()

                    if current_fitness < self.gbest_fitness:
                        self.gbest_fitness = current_fitness
                        self.gbest_pos = self.positions[i, :].copy()
            self.convergence_curve[t] = self.gbest_fitness
            if self.verbose_optimizer_level > 0 and (t + 1) % 10 == 0:
                print(
                    f"BPSO Iter {t+1}/{self.T} - Melhor Fitness (gBest): {self.gbest_fitness:.4f}, w: {current_w:.2f}"
                )

        if self.verbose_optimizer_level > 0:
            print(
                f"\nBPSO Otimização Concluída. Melhor fitness global encontrado: {self.gbest_fitness:.4f}"
            )
            num_selected_bpso = np.sum(self.gbest_pos)
            print(
                f"Número de features selecionadas pelo BPSO: {num_selected_bpso} de {self.dim}"
            )
        return self.gbest_pos, self.gbest_fitness, self.convergence_curve


if __name__ == "__main__":
    N_PARTICLES_BPSO_TEST = 5
    MAX_ITER_BPSO_TEST = 10
    DIM_FEATURES_BPSO_TEST = 45
    N_CLASSES_BPSO_TEST = 3
    N_TRAIN_SAMPLES_BPSO_TEST = 60

    y_train_bpso_dummy = np.concatenate(
        [
            np.zeros(N_TRAIN_SAMPLES_BPSO_TEST // N_CLASSES_BPSO_TEST, dtype=int),
            np.ones(N_TRAIN_SAMPLES_BPSO_TEST // N_CLASSES_BPSO_TEST, dtype=int),
            np.full(
                N_TRAIN_SAMPLES_BPSO_TEST
                - 2 * (N_TRAIN_SAMPLES_BPSO_TEST // N_CLASSES_BPSO_TEST),
                2,
                dtype=int,
            ),
        ]
    )
    np.random.shuffle(y_train_bpso_dummy)
    X_train_bpso_dummy = np.random.rand(
        N_TRAIN_SAMPLES_BPSO_TEST, DIM_FEATURES_BPSO_TEST
    )

    print(
        "\n--- Testando Binary Particle Swarm Optimization (BPSO) com KNN Fitness ---"
    )
    bpso_optimizer_test_obj = BinaryPSO(
        N=N_PARTICLES_BPSO_TEST,
        T=MAX_ITER_BPSO_TEST,
        dim=DIM_FEATURES_BPSO_TEST,
        fitness_func=evaluate_fitness,
        X_train_feat=X_train_bpso_dummy,
        y_train=y_train_bpso_dummy,
        # X_val_feat, y_val, dnn_params, verbose_fitness removidos
        w_max=0.9,
        w_min=0.4,
        c1=2.0,
        c2=2.0,
        Vmax=4.0,
        alpha_fitness=0.99,
        beta_fitness=0.01,
        seed=42,
        verbose_optimizer_level=1,
    )

    best_solution_bpso, best_fitness_bpso, convergence_bpso = (
        bpso_optimizer_test_obj.run()
    )

    print(
        f"\nMelhor solução BPSO (vetor binário): {''.join(map(str,best_solution_bpso.astype(int))[:20])}..."
    )
    print(f"Melhor fitness BPSO: {best_fitness_bpso:.4f}")
    print(f"Número de features selecionadas BPSO: {np.sum(best_solution_bpso)}")
    # print(f"Curva de convergência BPSO: {convergence_bpso}")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(convergence_bpso)
    plt.title("Curva de Convergência BPSO (Teste com KNN Fitness)")
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Fitness Global")
    plt.grid(True)
    plt.show()
