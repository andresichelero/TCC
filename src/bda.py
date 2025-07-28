# src/bda.py
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
        stagnation_limit=5,
        reinitialization_percent=0.7,
        # --- NEW: Parameters for adaptive weights ---
        c_cohesion_final=0.9,
        s_separation_final=0.01,
        # --- NEW: Add feature limits to the constructor ---
        min_features=1,
        max_features=None,
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
        self.c_cohesion_final = c_cohesion_final
        self.s_separation_final = s_separation_final

        # --- Mutation boost parameters ---
        self.mutation_boost_prob = 0.30  # 30% dos agentes recebem boost
        self.mutation_boost_bit_prob = 0.40  # 40% dos bits desses agentes são mutados
        self.mutation_boost_interval = 10  # a cada 10 iterações sem melhora


        if seed is not None:
            np.random.seed(seed)

        if max_features is None:
            max_features = dim

        def create_valid_position():
            """Creates a single random solution within the feature limits."""
            position = np.zeros(dim, dtype=np.int8)
            num_to_select = np.random.randint(min_features, max_features + 1)
            indices = np.random.choice(dim, num_to_select, replace=False)
            position[indices] = 1
            return position

        self._create_valid_position = create_valid_position
        self.min_features = min_features
        self.max_features = max_features

        self.positions = np.array([create_valid_position() for _ in range(self.N)], dtype=np.int8)
        self.steps = np.random.uniform(-1, 1, size=(self.N, self.dim)) * 0.1
        self.fitness_values = np.full(self.N, np.inf)

        self.food_pos = np.zeros(self.dim, dtype=int)
        self.food_fitness = np.inf
        self.enemy_pos = np.zeros(self.dim, dtype=int)
        self.enemy_fitness = -np.inf

        self.convergence_curve = np.zeros(self.T)
        self.best_accuracy_curve = np.zeros(self.T)
        self.best_num_features_curve = np.zeros(self.T)
        self.solutions_history = []
        # Internal state for stagnation tracking
        self._stagnation_counter = 0
        self._last_best_fitness = np.inf
        self.stagnation_limit = stagnation_limit
        self.reinitialization_percent = reinitialization_percent

        


    def _initialize_population_fitness(self):
        if self.verbose_optimizer_level > 0:
            print("BDA: Inicializando população e calculando fitness inicial...")
        for i in tqdm(
            range(self.N),
            desc="BDA Init Fitness",
            disable=self.verbose_optimizer_level == 0,
        ):
            results = self.fitness_func(
                self.positions[i, :],
                self.X_train_feat,
                self.y_train,
                alpha=self.alpha_fitness,
                beta=self.beta_fitness,
                verbose_level=1,
            )
            self.fitness_values[i] = results["fitness"]
            self.solutions_history.append((self.fitness_values[i], self.positions[i, :].copy()))
            if self.fitness_values[i] < self.food_fitness:
                self.food_fitness = self.fitness_values[i]
                self.food_pos = self.positions[i, :].copy()
                self.best_accuracy_curve[0] = results["accuracy"]
                self.best_num_features_curve[0] = results["num_features"]
            if self.fitness_values[i] > self.enemy_fitness:
                self.enemy_fitness = self.fitness_values[i]
                self.enemy_pos = self.positions[i, :].copy()
                self.best_accuracy_curve[0] = results["accuracy"]
                self.best_num_features_curve[0] = results["num_features"]
        if self.convergence_curve[0] == 0:
            self.convergence_curve[0] = self.food_fitness
        if np.isinf(self.food_fitness) and self.verbose_optimizer_level > 0:
            print(
                "ALERTA BDA: Nenhuma solução inicial válida encontrada, food_fitness é infinito!"
            )
        if self.verbose_optimizer_level > 0:
            print(f"BDA: Melhor fitness inicial (Food): {self.food_fitness:.4f}")
            print(f"BDA: Pior fitness inicial (Enemy): {self.enemy_fitness:.4f}")
        self._last_best_fitness = self.food_fitness


    def _reinitialize_worst_agents(self):
        """Finds the worst agents and replaces them with new random solutions respecting feature limits."""
        num_to_reinitialize = int(self.N * self.reinitialization_percent)
        if num_to_reinitialize == 0:
            return

        worst_indices = np.argsort(self.fitness_values)[-num_to_reinitialize:]

        # Replace their positions and re-evaluate their fitness
        for i in worst_indices:
            self.positions[i, :] = self._create_valid_position()
            self.steps[i, :] = np.random.uniform(-1, 1, size=self.dim) * 0.1
            
            # Re-evaluate fitness for the new position
            results = self.fitness_func(
                self.positions[i, :],
                self.X_train_feat,
                self.y_train,
                alpha=self.alpha_fitness,
                beta=self.beta_fitness,
                verbose_level=0, # Less verbose for re-init
            )
            self.fitness_values[i] = results["fitness"]
            if self.fitness_values[i] < self.food_fitness:
                self.food_fitness = self.fitness_values[i]
                self.food_pos = self.positions[i, :].copy()
            if self.fitness_values[i] > self.enemy_fitness:
                self.enemy_fitness = self.fitness_values[i]
                self.enemy_pos = self.positions[i, :].copy()


    def run(self):
        self._initialize_population_fitness()
        if np.isinf(self.food_fitness) and self.N > 0:
            if self.verbose_optimizer_level > 0:
                print("BDA: Otimização não pode prosseguir pois o fitness inicial é infinito.")
            if np.sum(self.food_pos) == 0:
                self.food_pos = self.positions[0, :].copy()
            return self.food_pos, self.food_fitness, self.convergence_curve, None, None, None
        elif self.N == 0:
            if self.verbose_optimizer_level > 0:
                print("BDA: Tamanho da população é 0. Não é possível executar.")
            return np.array([]), np.inf, self.convergence_curve, None, None, None

        if self.verbose_optimizer_level > 0:
            print(f"\nIniciando otimização BDA por {self.T} iterações...")

        mutation_boost_counter = 0

        for t in tqdm(range(self.T), desc="BDA Iterations", disable=self.verbose_optimizer_level == 0):
            if self.T > 1:
                ratio = t / (self.T - 1)
            else:
                ratio = 1.0

            current_tau = (1.0 - ratio) * self.tau_max + ratio * self.tau_min
            current_tau = max(current_tau, 1e-5)
            current_s = self.s - t * ((self.s - self.s_separation_final) / self.T)
            current_c = self.c_cohesion + t * ((self.c_cohesion_final - self.c_cohesion) / self.T)

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
                    current_s * S_i
                    + self.a * A_i
                    + current_c * C_i
                    + self.f_food * Fi
                    + self.e_enemy * Ei
                )
                current_step_velocity = behavioral_sum + self.w_inertia * self.steps[i, :]
                current_step_velocity = np.clip(current_step_velocity, self.clip_step_min, self.clip_step_max)
                self.steps[i, :] = current_step_velocity
                v_shaped_prob = np.abs(np.tanh(self.steps[i, :] / current_tau))
                flip_mask = np.random.rand(self.dim) < v_shaped_prob
                new_position_i = self.positions[i, :].copy()
                new_position_i[flip_mask] = 1 - new_position_i[flip_mask]
                self.positions[i, :] = new_position_i
                results = self.fitness_func(
                    self.positions[i, :],
                    self.X_train_feat,
                    self.y_train,
                    alpha=self.alpha_fitness,
                    beta=self.beta_fitness,
                    verbose_level=0,
                )
                current_fitness = results["fitness"]
                self.fitness_values[i] = current_fitness
                self.solutions_history.append((current_fitness, self.positions[i, :].copy()))
                if current_fitness < self.food_fitness:
                    self.food_fitness = current_fitness
                    self.food_pos = self.positions[i, :].copy()
                if current_fitness > self.enemy_fitness:
                    self.enemy_fitness = current_fitness
                    self.enemy_pos = self.positions[i, :].copy()

            # --- Stagnation Check and Handling ---
            if self.food_fitness < self._last_best_fitness:
                self._last_best_fitness = self.food_fitness
                self._stagnation_counter = 0
                mutation_boost_counter = 0
            else:
                self._stagnation_counter += 1
                mutation_boost_counter += 1

            # --- Mutation Boost: aplica "S-shaped" para alguns agentes se estagnar por mutation_boost_interval ---
            if mutation_boost_counter >= self.mutation_boost_interval:
                #print(f"\nBDA: Mutation Boost ativado na iteração {t+1}!")
                num_agents_boost = max(1, int(self.N * self.mutation_boost_prob))
                boost_indices = np.random.choice(self.N, num_agents_boost, replace=False)
                for idx in boost_indices:
                    num_bits_boost = max(1, int(self.dim * self.mutation_boost_bit_prob))
                    bits_to_mutate = np.random.choice(self.dim, num_bits_boost, replace=False)
                    for d in bits_to_mutate:
                        s_prob = 1 / (1 + np.exp(-self.steps[idx, d] / current_tau))
                        self.positions[idx, d] = 1 if np.random.rand() < s_prob else 0
                    # --- Correção: garantir limites de features após o boost ---
                    n_selected = np.sum(self.positions[idx, :])
                    if n_selected < self.min_features:
                        # Se ficou abaixo do mínimo, ativa bits aleatórios até atingir o mínimo
                        zeros = np.where(self.positions[idx, :] == 0)[0]
                        if len(zeros) > 0:
                            to_activate = np.random.choice(zeros, self.min_features - n_selected, replace=False)
                            self.positions[idx, to_activate] = 1
                    elif n_selected > self.max_features:
                        # Se ficou acima do máximo, desativa bits aleatórios até atingir o máximo
                        ones = np.where(self.positions[idx, :] == 1)[0]
                        if len(ones) > 0:
                            to_deactivate = np.random.choice(ones, n_selected - self.max_features, replace=False)
                            self.positions[idx, to_deactivate] = 0
                mutation_boost_counter = 0

            if self._stagnation_counter >= self.stagnation_limit:
                #print(f"\nBDA Stagnation detected at iter {t+1}. Re-initializing worst agents.")
                self._reinitialize_worst_agents()
                self._stagnation_counter = 0

            self.convergence_curve[t] = self.food_fitness
            if (self.verbose_optimizer_level > 0 and (t + 1) % 10 == 0):
                print(
                    f"BDA Iter {t+1}/{self.T} - Melhor Fitness (Food): {self.food_fitness:.4f}, "
                    f"Pior Fitness (Enemy): {self.enemy_fitness:.4f}, Tau: {current_tau:.2f}"
                )
            best_results_this_iter = self.fitness_func(
                self.food_pos,
                self.X_train_feat,
                self.y_train,
                self.alpha_fitness,
                self.beta_fitness,
            )
            self.best_accuracy_curve[t] = best_results_this_iter["accuracy"]
            self.best_num_features_curve[t] = best_results_this_iter["num_features"]

        # --- Checagem/correção final: garantir que food_pos respeita os limites ---
        n_selected_final = np.sum(self.food_pos)
        if n_selected_final < self.min_features:
            zeros = np.where(self.food_pos == 0)[0]
            if len(zeros) > 0:
                to_activate = np.random.choice(zeros, self.min_features - n_selected_final, replace=False)
                self.food_pos[to_activate] = 1
        elif n_selected_final > self.max_features:
            ones = np.where(self.food_pos == 1)[0]
            if len(ones) > 0:
                to_deactivate = np.random.choice(ones, n_selected_final - self.max_features, replace=False)
                self.food_pos[to_deactivate] = 0
        if self.verbose_optimizer_level > 0:
            print(f"\nBDA Otimização Concluída. Melhor fitness encontrado: {self.food_fitness:.4f}")
            num_selected_bda = np.sum(self.food_pos)
            print(f"Número de features selecionadas pelo BDA: {num_selected_bda} de {self.dim}")
        return (
            self.food_pos,
            self.food_fitness,
            self.convergence_curve,
            self.best_accuracy_curve,
            self.best_num_features_curve,
            self.solutions_history,
        )


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
        **bda_params_from_article_test,
        alpha_fitness=0.99,
        beta_fitness=0.01,
        seed=42,
        verbose_optimizer_level=1,
    )

    (
        best_solution_bda,
        best_fitness_bda,
        convergence_bda,
        acc_curve,
        feat_curve,
        history,
    ) = bda_optimizer_test_obj.run()


    print(
        f"\nMelhor solução BDA (vetor binário): {''.join(map(str,best_solution_bda.astype(int)[:20]))}..."
    )
    print(f"Melhor fitness BDA: {best_fitness_bda:.4f}")
    print(f"Número de features selecionadas BDA: {np.sum(best_solution_bda)}")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(convergence_bda)
    plt.title("Curva de Convergência BDA (Teste com KNN Fitness)")
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Fitness")
    plt.grid(True)
    plt.show()