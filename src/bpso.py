# Implementação do BPSO
# src/bpso.py
import gc
import numpy as np
from tqdm import tqdm

try:
    from .fitness_function import evaluate_fitness
except ImportError:
    from fitness_function import evaluate_fitness

class BinaryPSO:
    def __init__(self, N, T, dim, fitness_func, X_train_feat, y_train, X_val_feat, y_val, dnn_params,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, Vmax=4.0, # Vmax é recomendado
                 alpha_fitness=0.99, beta_fitness=0.01, verbose_fitness=0, seed=None):
        self.N = N  # Tamanho da população (número de partículas)
        self.T = T  # Número máximo de iterações
        self.dim = dim  # Dimensionalidade (número total de features)
        self.fitness_func = fitness_func
        self.X_train_feat = X_train_feat
        self.y_train = y_train
        self.X_val_feat = X_val_feat
        self.y_val = y_val
        self.dnn_params = dnn_params
        self.alpha_fitness = alpha_fitness
        self.beta_fitness = beta_fitness
        self.verbose_fitness = verbose_fitness


        # Parâmetros do BPSO
        self.w_max = w_max # Peso de inércia máximo
        self.w_min = w_min # Peso de inércia mínimo
        self.c1 = c1  # Coeficiente cognitivo
        self.c2 = c2  # Coeficiente social
        self.Vmax = Vmax # Limite da velocidade (recomendado para BPSO)

        if seed is not None:
            np.random.seed(seed)

        # Inicialização
        self.positions = np.random.randint(0, 2, size=(self.N, self.dim))
        # Velocidades iniciais pequenas e aleatórias (ou zero)
        self.velocities = np.random.uniform(-1, 1, size=(self.N, self.dim)) * 0.1
        # self.velocities = np.zeros((self.N, self.dim))

        self.pbest_pos = self.positions.copy()
        self.pbest_fitness = np.full(self.N, np.inf)

        self.gbest_pos = np.zeros(self.dim)
        self.gbest_fitness = np.inf

        self.convergence_curve = np.zeros(self.T)

    def _initialize_population(self):
        print("BPSO: Inicializando população e calculando fitness inicial...")
        for i in tqdm(range(self.N), desc="BPSO Init Fitness"):
            fitness_val = self.fitness_func(
                self.positions[i, :], self.X_train_feat, self.y_train,
                self.X_val_feat, self.y_val, self.dnn_params,
                self.alpha_fitness, self.beta_fitness, self.verbose_fitness
            )
            self.pbest_fitness[i] = fitness_val
            # pbest_pos já é uma cópia da posição inicial

            if fitness_val < self.gbest_fitness:
                self.gbest_fitness = fitness_val
                self.gbest_pos = self.positions[i, :].copy()
        print(f"BPSO: Melhor fitness inicial (gBest): {self.gbest_fitness:.4f}")

    def _sigmoid_transfer_function(self, v):
        """Função de transferência Sigmoide para BPSO"""
        return 1 / (1 + np.exp(-v))

    def run(self):
        self._initialize_population()
        print(f"\nIniciando otimização BPSO por {self.T} iterações...")

        for t in tqdm(range(self.T), desc="BPSO Iterations"):
            gc.collect()
            # Atualiza o peso de inércia (linearmente decrescente)
            current_w = self.w_max - (self.w_max - self.w_min) * (t / self.T)

            for i in range(self.N):
                # Atualização da Velocidade
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive_comp = self.c1 * r1 * (self.pbest_pos[i, :] - self.positions[i, :])
                social_comp = self.c2 * r2 * (self.gbest_pos - self.positions[i, :])
                
                self.velocities[i, :] = (current_w * self.velocities[i, :] +
                                         cognitive_comp + social_comp)

                # Limitação da Velocidade (clipping)
                if self.Vmax is not None:
                    self.velocities[i, :] = np.clip(self.velocities[i, :], -self.Vmax, self.Vmax)

                # Atualização da Posição Binária
                # S(Vid(t+1)) = 1 / (1 + exp(-Vid(t+1)))
                # Xid(t+1) = 1 se random() < S(Vid(t+1)), senão 0
                prob_to_be_1 = self._sigmoid_transfer_function(self.velocities[i, :])
                
                new_position_i = (np.random.rand(self.dim) < prob_to_be_1).astype(int)
                self.positions[i, :] = new_position_i

                # Avalia a nova posição
                current_fitness = self.fitness_func(
                    self.positions[i, :], self.X_train_feat, self.y_train,
                    self.X_val_feat, self.y_val, self.dnn_params,
                    self.alpha_fitness, self.beta_fitness, self.verbose_fitness
                )

                # Atualiza pbest
                if current_fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = current_fitness
                    self.pbest_pos[i, :] = self.positions[i, :].copy()

                    # Atualiza gbest se o novo pbest for melhor
                    if current_fitness < self.gbest_fitness:
                        self.gbest_fitness = current_fitness
                        self.gbest_pos = self.positions[i, :].copy()
            
            self.convergence_curve[t] = self.gbest_fitness
            if (t + 1) % 10 == 0: # Log a cada 10 iterações
                print(f"BPSO Iter {t+1}/{self.T} - Melhor Fitness (gBest): {self.gbest_fitness:.4f}, w: {current_w:.2f}")

        print(f"\nBPSO Otimização Concluída. Melhor fitness global encontrado: {self.gbest_fitness:.4f}")
        num_selected_bpso = np.sum(self.gbest_pos)
        print(f"Número de features selecionadas pelo BPSO: {num_selected_bpso} de {self.dim}")
        return self.gbest_pos, self.gbest_fitness, self.convergence_curve

if __name__ == '__main__':
    # Exemplo de uso com dados dummy para BPSO
    N_PARTICLES_BPSO = 10
    MAX_ITER_BPSO = 20 # Reduzido para teste rápido
    DIM_FEATURES_BPSO = 45
    N_CLASSES_BPSO = 3

    X_train_bpso = np.random.rand(100, DIM_FEATURES_BPSO)
    y_train_bpso = np.random.randint(0, N_CLASSES_BPSO, 100)
    X_val_bpso = np.random.rand(20, DIM_FEATURES_BPSO)
    y_val_bpso = np.random.randint(0, N_CLASSES_BPSO, 20)

    dnn_params_bpso_test = {'epochs': 3, 'batch_size': 16, 'patience': 2} # Teste rápido

    print("\n--- Testando Binary Particle Swarm Optimization (BPSO) ---")
    bpso_optimizer = BinaryPSO(
        N=N_PARTICLES_BPSO, T=MAX_ITER_BPSO, dim=DIM_FEATURES_BPSO,
        fitness_func=evaluate_fitness, # Passando a função real
        X_train_feat=X_train_bpso, y_train=y_train_bpso,
        X_val_feat=X_val_bpso, y_val=y_val_bpso,
        dnn_params=dnn_params_bpso_test,
        w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, Vmax=4.0,
        alpha_fitness=0.99, beta_fitness=0.01,
        verbose_fitness=0, # Silenciar Keras durante fitness
        seed=42
    )

    best_solution_bpso, best_fitness_bpso, convergence_bpso = bpso_optimizer.run()

    print(f"\nMelhor solução BPSO (vetor binário): {''.join(map(str,best_solution_bpso.astype(int))[:20])}...")
    print(f"Melhor fitness BPSO: {best_fitness_bpso:.4f}")
    print(f"Número de features selecionadas BPSO: {np.sum(best_solution_bpso)}")
    print(f"Curva de convergência BPSO: {convergence_bpso}")

    import matplotlib.pyplot as plt
    plt.plot(convergence_bpso)
    plt.title("Curva de Convergência BPSO")
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Fitness Global")
    plt.show()