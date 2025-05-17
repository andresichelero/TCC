# Implementação do BDA
# src/bda.py
import numpy as np
import math
from tqdm import tqdm

try:
    from .fitness_function import evaluate_fitness
except ImportError:
    from fitness_function import evaluate_fitness

class BinaryDragonflyAlgorithm:
    def __init__(self, N, T, dim, fitness_func, X_train_feat, y_train, X_val_feat, y_val, dnn_params,
                 s=0.1, a=0.1, c_cohesion=0.7, f_food=1.0, e_enemy=1.0, w_inertia=0.85, # Valores do artigo
                 tau_min=0.01, tau_max=4.0, # Valores tau do artigo
                 alpha_fitness=0.99, beta_fitness=0.01, verbose_fitness=0, seed=None):
        self.N = N  # Tamanho da população
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

        self.s = s  # Separação
        self.a = a  # Alinhamento
        self.c_cohesion = c_cohesion  # Coesão
        self.f_food = f_food  # Atração pela comida
        self.e_enemy = e_enemy  # Distração do inimigo (afastamento)
        self.w_inertia = w_inertia # Peso de inércia (fixo conforme artigo)

        # Parâmetros para a função de transferência V-Shaped
        # Tau aumenta de tau_min para tau_max, o que diminui a probabilidade de flip para o mesmo DeltaX,
        # favorecendo a explotação no final.
        self.tau_min = tau_min
        self.tau_max = tau_max

        if seed is not None:
            np.random.seed(seed)

        # Inicialização das posições (vetores binários de características)
        self.positions = np.random.randint(0, 2, size=(self.N, self.dim))
        # Inicialização dos vetores de passo (Delta_X)
        self.steps = np.random.uniform(-1, 1, size=(self.N, self.dim)) * 0.1 # Pequenos passos iniciais
        
        self.fitness_values = np.full(self.N, np.inf) # Armazena o fitness de cada libélula

        self.food_pos = np.zeros(self.dim, dtype=int)
        self.food_fitness = np.inf
        self.enemy_pos = np.zeros(self.dim, dtype=int)
        self.enemy_fitness = -np.inf # Fitness é minimizado, então o "pior" tem o maior valor de fitness

        self.convergence_curve = np.zeros(self.T)

    def _initialize_population_fitness(self):
        """Calcula o fitness inicial para toda a população e define food/enemy."""
        print("BDA: Inicializando população e calculando fitness inicial...")
        for i in tqdm(range(self.N), desc="BDA Init Fitness"):
            self.fitness_values[i] = self.fitness_func(
                self.positions[i, :], self.X_train_feat, self.y_train,
                self.X_val_feat, self.y_val, self.dnn_params,
                self.alpha_fitness, self.beta_fitness, self.verbose_fitness
            )
            if self.fitness_values[i] < self.food_fitness:
                self.food_fitness = self.fitness_values[i]
                self.food_pos = self.positions[i, :].copy()
            # O artigo define Enemy(E) como i.e. Worst Solution.
            # Se fitness é para ser minimizado, a pior solução tem o maior valor de fitness.
            if self.fitness_values[i] > self.enemy_fitness:
                self.enemy_fitness = self.fitness_values[i]
                self.enemy_pos = self.positions[i, :].copy()
        
        if np.isinf(self.food_fitness): # Se nenhuma solução inicial válida foi encontrada
            print("ALERTA BDA: Nenhuma solução inicial válida encontrada, food_fitness é infinito!")
        print(f"BDA: Melhor fitness inicial (Food): {self.food_fitness:.4f}")
        print(f"BDA: Pior fitness inicial (Enemy): {self.enemy_fitness:.4f}")


    def _v_shaped_transfer_function(self, delta_x_component_scaled):
        """Função de transferência V-shaped: |tanh(x)|.
        Onde x = delta_x_component / tau.
        O argumento já deve vir escalado por tau.
        """
        return np.abs(np.tanh(delta_x_component_scaled))

    def run(self):
        self._initialize_population_fitness()
        
        if np.isinf(self.food_fitness):
            print("BDA: Otimização não pode prosseguir pois o fitness inicial é infinito. Verifique a função de fitness e os dados.")
            # Retorna a posição inicial aleatória como "melhor" se nada mais
            # Ou poderia retornar None, None, self.convergence_curve
            # Para evitar erros, retorna uma posição aleatória da população como food_pos
            # se food_pos não foi atualizado.
            if np.sum(self.food_pos) == 0 and self.N > 0: # Se food_pos ainda é zeros e há população
                 self.food_pos = self.positions[0,:].copy() if self.N >0 else np.zeros(self.dim, dtype=int)

            return self.food_pos, self.food_fitness, self.convergence_curve


        print(f"\nIniciando otimização BDA por {self.T} iterações...")

        for t in tqdm(range(self.T), desc="BDA Iterations"):
            # Atualiza o parâmetro tau (linearmente crescente de tau_min para tau_max)
            # Isso faz com que 1/tau diminua, reduzindo a prob. de flip no final (favorece explotação)
            current_tau = self.tau_min + (self.tau_max - self.tau_min) * (t / self.T)
            if current_tau == 0: current_tau = 1e-6 # Evitar divisão por zero se tau_min e tau_max forem zero
            current_w = self.w_inertia

            # Atualizar os coeficientes s, a, c, f, e (o artigo não especifica que são dinâmicos, então usamos os valores fixos)
            # Se fossem dinâmicos, seriam atualizados aqui, por exemplo:
            # c_factor = c_max - t * (c_max - c_min) / self.T 
            # s_factor = s_max - t * (s_max - s_min) / self.T
            # ...etc. Mas o artigo parece usar valores fixos.

            for i in range(self.N): # Para cada libélula i
                # --- Calcular Vetores de Comportamento S, A, C ---
                # Baseado nas equações da Tabela 1 do artigo(interpretadas para N-1 vizinhos)
                S_i = np.zeros(self.dim)  # Vetor de Separação
                A_i = np.zeros(self.dim)  # Vetor de Alinhamento
                C_sum_Xj = np.zeros(self.dim) # Soma das posições dos vizinhos para Coesão
                
                num_neighbors_for_A_C = 0 # Para A e C, que são médias
                
                # Iterar sobre outras libélulas (j != i)
                for j in range(self.N):
                    if i == j:
                        continue
                    
                    # Separação S_i = - sum_{j!=i} (X_i - X_j) = sum_{j!=i} (X_j - X_i)
                    S_i += (self.positions[j, :] - self.positions[i, :])
                    
                    # Para Alinhamento A_i (soma dos passos dos vizinhos)
                    A_i += self.steps[j, :]
                    
                    # Para Coesão C_i (soma das posições dos vizinhos)
                    C_sum_Xj += self.positions[j, :]
                    num_neighbors_for_A_C +=1

                if num_neighbors_for_A_C > 0:
                    A_i /= num_neighbors_for_A_C # Média dos passos
                    C_i = (C_sum_Xj / num_neighbors_for_A_C) - self.positions[i, :] # (Centro de massa dos vizinhos) - X_i
                else: # Caso N=1, S,A,C são zero
                    A_i = np.zeros(self.dim)
                    C_i = np.zeros(self.dim)
                # S_i é a soma direta, não uma média.

                # Vetor de Atração pela Comida (Fi)
                Fi = self.food_pos - self.positions[i, :]

                # Vetor de Distração do Inimigo (Ei)
                # A fórmula do DA original é X_enemy + X_i.
                # Tabela 1 do artigo: E_i = E_enemy + X_i.
                Ei = self.enemy_pos + self.positions[i, :]

                # Atualização do Vetor de Passo (Delta_X)
                # Delta_X_i(t+1) = (s*S_i + a*A_i + c*C_i + f*F_i + e*E_i) + w*Delta_X_i(t)
                behavioral_sum = (self.s * S_i + 
                                  self.a * A_i + 
                                  self.c_cohesion * C_i + 
                                  self.f_food * Fi + 
                                  self.e_enemy * Ei)
                
                current_step_velocity = behavioral_sum + current_w * self.steps[i, :]
                self.steps[i, :] = current_step_velocity # Salva o passo contínuo

                # Atualização da Posição Binária usando Função de Transferência V-shaped
                # P_flip = V(Delta_X_id / current_tau)
                # X_id(t+1) = 1 - X_id(t) se random() < P_flip, senão X_id(t+1) = X_id(t)
                
                new_position_i = self.positions[i, :].copy()
                for d in range(self.dim):
                    scaled_delta_x_component = current_step_velocity[d] / current_tau
                    prob_flip = self._v_shaped_transfer_function(scaled_delta_x_component)
                    if np.random.rand() < prob_flip:
                        new_position_i[d] = 1 - new_position_i[d] # Inverte o bit
                
                self.positions[i, :] = new_position_i

                # Avalia a nova posição
                current_fitness = self.fitness_func(
                    self.positions[i, :], self.X_train_feat, self.y_train,
                    self.X_val_feat, self.y_val, self.dnn_params,
                    self.alpha_fitness, self.beta_fitness, self.verbose_fitness
                )
                self.fitness_values[i] = current_fitness

                # Atualiza Food (melhor solução) e Enemy (pior solução)
                if current_fitness < self.food_fitness:
                    self.food_fitness = current_fitness
                    self.food_pos = self.positions[i, :].copy()
                if current_fitness > self.enemy_fitness: # Pior fitness
                    self.enemy_fitness = current_fitness
                    self.enemy_pos = self.positions[i, :].copy()
            
            self.convergence_curve[t] = self.food_fitness
            if (t + 1) % 10 == 0 or t == self.T - 1: # Log a cada 10 iterações e na última
                print(f"BDA Iter {t+1}/{self.T} - Melhor Fitness (Food): {self.food_fitness:.4f}, "
                      f"Pior Fitness (Enemy): {self.enemy_fitness:.4f}, Tau: {current_tau:.2f}")

        print(f"\nBDA Otimização Concluída. Melhor fitness encontrado: {self.food_fitness:.4f}")
        num_selected_bda = np.sum(self.food_pos)
        print(f"Número de features selecionadas pelo BDA: {num_selected_bda} de {self.dim}")
        return self.food_pos, self.food_fitness, self.convergence_curve


if __name__ == '__main__':
    # Exemplo de uso com dados dummy para BDA
    print("--- Testando Binary Dragonfly Algorithm (BDA) com implementação S,A,C ---")
    DEBUG_FEATURES = False

    N_AGENTS_BDA = 10
    MAX_ITER_BDA = 20 # Reduzido para teste rápido
    DIM_FEATURES_BDA = 45
    N_CLASSES_BDA = 3

    # Gerar dados dummy para fitness
    X_train_bda_test = np.random.rand(100, DIM_FEATURES_BDA)
    y_train_bda_test = np.random.randint(0, N_CLASSES_BDA, 100)
    X_val_bda_test = np.random.rand(30, DIM_FEATURES_BDA)
    y_val_bda_test = np.random.randint(0, N_CLASSES_BDA, 30)

    # Parâmetros da DNN para teste rápido da função de fitness
    dnn_params_bda_test = {'epochs': 5, 'batch_size': 16, 'patience': 3} 

    # Parâmetros do BDA conforme artigo
    bda_params_from_article = {
        's': 0.1, 'a': 0.1, 'c_cohesion': 0.7, 
        'f_food': 1.0, 'e_enemy': 1.0, 'w_inertia': 0.85,
        'tau_min': 0.01, 'tau_max': 4.0
    }

    bda_optimizer_test = BinaryDragonflyAlgorithm(
        N=N_AGENTS_BDA, T=MAX_ITER_BDA, dim=DIM_FEATURES_BDA,
        fitness_func=evaluate_fitness, 
        X_train_feat=X_train_bda_test, y_train=y_train_bda_test,
        X_val_feat=X_val_bda_test, y_val=y_val_bda_test,
        dnn_params=dnn_params_bda_test,
        **bda_params_from_article,
        alpha_fitness=0.99, beta_fitness=0.01,
        verbose_fitness=0, # Silenciar Keras durante fitness
        seed=42
    )

    best_solution_bda, best_fitness_bda, convergence_bda = bda_optimizer_test.run()

    print(f"\nMelhor solução BDA (vetor binário): {''.join(map(str,best_solution_bda.astype(int)[:20]))}...")
    print(f"Melhor fitness BDA: {best_fitness_bda:.4f}")
    print(f"Número de features selecionadas BDA: {np.sum(best_solution_bda)}")
    print(f"Curva de convergência BDA: {convergence_bda}")

    import matplotlib.pyplot as plt
    plt.plot(convergence_bda)
    plt.title("Curva de Convergência BDA")
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Fitness")
    plt.show()