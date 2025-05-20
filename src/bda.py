# Implementação do BDA
# src/bda.py
import math
import numpy as np
from tqdm import tqdm

try:
    from .fitness_function import evaluate_fitness, reset_fitness_call_count
except ImportError:
    from fitness_function import evaluate_fitness, reset_fitness_call_count

class BinaryDragonflyAlgorithm:
    def __init__(self, N, T, dim, fitness_func, X_train_feat, y_train, X_val_feat, y_val, dnn_params,
                 s=0.1, a=0.1, c_cohesion=0.7, f_food=1.0, e_enemy=1.0, w_inertia=0.85, # Valores do artigo
                 tau_min=0.01, tau_max=4.0, clip_step_min=-6.0, clip_step_max=6.0, # Valores tau do artigo + clipping (Não mencionado no artigo mas mencionado no BDA original, pode ajudar nas curvas agressivas de aprendizado)
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

        self.tau_min = tau_min
        self.tau_max = tau_max

        # Limites para o clipping do vetor de passo (Delta_X)
        self.clip_step_min = clip_step_min
        self.clip_step_max = clip_step_max

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
                self.alpha_fitness, self.beta_fitness, self.verbose_fitness,
                optimizer_name="BDA", current_iter=0, agent_idx=i,
                plot_this_fitness_dnn_history=True
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
        reset_fitness_call_count()


    def run(self):
        self._initialize_population_fitness()
        
        if np.isinf(self.food_fitness) and self.N > 0:
            print("BDA: Otimização não pode prosseguir pois o fitness inicial é infinito. Verifique a função de fitness e os dados.")
            if np.sum(self.food_pos) == 0: # food_pos não foi atualizado
                 self.food_pos = self.positions[0,:].copy() 
            return self.food_pos, self.food_fitness, self.convergence_curve
        elif self.N == 0:
            print("BDA: Tamanho da população é 0. Não é possível executar.")
            return np.array([]), np.inf, self.convergence_curve


        print(f"\nIniciando otimização BDA por {self.T} iterações...")

        for t in tqdm(range(self.T), desc="BDA Iterations"):
            # Atualização de Tau conforme artigo (diminui de tau_max para tau_min)
            # Isto faz tau DECRESCER de tau_max para tau_min.
            if self.T > 1:
                ratio = t / (self.T - 1)
            else: # Evitar divisão por zero se T=1
                ratio = 1.0
            current_tau = (1.0 - ratio) * self.tau_max + ratio * self.tau_min
            current_tau = max(current_tau, 1e-5) # Evitar tau zero ou negativo
            
            current_w = self.w_inertia

            # Plotar a primeira DNN desta iteração se for uma iteração de interesse
            plot_first_agent_in_iter = True #(t % 2 == 0 or t == self.T -1 ) # Ex: a cada 2 iterações e na última

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

                # Atração pela Comida F_i = Food_pos - X_i -> Tabela 1 do artigo
                Fi = self.food_pos - self.positions[i, :]
                # Distração do Inimigo E_i = Enemy_pos + X_i -> Tabela 1 do artigo
                Ei = self.enemy_pos + self.positions[i, :]

                # Atualização do Vetor de Passo (Delta_X)
                # Delta_X_i(t+1) = (s*S_i + a*A_i + c*C_i + f*F_i + e*E_i) + w*Delta_X_i(t)
                behavioral_sum = (self.s * S_i + 
                                  self.a * A_i + 
                                  self.c_cohesion * C_i + 
                                  self.f_food * Fi + 
                                  self.e_enemy * Ei)
                
                current_step_velocity = behavioral_sum + current_w * self.steps[i, :]
                current_step_velocity = np.clip(current_step_velocity, self.clip_step_min, self.clip_step_max) # Clipping de velocidade
                self.steps[i, :] = current_step_velocity # Salva o passo contínuo

                # Atualização da Posição Binária usando Função de Transferência V-shaped
                # P_flip = V(Delta_X_id / current_tau)
                # X_id(t+1) = 1 - X_id(t) se random() < P_flip, senão X_id(t+1) = X_id(t)
                
                new_position_i = self.positions[i, :].copy()
                for d in range(self.dim):
                    # delta_x_component é o elemento do vetor de passo para a dimensão d
                    delta_x_component = current_step_velocity[d]
                    
                    # Função de Transferência T(DeltaX) da Tabela 1 do artigo
                    x_param_transfer = delta_x_component / current_tau # Conforme Tabela 1: Δx/τ
                                        
                    if x_param_transfer <= 0:
                        # Artigo: 1 - (2 / (1 + exp(Δx/τ)))
                        # math.exp vs np.exp para arrays. Aqui é escalar.
                        try:
                            prob_value = 1 - (2 / (1 + math.exp(x_param_transfer)))
                        except OverflowError: # exp(grande_positivo) -> inf
                            prob_value = 1 - 0 # 1 - (2/inf) = 1
                    else: # x_param_transfer > 0
                        # Artigo: (2 / (1 + exp(-(Δx/τ)))) - 1
                        try:
                            prob_value = (2 / (1 + math.exp(-x_param_transfer))) - 1
                        except OverflowError: # exp(grande_positivo) para -x_param_transfer pequeno (próximo de zero)
                            prob_value = (2 / (1 + 0)) - 1 # (2/1)-1 = 1
                            
                    # A função de transferência da Tabela 1 deve retornar uma probabilidade [0,1]
                    # Ex: se exp(x_param_transfer) for muito pequeno (x_param_transfer muito negativo), 1 - (2/ (1+small)) pode ser < 0
                    # Ex: se exp(-x_param_transfer) for muito pequeno (x_param_transfer muito positivo), (2/(1+small)) - 1 pode ser > 1
                    # O clipping é pra garantir, mas a função já deveria ser bounded
                    prob_value = np.clip(prob_value, 0, 1)

                    # Atualização da Posição X_d(t+1) conforme Tabela 1 do artigo
                    # X_d(t+1) = 1 se rand < T(v_d(t+1)), senão 0
                    if np.random.rand() < prob_value:
                        new_position_i[d] = 1
                    else:
                        new_position_i[d] = 0
                
                self.positions[i, :] = new_position_i

                # Avalia a nova posição
                current_fitness = self.fitness_func(
                    self.positions[i, :], self.X_train_feat, self.y_train,
                    self.X_val_feat, self.y_val, self.dnn_params,
                    self.alpha_fitness, self.beta_fitness, self.verbose_fitness,
                    optimizer_name="BDA", current_iter=t+1, agent_idx=i,
                    plot_this_fitness_dnn_history=True
                )
                self.fitness_values[i] = current_fitness

                # Atualiza Food (melhor solução) e Enemy (pior solução)
                if current_fitness < self.food_fitness:
                    self.food_fitness = current_fitness
                    self.food_pos = self.positions[i, :].copy()
                if current_fitness > self.enemy_fitness: # Pior fitness
                    self.enemy_fitness = current_fitness
                    self.enemy_pos = self.positions[i, :].copy()
            
            if plot_first_agent_in_iter: reset_fitness_call_count()
            
            self.convergence_curve[t] = self.food_fitness
            print(f"BDA Iter {t+1}/{self.T} - Melhor Fitness (Food): {self.food_fitness:.4f}, "f"Pior Fitness (Enemy): {self.enemy_fitness:.4f}, Tau: {current_tau:.2f}")

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