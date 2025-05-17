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
                 s=0.1, a=0.2, c_cohesion=0.7, f_food=1.0, e_enemy=1.0, w_inertia=0.9,
                 # Parâmetros da função de transferência V-Shaped e tau
                 # O artigo diz "tau: Dinâmico, variando de tmin a tmax"
                 # Valores padrões usados: tmin=1, tmax=6 (comum para Vmax em PSO)
                 # ou uma estratégia de decaimento para 1/tau para que a probabilidade de flip diminua.
                 # Pflip = V(DeltaX_id / tau).
                 # Se tau aumenta, Pflip diminui (para o mesmo DeltaX).
                 # Parece correto pra exploração -> explotação.
                 tau_min=1.0, tau_max=6.0,
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

        # Pesos dos comportamentos (valores de exemplo, artigo não especifica)
        self.s = s  # Separação
        self.a = a  # Alinhamento
        self.c_cohesion = c_cohesion  # Coesão
        self.f_food = f_food  # Atração pela comida
        self.e_enemy = e_enemy  # Distração do inimigo
        self.w_inertia = w_inertia # Peso de inércia para o passo (pode ser dinâmico também)

        self.tau_min = tau_min
        self.tau_max = tau_max

        if seed is not None:
            np.random.seed(seed)

        # Inicialização
        self.positions = np.random.randint(0, 2, size=(self.N, self.dim))
        self.steps = np.zeros((self.N, self.dim)) # Vetor de passo Delta_X
        self.fitness = np.full(self.N, np.inf)

        self.food_pos = np.zeros(self.dim)
        self.food_fitness = np.inf
        self.enemy_pos = np.zeros(self.dim)
        self.enemy_fitness = -np.inf # Para inimigo, queremos maximizar a 'pior' fitness (que é minimizada)

        self.convergence_curve = np.zeros(self.T)

    def _initialize_population(self):
        print("BDA: Inicializando população e calculando fitness inicial...")
        for i in tqdm(range(self.N), desc="BDA Init Fitness"):
            self.fitness[i] = self.fitness_func(
                self.positions[i, :], self.X_train_feat, self.y_train,
                self.X_val_feat, self.y_val, self.dnn_params,
                self.alpha_fitness, self.beta_fitness, self.verbose_fitness
            )
            if self.fitness[i] < self.food_fitness:
                self.food_fitness = self.fitness[i]
                self.food_pos = self.positions[i, :].copy()
            if self.fitness[i] > self.enemy_fitness: # Fitness é minimizado, então o 'pior' é o maior
                self.enemy_fitness = self.fitness[i]
                self.enemy_pos = self.positions[i, :].copy()
        print(f"BDA: Melhor fitness inicial (Food): {self.food_fitness:.4f}")
        print(f"BDA: Pior fitness inicial (Enemy): {self.enemy_fitness:.4f}")


    def _v_shaped_transfer_function(self, x):
        """Função de transferência V-shaped comum: |tanh(x)|"""
        return np.abs(np.tanh(x))

    def run(self):
        self._initialize_population()
        print(f"\nIniciando otimização BDA por {self.T} iterações...")

        for t in tqdm(range(self.T), desc="BDA Iterations"):
            # Atualiza o parâmetro tau (linearmente crescente de tau_min para tau_max)
            # Ou outra estratégia de atualização para tau ou 1/tau
            current_tau = self.tau_min + (self.tau_max - self.tau_min) * (t / self.T)
            # current_w = self.w_inertia # Poderia ser dinâmico, ex: w_max - t * (w_max - w_min) / T

            for i in range(self.N):
                # Calcula os 5 vetores de comportamento para a libélula i
                # Vizinhança: geralmente todas as outras libélulas no BDA padrão
                # Cálculo de S, A, C: fórmulas conceituais do DA original, adaptando para a posição da comida/inimigo.
                # Separação Si: - sum(Xi - Xj) for vizinhos j
                # Alinhamento Ai: sum(Delta_Xj) / N_vizinhos for vizinhos j
                # Coesão Ci: (sum(Xj) / N_vizinhos) - Xi for vizinhos j
                # Estes são mais complexos de definir claramente sem referências do BDA específico usado no artigo
                
                # --- Componentes de Comportamento ---
                # Para simplificar, se N é pequeno (e.g., 10), a vizinhança pode ser todas as outras.
                # Se Si, Ai, Ci fossem calculados, seriam somas/médias sobre vizinhos.
                # Uma forma comum é que S, A, C são zero se não há vizinhos em um raio (não aplicável aqui se todos são vizinhos).
                
                # Por enquanto, vamos focar em F e E que são mais claramente definidos
                # F_i = X_food - X_i
                # E_i = X_enemy + X_i  (inimigo repele, comida atrai)

                # Para S, A, C, a forma como são calculados (e.g., média das diferenças, etc.) é crucial.
                # Dado que N=10, vamos assumir um impacto mais direto de F e E.

                # Vetor de Atração pela Comida (Fi)
                Fi = self.food_pos - self.positions[i, :]

                # Vetor de Distração do Inimigo (Ei)
                Ei = self.enemy_pos + self.positions[i, :] # O artigo diz afastar-se, então X_i - X_enemy ou X_enemy - X_i com peso negativo.
                                                           # A fórmula original DA é X_inimigo + X_i
                                                           # O efeito é que E aponta para longe do inimigo se X_i estiver próximo.

                # Si, Ai, Ci são mais complexos e geralmente envolvem iteração sobre outros agentes.
                # Para uma implementação inicial e dado N=10, podemos testar sem eles ou com placeholders.
                # DeltaXi(t+1) = (sSi + aAi + cCi + fFi + eEi) + w * DeltaXi(t)
                # Se S,A,C forem zero:
                S_i = np.zeros(self.dim) # Placeholder
                A_i = np.zeros(self.dim) # Placeholder
                C_i = np.zeros(self.dim) # Placeholder

                # Atualização do Vetor de Passo (Delta_X)
                # DeltaXi(t+1) = (s*Si + a*Ai + c*Ci + f*Fi + e*Ei) + w*DeltaXi(t)
                # O artigo não especifica se os pesos s,a,c,f,e são aplicados aos vetores normalizados
                # ou diretamente. As equações originais do DA são:
                # Si = - sum(X - Xj)
                # Ai = sum(Vj) / N_neighbors
                # Ci = (sum(Xj)/N_neighbors) - X
                # Fi = X_food - X
                # Ei = X_enemy + X
                # Delta_X(t+1) = (s*Si + a*Ai + c*Ci + f*Fi + e*Ei) + w*Delta_X(t)

                # Como os Xi são binários (0 ou 1), (X_food - Xi) etc., resultarão em -1, 0, 1.
                # Esta é uma simplificação. O BDA real pode usar regras mais complexas para S,A,C.
                current_step_velocity = (self.s * S_i + self.a * A_i + self.c_cohesion * C_i +
                                         self.f_food * Fi + self.e_enemy * Ei) + \
                                        self.w_inertia * self.steps[i, :]

                self.steps[i, :] = current_step_velocity # Salva o passo contínuo

                # Atualização da Posição Binária usando Função de Transferência V-shaped
                # P_flip = V(Delta_X_id / tau)
                # X_id(t+1) = 1 - X_id(t) se random() < P_flip, senão X_id(t+1) = X_id(t)
                
                new_position_i = self.positions[i, :].copy()
                for d in range(self.dim):
                    # O artigo sugere: prob_flip = abs(tanh(DeltaX_id(t+1) / tau))
                    prob_flip = self._v_shaped_transfer_function(current_step_velocity[d] / current_tau)
                    if np.random.rand() < prob_flip:
                        new_position_i[d] = 1 - new_position_i[d] # Inverte o bit
                
                self.positions[i, :] = new_position_i

                # Avalia a nova posição
                current_fitness = self.fitness_func(
                    self.positions[i, :], self.X_train_feat, self.y_train,
                    self.X_val_feat, self.y_val, self.dnn_params,
                    self.alpha_fitness, self.beta_fitness, self.verbose_fitness
                )
                self.fitness[i] = current_fitness

                # Atualiza Xfood e Xenemy
                if current_fitness < self.food_fitness:
                    self.food_fitness = current_fitness
                    self.food_pos = self.positions[i, :].copy()
                if current_fitness > self.enemy_fitness: # Pior fitness
                    self.enemy_fitness = current_fitness
                    self.enemy_pos = self.positions[i, :].copy()
            
            self.convergence_curve[t] = self.food_fitness
            if (t + 1) % 10 == 0: # Log a cada 10 iterações
                print(f"BDA Iter {t+1}/{self.T} - Melhor Fitness (Food): {self.food_fitness:.4f}, "
                      f"Pior Fitness (Enemy): {self.enemy_fitness:.4f}, Tau: {current_tau:.2f}")

        print(f"\nBDA Otimização Concluída. Melhor fitness encontrado: {self.food_fitness:.4f}")
        num_selected_bda = np.sum(self.food_pos)
        print(f"Número de features selecionadas pelo BDA: {num_selected_bda} de {self.dim}")
        return self.food_pos, self.food_fitness, self.convergence_curve


if __name__ == '__main__':
    # Exemplo de uso com dados dummy para BDA
    N_AGENTS_BDA = 10
    MAX_ITER_BDA = 20 # Reduzido para teste rápido
    DIM_FEATURES_BDA = 45
    N_CLASSES_BDA = 3

    X_train_bda = np.random.rand(100, DIM_FEATURES_BDA)
    y_train_bda = np.random.randint(0, N_CLASSES_BDA, 100)
    X_val_bda = np.random.rand(20, DIM_FEATURES_BDA)
    y_val_bda = np.random.randint(0, N_CLASSES_BDA, 20)

    dnn_params_bda_test = {'epochs': 3, 'batch_size': 16, 'patience': 2} # Teste rápido

    print("\n--- Testando Binary Dragonfly Algorithm (BDA) ---")
    bda_optimizer = BinaryDragonflyAlgorithm(
        N=N_AGENTS_BDA, T=MAX_ITER_BDA, dim=DIM_FEATURES_BDA,
        fitness_func=evaluate_fitness, # Passando a função real
        X_train_feat=X_train_bda, y_train=y_train_bda,
        X_val_feat=X_val_bda, y_val=y_val_bda,
        dnn_params=dnn_params_bda_test,
        # Parâmetros BDA (pode precisar de ajuste)
        s=0.1, a=0.0, c_cohesion=0.0, f_food=1.0, e_enemy=0.5, w_inertia=0.9, # Exemplo com S,A,C zerados
        tau_min=1.0, tau_max=4.0,
        alpha_fitness=0.99, beta_fitness=0.01,
        verbose_fitness=0, # Silenciar Keras durante fitness
        seed=42
    )

    best_solution_bda, best_fitness_bda, convergence_bda = bda_optimizer.run()

    print(f"\nMelhor solução BDA (vetor binário): {''.join(map(str,best_solution_bda.astype(int))[:20])}...") # Mostra os primeiros 20
    print(f"Melhor fitness BDA: {best_fitness_bda:.4f}")
    print(f"Número de features selecionadas BDA: {np.sum(best_solution_bda)}")
    print(f"Curva de convergência BDA: {convergence_bda}")

    import matplotlib.pyplot as plt
    plt.plot(convergence_bda)
    plt.title("Curva de Convergência BDA")
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Fitness")
    plt.show()