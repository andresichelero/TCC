# src/feature_extractor.py
import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from tqdm import tqdm

DEBUG_FEATURES = False

def _is_valid_coeffs_array(coeffs, min_len=1, segment_idx=-1, band_name_debug="N/A"):
    if not isinstance(coeffs, np.ndarray) or coeffs.ndim == 0:
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1):
            print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Entrada não é um array NumPy válido ou é escalar. Tipo: {type(coeffs)}, Conteúdo: {coeffs}", flush=True)
        return False
    if len(coeffs) < min_len:
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1):
            print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Comprimento do array ({len(coeffs)}) < mínimo ({min_len}).", flush=True)
        return False
    return True

def calculate_mav(coeffs, segment_idx=-1, band_name_debug="N/A"):
    if not _is_valid_coeffs_array(coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mav_input"): return np.nan
    return np.mean(np.abs(coeffs))

def calculate_std_dev(coeffs, segment_idx=-1, band_name_debug="N/A"):
    if not _is_valid_coeffs_array(coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_std_input"): return np.nan
    return np.std(coeffs, ddof=1) if len(coeffs) > 1 else 0.0

def calculate_skewness(coeffs, segment_idx=-1, band_name_debug="N/A"):
    if not _is_valid_coeffs_array(coeffs, min_len=3, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_skew_input"): return np.nan
    if np.all(coeffs == coeffs[0]):
         if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Sinal constante para skewness, retornando 0.", flush=True)
         return 0.0
    val = skew(coeffs, bias=False)
    return val

def calculate_kurtosis_val(coeffs, segment_idx=-1, band_name_debug="N/A"):
    if not _is_valid_coeffs_array(coeffs, min_len=4, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_kurt_input"): return np.nan
    if np.all(coeffs == coeffs[0]):
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Sinal constante para kurtosis, retornando NaN (ou 3.0 se fosse normal).", flush=True)
        return np.nan
    val = kurtosis(coeffs, fisher=False, bias=False)
    return val

def calculate_rms(coeffs, segment_idx=-1, band_name_debug="N/A"):
    if not _is_valid_coeffs_array(coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_rms_input"): return np.nan
    return np.sqrt(np.mean(coeffs**2))

def calculate_mavs_ratio(coeffs_band_numerator, mav_denominator_band, segment_idx=-1, band_name_debug="N/A"):
    """Calcula a razão MAV(banda_numerador) / MAV(banda_denominador)."""
    is_coeffs_band_valid = _is_valid_coeffs_array(coeffs_band_numerator, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mavs_num")

    if not is_coeffs_band_valid or np.isnan(mav_denominator_band):
        return np.nan

    mav_coeffs_numerator = calculate_mav(coeffs_band_numerator, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mavs_num_mav")

    if np.isnan(mav_coeffs_numerator): return np.nan
    if mav_denominator_band == 0:
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1) : print(f"Debug (seg {segment_idx}, banda {band_name_debug}): MAV da banda denominador é zero em MAVsRatio. Numerador MAV: {mav_coeffs_numerator}", flush=True)
        return np.nan
    return mav_coeffs_numerator / mav_denominator_band

def calculate_activity(coeffs, segment_idx=-1, band_name_debug="N/A"):
    """Atividade é a variância do sinal."""
    if not _is_valid_coeffs_array(coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_act_input"): return np.nan
    return np.var(coeffs, ddof=1) if len(coeffs) > 1 else 0.0

def calculate_mobility(coeffs, segment_idx=-1, band_name_debug="N/A"):
    """Mobilidade de Hjorth."""
    if not _is_valid_coeffs_array(coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mob_input"): return np.nan
    var_coeffs = np.var(coeffs, ddof=1)
    if var_coeffs < 1e-10:
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Variância do sinal ~zero ({var_coeffs}) em mobility. Retornando 0.", flush=True)
        return 0.0
    diff_coeffs = np.diff(coeffs)
    if not _is_valid_coeffs_array(diff_coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mob_diff_input"):
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Array de diferenças muito curto para variância em mobility.", flush=True)
        return np.nan

    var_diff = np.var(diff_coeffs, ddof=1)
    ratio = var_diff / var_coeffs
    if ratio < 0:
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Razão negativa ({ratio}) em mobility.", flush=True)
        return np.nan
    return np.sqrt(ratio)

def calculate_complexity(coeffs, segment_idx=-1, band_name_debug="N/A"):
    """Complexidade de Hjorth."""
    if not _is_valid_coeffs_array(coeffs, min_len=3, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_input"):
        return np.nan

    mobility_coeffs = calculate_mobility(coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_mob_coeffs")
    if np.isnan(mobility_coeffs): return np.nan
    if mobility_coeffs < 1e-10:
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Mobilidade do sinal ~zero ({mobility_coeffs}) em complexity. Retornando 0.", flush=True)
        return 0.0

    diff_coeffs = np.diff(coeffs)
    if not _is_valid_coeffs_array(diff_coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_diff_for_mob_input"):
         if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Array de primeiras diferenças muito curto para mobilidade em complexity.", flush=True)
         return np.nan

    mobility_diff = calculate_mobility(diff_coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_mob_diff")
    if np.isnan(mobility_diff): return np.nan

    return mobility_diff / mobility_coeffs

# Dicionário de funções base para as 8 características
feature_functions_base = {
    'MAV': calculate_mav, 'StdDev': calculate_std_dev, 'Skewness': calculate_skewness,
    'Kurtosis': calculate_kurtosis_val, 'RMS': calculate_rms, 'Activity': calculate_activity,
    'Mobility': calculate_mobility, 'Complexity': calculate_complexity,
}


def get_swt_subbands_recursive(signal, wavelet, current_level, max_level, band_prefix=""):
    """
    Decompõe recursivamente o sinal usando SWT para obter 2^max_level sub-bandas.
    Retorna uma lista de tuplas (nome_da_banda, coeficientes_da_banda).
    A ordem será: todas as que começam com 'A' recursivamente, depois todas com 'D'.
    Ex para max_level=2: AAAA, AAAD, AADA, AADD (do ramo A original) ... DDDD
    A ordem das 16 bandas será:
    AAAA, AAAD, AADA, AADD, ADAA, ADAD, ADDA, ADDD, (originadas de A no primeiro nível)
    DAAA, DAAD, DADA, DADD, DDAA, DDAD, DDDA, DDDD  (originadas de D no primeiro nível)
    """
    if current_level == max_level:
        return [(band_prefix, signal)]

    coeffs_level_1 = pywt.swt(signal, wavelet, level=1, trim_approx=True, norm=True)
    cA = coeffs_level_1[0]
    cD = coeffs_level_1[1]

    subbands = []
    subbands.extend(get_swt_subbands_recursive(cA, wavelet, current_level + 1, max_level, band_prefix + "A"))
    subbands.extend(get_swt_subbands_recursive(cD, wavelet, current_level + 1, max_level, band_prefix + "D"))
    return subbands


def extract_swt_features(eeg_data, wavelet='db4', level=4):
    """
    Extrai 143 características de EEG usando decomposição SWT completa de 4 níveis.
    - 16 sub-bandas são geradas.
    - 8 características base são calculadas para cada uma das 16 sub-bandas (128 features).
    - 1 característica de Razão de MAVs é calculada para 15 sub-bandas em relação
      à MAV da sub-banda de referência 'AAAA' (15 features).
    Total = 128 + 15 = 143 features.
    """
    num_segments = eeg_data.shape[0]
    original_signal_length = eeg_data.shape[1]

    # SWT requer que o comprimento do sinal seja múltiplo de 2^level para alguns modos,
    # ou pelo menos par e suficientemente longo. pywt.swt lida com isso internamente,
    # mas é bom garantir um comprimento par se possível, ou estar ciente do padding.
    # O código original truncava para o comprimento par mais próximo. Vamos manter isso.
    signal_length_for_swt = original_signal_length
    if original_signal_length % 2 != 0:
        signal_length_for_swt = original_signal_length - 1
        print(f"Aviso: Comprimento original do sinal ({original_signal_length}) é ímpar. "
              f"Será truncado para {signal_length_for_swt} para SWT.", flush=True)

    # Verifica se o sinal é suficientemente longo para a decomposição
    min_len_for_swt = pywt.Wavelet(wavelet).dec_len * (2**level) # Estimativa grosseira
    # pywt.swt com level=1 repetido 4 vezes. Cada chamada a pywt.swt(level=1)
    # pode ter seus próprios requisitos de comprimento.
    # A verificação de comprimento dentro do loop é mais robusta.

    num_base_features = len(feature_functions_base) # 8
    num_subbands = 2**level # 16 para level=4
    # Total de features = (16 bandas * 8 features base) + 15 MAVsRatios = 128 + 15 = 143
    total_features_to_extract = (num_subbands * num_base_features) + (num_subbands - 1)

    feature_matrix = np.full((num_segments, total_features_to_extract), np.nan)
    feature_names = [] # Será populada na primeira iteração bem-sucedida

    ref_band_name_for_mav_ratio = 'A' * level # Ex: "AAAA" para level=4

    print(f"Iniciando extração de {total_features_to_extract} características SWT...", flush=True)
    for i in tqdm(range(num_segments), desc="Extraindo Características SWT (143)"):
        signal_full = eeg_data[i, :]
        signal_to_process = signal_full[:signal_length_for_swt]

        if len(signal_to_process) < pywt.Wavelet(wavelet).dec_len: # Mínimo para uma decomposição de nível 1
            if DEBUG_FEATURES or i < 10: print(f"Debug (seg {i}): Sinal processado muito curto ({len(signal_to_process)}) para wavelet '{wavelet}'. Features serão NaN.", flush=True)
            continue

        all_16_subbands_tuples = []
        try:
            # A função recursiva pode falhar se o sinal ficar muito curto em níveis profundos.
            # Adicionar verificações de comprimento dentro da função recursiva ou aqui.
            # Para pywt.swt(level=1), o comprimento do sinal deve ser >= filter_len - 1.
            # Para db4, filter_len é 8. Então, sinal >= 7.
            # Após cada nível, o comprimento dos coeficientes é (N + filter_len - 1) // 2 para DWT.
            # Para SWT, o comprimento dos coeficientes é o mesmo do sinal de entrada.
            # A principal restrição é que o sinal não seja muito curto para os filtros da wavelet.
            all_16_subbands_tuples = get_swt_subbands_recursive(signal_to_process, wavelet, 0, level)
        except Exception as e_swt_call:
            if DEBUG_FEATURES or i < 10: print(f"Debug (seg {i}): Erro na decomposição SWT recursiva: {e_swt_call}. Features serão NaN.", flush=True)
            if i == 0 and not feature_names: # Tenta gerar nomes placeholder
                temp_band_names_ordered = [('A'*level)] + [('A'*(level-k) + 'D' + 'X'*(k-1)) for k in range(1,level+1)] # Nomes placeholder
                # Esta lógica de nomes placeholder precisa ser melhorada para os 16 nomes corretos.
                # Por enquanto, vamos focar em obter as features.
                pass # Os nomes serão gerados corretamente se a primeira iteração for bem-sucedida.
            continue

        if len(all_16_subbands_tuples) != num_subbands:
            if DEBUG_FEATURES or i < 10: print(f"Debug (seg {i}): Número inesperado de sub-bandas da SWT recursiva. Esperado {num_subbands}, obtido {len(all_16_subbands_tuples)}. Features serão NaN.", flush=True)
            continue

        # Ordenar as bandas por nome para consistência (AAAA, AAAD, ..., DDDD)
        # A saída de get_swt_subbands_recursive já deve ter uma ordem consistente (A-first).
        # Se quisermos uma ordem lexicográfica estrita:
        # all_16_subbands_tuples.sort(key=lambda x: x[0])

        # Extrair MAV da banda de referência primeiro
        mav_ref_band_value = np.nan
        for band_name_tuple, coeffs_tuple in all_16_subbands_tuples:
            if band_name_tuple == ref_band_name_for_mav_ratio:
                mav_ref_band_value = calculate_mav(coeffs_tuple, segment_idx=i, band_name_debug=band_name_tuple)
                break
        
        if np.isnan(mav_ref_band_value):
            if DEBUG_FEATURES or i < 10: print(f"Debug (seg {i}): MAV da banda de referência '{ref_band_name_for_mav_ratio}' é NaN. Ratios serão NaN.", flush=True)
            # As features base ainda podem ser calculadas, mas as MAVsRatios serão NaN.

        feature_col_idx = 0
        # Calcular as 8 características base para todas as 16 sub-bandas
        for band_idx, (band_name, coeffs_current_band) in enumerate(all_16_subbands_tuples):
            for feat_name_key, feat_func in feature_functions_base.items():
                value = feat_func(coeffs_current_band, segment_idx=i, band_name_debug=band_name)
                feature_matrix[i, feature_col_idx] = value
                if i == 0: feature_names.append(f"{band_name}_{feat_name_key}")
                feature_col_idx += 1

        # Calcular as 15 MAVsRatios
        for band_idx, (band_name, coeffs_current_band) in enumerate(all_16_subbands_tuples):
            if band_name == ref_band_name_for_mav_ratio:
                continue # Não calcula MAVsRatio para a banda de referência com ela mesma

            value_mav_ratio = calculate_mavs_ratio(coeffs_current_band, mav_ref_band_value, segment_idx=i, band_name_debug=band_name)
            feature_matrix[i, feature_col_idx] = value_mav_ratio
            if i == 0: feature_names.append(f"{band_name}_MAVsRatio_vs_{ref_band_name_for_mav_ratio}")
            feature_col_idx += 1
            
    # Impressões finais e verificações
    if num_segments > 0:
        print(f"Matriz de características (143) extraída: {feature_matrix.shape}", flush=True)
        if not feature_names and total_features_to_extract > 0:
            # Gerar nomes placeholder se a primeira iteração falhou completamente
            # Esta lógica precisa ser robusta para criar os 143 nomes corretos se necessário.
            # Exemplo simplificado:
            temp_names = []
            # Nomes para as 8 features base * 16 bandas
            placeholder_band_names = [f"Band{b+1}" for b in range(num_subbands)] # Nomes genéricos
            # Para nomes corretos, precisaríamos da saída de get_swt_subbands_recursive
            # ou de uma função que gere os nomes ('AAAA', 'AAAD', etc.)
            
            # Se all_16_subbands_tuples foi populado em alguma iteração, mas não na primeira,
            # os nomes podem estar incompletos. Idealmente, feature_names é construído
            # confiavelmente na primeira iteração bem-sucedida.

            # Se feature_names ainda estiver vazia, é um problema maior.
            print(f"Nomes de features: {len(feature_names)} (Esperado: {total_features_to_extract})", flush=True)


        if np.isnan(feature_matrix).all():
            print("ALERTA CRÍTICO: TODAS as features em TODA a matriz são NaN!", flush=True)
        elif np.isnan(feature_matrix).any():
            num_nan_features = np.sum(np.isnan(feature_matrix))
            total_possible_features_in_matrix = feature_matrix.size
            percent_nan = (num_nan_features / total_possible_features_in_matrix) * 100 if total_possible_features_in_matrix > 0 else 0
            print(f"Alerta: {num_nan_features} ({percent_nan:.2f}%) valores NaN encontrados na matriz de características.", flush=True)
    else:
        print("Nenhum segmento para processar na extração de features.", flush=True)

    # Garantir que feature_names tenha o comprimento correto se a primeira iteração falhou
    # mas as subsequentes não, ou se nenhuma funcionou.
    if len(feature_names) != total_features_to_extract and num_segments > 0:
        print(f"Aviso: O número de nomes de features ({len(feature_names)}) não corresponde ao esperado ({total_features_to_extract}). Isso pode ocorrer se a primeira amostra falhou na extração.", flush=True)
        # Tentativa de reconstruir nomes se possível (requer uma estrutura de nomes previsível)
        if total_features_to_extract == 143: # Hardcode para este caso específico
            temp_feature_names_rebuilt = []
            # Gerar nomes como 'AAAA_MAV', ..., 'DDDD_Complexity' (16*8 = 128)
            # e 'AAAD_MAVsRatio_vs_AAAA', ..., 'DDDD_MAVsRatio_vs_AAAA' (15)
            # Isso requer a lista ordenada de nomes das 16 bandas.
            # Por simplicidade, se os nomes não foram gerados corretamente, o usuário precisará investigar.
            # Aqui, apenas truncamos/preenchemos para o tamanho esperado se houver uma grande discrepância.
            if not feature_names: # Se estiver completamente vazio
                 feature_names = [f"feature_{k}" for k in range(total_features_to_extract)]


    return feature_matrix, feature_names


if __name__ == '__main__':
    print("--- Testando extract_swt_features_143 com dados dummy ---")
    DEBUG_FEATURES = True # Ativar logs de debug das funções de cálculo

    # Teste 1: Dados aleatórios com comprimento par
    dummy_data_even = np.random.rand(3, 4096) * 100 - 50
    print(f"\nDados (comprimento par): {dummy_data_even.shape}")
    X_feat_even, names_even = extract_swt_features_143(dummy_data_even, wavelet='db4', level=4)
    print(f"Shape feat (par): {X_feat_even.shape}, Nomes: {len(names_even) if names_even else 'N/A'}")
    if X_feat_even.size > 0 and names_even:
        print(f"Primeiras 5 features da primeira amostra (par):")
        for k in range(min(5, X_feat_even.shape[1])):
            print(f"  {names_even[k]}: {X_feat_even[0,k]:.4f}")
    elif X_feat_even.size > 0:
         print(f"Primeira linha de features (par, sem nomes):\n{X_feat_even[0,:5]}...")


    # Teste 2: Dados com comprimento ímpar
    dummy_data_odd = np.random.rand(2, 4097) * 100 - 50
    print(f"\nDados (comprimento ímpar): {dummy_data_odd.shape}")
    X_feat_odd, names_odd = extract_swt_features_143(dummy_data_odd, wavelet='db4', level=4)
    print(f"Shape feat (ímpar): {X_feat_odd.shape}, Nomes: {len(names_odd) if names_odd else 'N/A'}")
    if X_feat_odd.size > 0 and names_odd:
        print(f"Últimas 5 features da primeira amostra (ímpar):")
        for k in range(max(0, X_feat_odd.shape[1]-5), X_feat_odd.shape[1]):
            print(f"  {names_odd[k]}: {X_feat_odd[0,k]:.4f}")


    # Teste 3: Sinal constante
    dummy_data_const = np.ones((1, 4096)) * 5
    print(f"\nDados (sinal constante): {dummy_data_const.shape}")
    X_feat_c, names_c = extract_swt_features_143(dummy_data_const, wavelet='db4', level=4)
    print(f"Shape feat (constante): {X_feat_c.shape}, Nomes: {len(names_c) if names_c else 'N/A'}")
    if X_feat_c.size > 0:
        print(f"Features para sinal constante (algumas podem ser 0 ou NaN):\n{X_feat_c[0, ::15]}...") # Imprime algumas

    # Teste 4: Sinal muito curto
    # Para db4, dec_len é 8. Se level=4, a decomposição recursiva pode falhar.
    # pywt.swt(level=1) precisa de sinal de comprimento >= dec_len - 1 = 7.
    # O sinal original para extract_swt_features_143 deve ser um pouco mais longo.
    dummy_data_short = np.random.rand(1, 30) # Aumentado para tentar passar alguns níveis
    print(f"\nDados (sinal curto): {dummy_data_short.shape}")
    X_feat_s, names_s = extract_swt_features_143(dummy_data_short, wavelet='db4', level=4)
    print(f"Shape feat (curto): {X_feat_s.shape}, Nomes: {len(names_s) if names_s else 'N/A'}")
    print(f"Conteúdo (curto - esperado muitos NaNs): {X_feat_s}")

    DEBUG_FEATURES = False