# src/feature_extractor.py
import numpy as np
import pywt
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# --- Funções de Cálculo de Características ---
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
    return np.std(coeffs, ddof=1)

def calculate_skewness(coeffs, segment_idx=-1, band_name_debug="N/A"):
    if not _is_valid_coeffs_array(coeffs, min_len=3, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_skew_input"): return np.nan
    if np.all(coeffs == coeffs[0]): 
         if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Sinal constante para skewness, retornando 0.", flush=True)
         return 0.0
    val = skew(coeffs)
    return val

def calculate_kurtosis_val(coeffs, segment_idx=-1, band_name_debug="N/A"):
    if not _is_valid_coeffs_array(coeffs, min_len=4, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_kurt_input"): return np.nan
    if np.all(coeffs == coeffs[0]): 
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Sinal constante para kurtosis, retornando NaN.", flush=True)
        return np.nan 
    val = kurtosis(coeffs, fisher=False)
    return val

def calculate_rms(coeffs, segment_idx=-1, band_name_debug="N/A"):
    if not _is_valid_coeffs_array(coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_rms_input"): return np.nan
    return np.sqrt(np.mean(coeffs**2))

def calculate_mavs_ratio(coeffs_band, coeffs_a_level_band, segment_idx=-1, band_name_debug="N/A"):
    is_coeffs_band_valid = _is_valid_coeffs_array(coeffs_band, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mavs_num")
    is_coeffs_a_level_valid = _is_valid_coeffs_array(coeffs_a_level_band, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mavs_den_Alevel")

    if not is_coeffs_band_valid or not is_coeffs_a_level_valid:
        return np.nan

    mav_coeffs = calculate_mav(coeffs_band, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mavs_num_mav")
    mav_a_level = calculate_mav(coeffs_a_level_band, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mavs_den_Alevel_mav")
    
    if np.isnan(mav_coeffs) or np.isnan(mav_a_level): return np.nan
    if mav_a_level == 0:
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1) : print(f"Debug (seg {segment_idx}, banda {band_name_debug}): MAV da banda A-level é zero em MAVsRatio. Numerador MAV: {mav_coeffs}", flush=True)
        return np.nan 
    return mav_coeffs / mav_a_level

def calculate_activity(coeffs, segment_idx=-1, band_name_debug="N/A"):
    if not _is_valid_coeffs_array(coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_act_input"): return np.nan
    return np.var(coeffs, ddof=1)

def calculate_mobility(coeffs, segment_idx=-1, band_name_debug="N/A"):
    if not _is_valid_coeffs_array(coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mob_input"): return np.nan
    var_coeffs = np.var(coeffs, ddof=1)
    if var_coeffs < 1e-10: 
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Variância do sinal ~zero ({var_coeffs}) em mobility. Retornando 0.", flush=True)
        return 0.0
    diff_coeffs = np.diff(coeffs)
    if not _is_valid_coeffs_array(diff_coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_mob_diff_input"):
        return np.nan
    var_diff = np.var(diff_coeffs, ddof=1)
    ratio = var_diff / var_coeffs 
    if ratio < 0: 
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Razão negativa ({ratio}) em mobility.", flush=True)
        return np.nan
    return np.sqrt(ratio)

def calculate_complexity(coeffs, segment_idx=-1, band_name_debug="N/A"):
    if not _is_valid_coeffs_array(coeffs, min_len=3, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_input"): 
        return np.nan
    mobility_coeffs = calculate_mobility(coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_mob_coeffs")
    if np.isnan(mobility_coeffs): return np.nan
    if mobility_coeffs < 1e-10:
        if DEBUG_FEATURES and (segment_idx < 3 or segment_idx == -1): print(f"Debug (seg {segment_idx}, banda {band_name_debug}): Mobilidade do sinal ~zero ({mobility_coeffs}) em complexity. Retornando 0.", flush=True)
        return 0.0
    diff_coeffs = np.diff(coeffs)
    if not _is_valid_coeffs_array(diff_coeffs, min_len=2, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_diff_for_mob_input"):
         return np.nan
    mobility_diff = calculate_mobility(diff_coeffs, segment_idx=segment_idx, band_name_debug=f"{band_name_debug}_comp_mob_diff")
    if np.isnan(mobility_diff): return np.nan
    return mobility_diff / mobility_coeffs

feature_functions_base = {
    'MAV': calculate_mav, 'StdDev': calculate_std_dev, 'Skewness': calculate_skewness,
    'Kurtosis': calculate_kurtosis_val, 'RMS': calculate_rms, 'Activity': calculate_activity,
    'Mobility': calculate_mobility, 'Complexity': calculate_complexity,
}

def apply_swt(signal, wavelet='db4', level=4):
    # print("############## APPLY SWT (DEBUG USER) ###############")
    # temp_coeffs = pywt.swt(signal, wavelet, level=level, trim_approx=True)
    # print(temp_coeffs)
    # print("############## END APPLY SWT (DEBUG USER) ###############")
    # return temp_coeffs
    return pywt.swt(signal, wavelet, level=level, trim_approx=True)


def extract_swt_features(eeg_data, wavelet='db4', level=4):
    num_segments = eeg_data.shape[0]
    original_signal_length = eeg_data.shape[1]
    signal_length_for_swt = original_signal_length - (original_signal_length % 2)
    printed_length_warning_once = False

    total_features_to_extract = (len(feature_functions_base) + 1) * (level + 1)
    feature_matrix = np.full((num_segments, total_features_to_extract), np.nan)
    feature_names = [] # Será populada na primeira iteração bem-sucedida
    band_names_ordered = [f'A{level}'] + [f'D{i}' for i in range(level, 0, -1)] # Ex: ['A4', 'D4', 'D3', 'D2', 'D1']

    print("Iniciando extração de características SWT...", flush=True)
    for i in tqdm(range(num_segments), desc="Extraindo Características SWT"):
        signal_full = eeg_data[i, :]
        signal_to_process = signal_full[:signal_length_for_swt]

        if original_signal_length % 2 != 0 and not printed_length_warning_once:
            print(f"Aviso: Comp. original ({original_signal_length}) -> truncado para {signal_length_for_swt} para SWT.", flush=True)
            printed_length_warning_once = True

        if DEBUG_FEATURES and i < 3:
            print(f"\nDebug DETALHADO para Segmento {i} ANTES DO SWT:")
            print(f"  signal_to_process shape: {signal_to_process.shape}, dtype: {signal_to_process.dtype}")
            if signal_to_process.size > 0:
                 print(f"  min: {np.min(signal_to_process):.4f}, max: {np.max(signal_to_process):.4f}, mean: {np.mean(signal_to_process):.4f}, std: {np.std(signal_to_process):.4f}", flush=True)
            if np.isnan(signal_to_process).any(): print(f"  ALERTA: signal_to_process para seg {i} CONTÉM NaNs ANTES do SWT!", flush=True)
            if signal_to_process.size > 0 and np.all(signal_to_process == signal_to_process[0]):
                 print(f"  ALERTA: signal_to_process para seg {i} é CONSTANTE ANTES do SWT! (Valor: {signal_to_process[0]})", flush=True)
        
        if len(signal_to_process) < (2**level): 
            if DEBUG_FEATURES or i < 10: print(f"Debug (seg {i}): Sinal processado muito curto ({len(signal_to_process)}) para nível {level}. Features serão NaN.", flush=True)
            continue # Matriz já é NaN

        swt_output_arrays = []
        try:
            swt_output_arrays = apply_swt(signal_to_process, wavelet, level)
        except Exception as e_swt_call:
            if DEBUG_FEATURES or i < 10: print(f"Debug (seg {i}): Erro na chamada pywt.swt: {e_swt_call}. Features serão NaN.", flush=True)
            continue

        sub_bands_coeffs_map = {}
        valid_coeffs_extracted = True

        # Esperado: swt_output_arrays é uma lista de (level+1) arrays: [cA_L, cD_L, cD_L-1, ..., cD_1]
        if not isinstance(swt_output_arrays, list) or len(swt_output_arrays) != (level + 1):
            if DEBUG_FEATURES or i < 10: print(f"Debug (seg {i}): Estrutura inesperada da saída do SWT. Esperado lista de {level+1} arrays. Obtido tipo: {type(swt_output_arrays)}, Len: {len(swt_output_arrays) if isinstance(swt_output_arrays,list) else 'N/A'}. Features serão NaN.", flush=True)
            valid_coeffs_extracted = False
        else:
            # Aproximação A_level (e.g. A4)
            coeff_A_level_candidate = np.asarray(swt_output_arrays[0])
            if coeff_A_level_candidate.ndim == 0: # É escalar
                if DEBUG_FEATURES or i < 10: print(f"Debug (seg {i}): Coeficiente cA{level} (saída SWT[0]) é escalar: {coeff_A_level_candidate}. Tipo: {type(coeff_A_level_candidate)}", flush=True)
                valid_coeffs_extracted = False
            sub_bands_coeffs_map[f'A{level}'] = coeff_A_level_candidate
            
            # Detalhes D_level, D_level-1, ..., D1
            for k_idx in range(level): # k_idx de 0 a level-1
                detail_level_value = level - k_idx # D4, D3, D2, D1
                array_index_in_swt_output = k_idx + 1 # D4 é swt_output_arrays[1], D3 é [2], ..., D1 é [level]
                
                coeff_D_candidate = np.asarray(swt_output_arrays[array_index_in_swt_output])
                if coeff_D_candidate.ndim == 0: # É escalar
                    if DEBUG_FEATURES or i < 10: print(f"Debug (seg {i}): Coeficiente cD{detail_level_value} (saída SWT[{array_index_in_swt_output}]) é escalar: {coeff_D_candidate}. Tipo: {type(coeff_D_candidate)}", flush=True)
                    valid_coeffs_extracted = False 
                sub_bands_coeffs_map[f'D{detail_level_value}'] = coeff_D_candidate
        
        if not valid_coeffs_extracted:
            # features para este segmento permanecerão NaN (como inicializado)
            if i == 0 and not feature_names: # Se o primeiro segmento falhar, tente criar nomes de placeholder
                 for band_name_ph in band_names_ordered:
                    for feat_name_ph_key in feature_functions_base.keys(): feature_names.append(f"{band_name_ph}_{feat_name_ph_key}")
                    feature_names.append(f"{band_name_ph}_MAVsRatio")
            continue # Pula para o próximo segmento
            
        coeffs_A_level_band_arr = sub_bands_coeffs_map.get(f'A{level}')

        feature_col_idx = 0
        for band_name in band_names_ordered:
            coeffs_current_band_arr = sub_bands_coeffs_map.get(band_name)
            
            for feat_name, feat_func in feature_functions_base.items():
                value = feat_func(coeffs_current_band_arr, segment_idx=i, band_name_debug=band_name)
                feature_matrix[i, feature_col_idx] = value
                if i == 0: feature_names.append(f"{band_name}_{feat_name}") # Popular nomes apenas na primeira iteração bem-sucedida
                feature_col_idx += 1
            
            value_mav_ratio = calculate_mavs_ratio(coeffs_current_band_arr, coeffs_A_level_band_arr, segment_idx=i, band_name_debug=band_name)
            feature_matrix[i, feature_col_idx] = value_mav_ratio
            if i == 0: feature_names.append(f"{band_name}_MAVsRatio")
            feature_col_idx += 1

    # Impressões finais e verificações
    if num_segments > 0:
        print(f"Matriz de características extraída: {feature_matrix.shape}", flush=True)
        if not feature_names and total_features_to_extract > 0: # Se nenhum segmento produziu nomes
            temp_feature_names = []
            for bn_temp in band_names_ordered:
                for fk_temp in feature_functions_base.keys(): temp_feature_names.append(f"{bn_temp}_{fk_temp}")
                temp_feature_names.append(f"{bn_temp}_MAVsRatio")
            feature_names = temp_feature_names[:total_features_to_extract]
            print(f"Nomes de features gerados por placeholder (nenhum segmento processado com sucesso para nomes): {len(feature_names)}", flush=True)
        elif feature_names:
             print(f"Total de {len(feature_names)} nomes de características gerados.", flush=True)

        if np.isnan(feature_matrix).all():
            print("ALERTA CRÍTICO: TODAS as features em TODA a matriz são NaN!", flush=True)
        elif np.isnan(feature_matrix).any():
            num_nan_features = np.sum(np.isnan(feature_matrix))
            total_possible_features = feature_matrix.size
            percent_nan = (num_nan_features / total_possible_features) * 100 if total_possible_features > 0 else 0
            print(f"Alerta: {num_nan_features} ({percent_nan:.2f}%) valores NaN encontrados na matriz de características.", flush=True)
            # nan_rows_sample = np.unique(np.where(np.isnan(feature_matrix))[0])[:10]
            # print(f"  Amostra de índices de LINHAS com NaN (até 10): {nan_rows_sample}", flush=True)
            # if feature_names:
            #     nan_cols_indices = np.unique(np.where(np.isnan(feature_matrix))[1])[:5]
            #     print(f"  Amostra de NOMES de COLUNAS com NaN (até 5): {[feature_names[c] for c in nan_cols_indices if c < len(feature_names)]}", flush=True)
    else:
        print("Nenhum segmento para processar na extração de features.", flush=True)

    return feature_matrix, feature_names

if __name__ == '__main__':
    print("--- Testando extract_swt_features com dados dummy (DEBUG_FEATURES ATIVADO) ---")
    DEBUG_FEATURES = False 

    dummy_data_odd = np.random.rand(3, 4097) * 100 - 50 # Dados com alguma variação
    print(f"\nDados (comprimento ímpar): {dummy_data_odd.shape}")
    X_feat_odd, names_odd = extract_swt_features(dummy_data_odd)
    print(f"Shape feat (ímpar): {X_feat_odd.shape}, Nomes: {len(names_odd) if names_odd else 'N/A'}")
    if X_feat_odd.size > 0 : print(f"Primeira linha de features (ímpar):\n{X_feat_odd[0,:]}")

    dummy_data_const = np.ones((2, 4096)) * 5 
    print(f"\nDados (sinal constante): {dummy_data_const.shape}")
    X_feat_c, names_c = extract_swt_features(dummy_data_const)
    print(f"Shape feat (constante): {X_feat_c.shape}, Nomes: {len(names_c) if names_c else 'N/A'}")
    if X_feat_c.size > 0 : print(f"Features sinal constante (seg 0):\n{X_feat_c[0,:]}")
    
    dummy_data_zeros = np.zeros((1, 4096))
    print(f"\nDados (zeros): {dummy_data_zeros.shape}")
    X_feat_z, names_z = extract_swt_features(dummy_data_zeros)
    print(f"Shape feat (zeros): {X_feat_z.shape}, Nomes: {len(names_z) if names_z else 'N/A'}")
    print(f"Conteúdo (zeros - algumas features podem ser 0, outras NaN):\n{X_feat_z}")

    dummy_data_short = np.random.rand(1, 10) 
    print(f"\nDados (sinal curto): {dummy_data_short.shape}")
    X_feat_s, names_s = extract_swt_features(dummy_data_short, wavelet='db4', level=4)
    print(f"Shape feat (curto): {X_feat_s.shape}, Nomes: {len(names_s) if names_s else 'N/A'}")
    print(f"Conteúdo (curto - deve ser todo NaN): {X_feat_s}")
    
    DEBUG_FEATURES = False # Reverter para False para execuções normais