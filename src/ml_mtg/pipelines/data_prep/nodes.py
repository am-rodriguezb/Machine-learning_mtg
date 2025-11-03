import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def normalize_frames(cards: pd.DataFrame, decks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_cards = cards.copy()
    df_decks = decks.copy()
    df_cards['name'] = df_cards['name'].str.strip().str.lower()
    df_decks['Name']  = df_decks['Name'].str.strip().str.lower()
    return df_cards, df_decks

def parse_decklist(deck_text: str):
    rows = []
    for line in str(deck_text).split('\n'):
        m = re.match(r'(\d+)\s+(.+)', line.strip())
        if m:
            rows.append((m.group(2).strip().lower(), int(m.group(1))))
    return rows

def expand_decks(df_decks: pd.DataFrame) -> pd.DataFrame:
    """
    Expande los mazos optimizado para memoria usando list comprehensions
    y tipos de datos optimizados
    """
    bag = []
    for _, r in df_decks.iterrows():
        for name, qty in parse_decklist(r['Deck list']):
            bag.append([r['Name'], r['Tier'], r['Year'], name, qty])
    
    # Crear DataFrame con tipos optimizados para reducir memoria
    df_result = pd.DataFrame(bag, columns=['deck_name','tier','year','card_name','quantity'])
    
    # Optimizar tipos de datos
    # Convertir strings a category para columnas con valores repetidos
    if len(df_result) > 0:
        if df_result['deck_name'].nunique() < len(df_result):
            df_result['deck_name'] = df_result['deck_name'].astype('category')
        if df_result['card_name'].nunique() < len(df_result):
            df_result['card_name'] = df_result['card_name'].astype('category')
    
    # Optimizar tipos numéricos
    if 'tier' in df_result.columns:
        df_result['tier'] = pd.to_numeric(df_result['tier'], errors='coerce').astype('int8')
    if 'year' in df_result.columns:
        df_result['year'] = pd.to_numeric(df_result['year'], errors='coerce').astype('int16')
    if 'quantity' in df_result.columns:
        df_result['quantity'] = pd.to_numeric(df_result['quantity'], errors='coerce').astype('int8')
    
    return df_result

def merge_cards(deck_cards_intermediate, norm_cards):
    """
    Merge optimizado para reducir uso de memoria:
    - Selecciona solo las columnas necesarias antes del merge
    - Optimiza tipos de datos
    - Evita copias innecesarias
    """
    # Identificar columnas necesarias para build_features
    # Estas columnas se usan: cmc, rarity, color_identity, type, power, toughness, text (para keywords)
    required_cols = ['name', 'cmc', 'rarity', 'color_identity', 'type', 'power', 'toughness', 'text']
    
    # Asegurar que todas las columnas requeridas existen
    available_cols = [col for col in required_cols if col in norm_cards.columns]
    if 'name' not in available_cols:
        raise ValueError("La columna 'name' es requerida en norm_cards")
    
    # Seleccionar solo columnas necesarias de norm_cards para reducir memoria
    norm_cards_subset = norm_cards[available_cols].copy()
    
    # Optimizar tipos de datos antes del merge
    # Convertir strings a category cuando hay repetición
    if 'name' in norm_cards_subset.columns and len(norm_cards_subset) > 0:
        if norm_cards_subset['name'].nunique() < len(norm_cards_subset):
            norm_cards_subset['name'] = norm_cards_subset['name'].astype('category')
    
    # Hacer el merge solo con las columnas necesarias
    merged = deck_cards_intermediate.merge(
        norm_cards_subset, 
        left_on="card_name", 
        right_on="name", 
        how="left"
    )
    
    # Eliminar columna duplicada 'name' después del merge
    merged.drop(columns=["name"], inplace=True, errors="ignore")
    
    # Optimizar tipos de datos después del merge
    # Convertir cmc a float32 para ahorrar memoria si es numérica
    if 'cmc' in merged.columns:
        merged['cmc'] = pd.to_numeric(merged['cmc'], errors='coerce').astype('float32')
    
    # Normalizar columna 'loyalty' para evitar errores de tipo mixto (si existe)
    if "loyalty" in merged.columns:
        merged["loyalty"] = merged["loyalty"].fillna("").astype(str)
    
    # Limpiar memoria intermedia explícitamente
    del norm_cards_subset
    
    return merged

def build_features(df_merged: pd.DataFrame, df_decks: pd.DataFrame) -> pd.DataFrame:
    rarity_map = {'common':1,'uncommon':2,'rare':3,'mythic':4}
    df = df_merged.copy()
    df['rarity_value'] = df['rarity'].astype(str).str.lower().map(rarity_map)
    
    # Calcular Power Score determinísticamente (FEATURE, no target) - VECTORIZADO
    # Power Score combina: power, toughness, cmc, rarity, y tipo de carta
    # Convertir power y toughness a numéricos (vectorizado)
    power_clean = df['power'].astype(str).str.replace('*', '0', regex=False)
    toughness_clean = df['toughness'].astype(str).str.replace('*', '0', regex=False)
    power_num = pd.to_numeric(power_clean, errors='coerce')
    toughness_num = pd.to_numeric(toughness_clean, errors='coerce')
    
    # Liberar memoria inmediatamente después de usar
    del power_clean, toughness_clean
    
    # Inicializar score
    df['card_power_score'] = 0.0
    
    # Power y Toughness: eficiencia (power + toughness) / cmc * 0.5
    valid_creature = power_num.notna() & toughness_num.notna()
    cmc = df['cmc'].fillna(1.0)
    p_t_sum = power_num + toughness_num
    creature_efficiency = pd.Series(0.0, index=df.index, dtype='float64')
    mask_valid_cmc = valid_creature & (cmc > 0)
    mask_valid_zero = valid_creature & (cmc <= 0)
    creature_efficiency.loc[mask_valid_cmc] = (p_t_sum / cmc * 0.5).loc[mask_valid_cmc].astype('float64')
    creature_efficiency.loc[mask_valid_zero] = (p_t_sum * 0.3).loc[mask_valid_zero].astype('float64')
    df['card_power_score'] += creature_efficiency.fillna(0.0)
    
    # Liberar memoria
    del power_num, toughness_num, creature_efficiency, p_t_sum, mask_valid_cmc, mask_valid_zero, valid_creature, cmc
    
    # Rareza (peso)
    df['card_power_score'] += df['rarity_value'].fillna(1.0) * 0.2
    
    # Tipo de carta (vectorizado)
    card_type_lower = df['type'].astype(str).str.lower()
    type_bonus = pd.Series(0.0, index=df.index)
    type_bonus.loc[card_type_lower.str.contains('planeswalker', na=False)] = 2.0
    type_bonus.loc[card_type_lower.str.contains('instant|sorcery', na=False, regex=True) & 
                   ~card_type_lower.str.contains('planeswalker', na=False)] = 0.5
    type_bonus.loc[card_type_lower.str.contains('creature', na=False) & 
                   (type_bonus == 0.0)] = 0.3
    df['card_power_score'] += type_bonus
    
    # Liberar memoria
    del card_type_lower, type_bonus

    # Calcular Power Score agregado por mazo (VECTORIZADO)
    # Promedio ponderado: sum(score * quantity) / sum(quantity)
    df['weighted_score'] = df['card_power_score'] * df['quantity']
    deck_power_score_numerator = df.groupby('deck_name', observed=False)['weighted_score'].sum()
    deck_power_score_denominator = df.groupby('deck_name', observed=False)['quantity'].sum()
    deck_power_scores = (deck_power_score_numerator / deck_power_score_denominator).fillna(0.0).reset_index(name='power_score')
    
    # Liberar columna temporal inmediatamente
    df.drop(columns=['weighted_score'], inplace=True)
    del deck_power_score_numerator, deck_power_score_denominator
    
    # ============================================================================
    # FEATURES RECOMENDADAS POR EL EXPERTO - MEJORAS PRIORITARIAS
    # ============================================================================
    
    # 1. CURVA DE MANÁ: varianza de CMC (además del promedio)
    agg_cmc = df.groupby('deck_name', observed=False).agg(
        avg_cmc=('cmc','mean'),
        std_cmc=('cmc','std'),
        var_cmc=('cmc','var')
    ).reset_index()
    agg_cmc['std_cmc'] = agg_cmc['std_cmc'].fillna(0.0)
    agg_cmc['var_cmc'] = agg_cmc['var_cmc'].fillna(0.0)
    
    # 2. RAREZA: media ponderada (ya tenemos avg_rarity, pero mejoramos con peso)
    df['weighted_rarity'] = df['rarity_value'] * df['quantity']
    agg_rarity = df.groupby('deck_name', observed=False).agg(
        total_rarity_weighted=('weighted_rarity','sum'),
        total_cards=('quantity','sum')
    ).reset_index()
    agg_rarity['avg_rarity_weighted'] = (agg_rarity['total_rarity_weighted'] / agg_rarity['total_cards']).fillna(0.0)
    agg_rarity = agg_rarity[['deck_name', 'total_cards', 'avg_rarity_weighted']]
    df.drop(columns=['weighted_rarity'], inplace=True)
    
    # 3. TIPOS: % criaturas, instant, sorcery, planeswalker, enchantment, artifact
    type_lower = df['type'].astype(str).str.lower()
    df['is_creature'] = type_lower.str.contains('creature', na=False).astype(int) * df['quantity']
    df['is_instant'] = type_lower.str.contains('instant', na=False).astype(int) * df['quantity']
    df['is_sorcery'] = type_lower.str.contains('sorcery', na=False).astype(int) * df['quantity']
    df['is_planeswalker'] = type_lower.str.contains('planeswalker', na=False).astype(int) * df['quantity']
    df['is_enchantment'] = type_lower.str.contains('enchantment', na=False).astype(int) * df['quantity']
    df['is_artifact'] = type_lower.str.contains('artifact', na=False).astype(int) * df['quantity']
    
    agg_types_pct = df.groupby('deck_name', observed=False).agg(
        total_creatures=('is_creature','sum'),
        total_instants=('is_instant','sum'),
        total_sorceries=('is_sorcery','sum'),
        total_planeswalkers=('is_planeswalker','sum'),
        total_enchantments=('is_enchantment','sum'),
        total_artifacts=('is_artifact','sum'),
        total_cards_for_pct=('quantity','sum')
    ).reset_index()
    
    agg_types_pct['pct_creatures'] = (agg_types_pct['total_creatures'] / agg_types_pct['total_cards_for_pct'] * 100).fillna(0.0)
    agg_types_pct['pct_instants'] = (agg_types_pct['total_instants'] / agg_types_pct['total_cards_for_pct'] * 100).fillna(0.0)
    agg_types_pct['pct_sorceries'] = (agg_types_pct['total_sorceries'] / agg_types_pct['total_cards_for_pct'] * 100).fillna(0.0)
    agg_types_pct['pct_planeswalkers'] = (agg_types_pct['total_planeswalkers'] / agg_types_pct['total_cards_for_pct'] * 100).fillna(0.0)
    agg_types_pct['pct_enchantments'] = (agg_types_pct['total_enchantments'] / agg_types_pct['total_cards_for_pct'] * 100).fillna(0.0)
    agg_types_pct['pct_artifacts'] = (agg_types_pct['total_artifacts'] / agg_types_pct['total_cards_for_pct'] * 100).fillna(0.0)
    agg_types_pct = agg_types_pct[['deck_name', 'pct_creatures', 'pct_instants', 'pct_sorceries', 
                                    'pct_planeswalkers', 'pct_enchantments', 'pct_artifacts']]
    
    # Liberar columnas temporales
    df.drop(columns=['is_creature', 'is_instant', 'is_sorcery', 'is_planeswalker', 
                     'is_enchantment', 'is_artifact'], inplace=True)
    del type_lower
    
    # 4. KEYWORDS EN TEXT: removal, draw, ramp, counter, lifegain
    if 'text' in df.columns:
        text_lower = df['text'].astype(str).str.lower()
        # Detección de keywords comunes
        df['has_removal'] = (text_lower.str.contains('destroy|exile|sacrifice|shuffle', na=False, regex=True) | 
                           text_lower.str.contains('damage', na=False)).astype(int) * df['quantity']
        df['has_draw'] = text_lower.str.contains('draw', na=False).astype(int) * df['quantity']
        df['has_ramp'] = text_lower.str.contains('land|mana', na=False, regex=True).astype(int) * df['quantity']
        df['has_counter'] = text_lower.str.contains('counter', na=False).astype(int) * df['quantity']
        df['has_lifegain'] = text_lower.str.contains('life|gain', na=False, regex=True).astype(int) * df['quantity']
        
        agg_keywords = df.groupby('deck_name', observed=False).agg(
            total_removal=('has_removal','sum'),
            total_draw=('has_draw','sum'),
            total_ramp=('has_ramp','sum'),
            total_counter=('has_counter','sum'),
            total_lifegain=('has_lifegain','sum'),
            total_cards_for_keywords=('quantity','sum')
        ).reset_index()
        
        agg_keywords['pct_removal'] = (agg_keywords['total_removal'] / agg_keywords['total_cards_for_keywords'] * 100).fillna(0.0)
        agg_keywords['pct_draw'] = (agg_keywords['total_draw'] / agg_keywords['total_cards_for_keywords'] * 100).fillna(0.0)
        agg_keywords['pct_ramp'] = (agg_keywords['total_ramp'] / agg_keywords['total_cards_for_keywords'] * 100).fillna(0.0)
        agg_keywords['pct_counter'] = (agg_keywords['total_counter'] / agg_keywords['total_cards_for_keywords'] * 100).fillna(0.0)
        agg_keywords['pct_lifegain'] = (agg_keywords['total_lifegain'] / agg_keywords['total_cards_for_keywords'] * 100).fillna(0.0)
        agg_keywords = agg_keywords[['deck_name', 'pct_removal', 'pct_draw', 'pct_ramp', 'pct_counter', 'pct_lifegain']]
        
        df.drop(columns=['has_removal', 'has_draw', 'has_ramp', 'has_counter', 'has_lifegain'], inplace=True)
        del text_lower
    else:
        # Si no hay columna text, crear features vacías
        agg_keywords = pd.DataFrame({'deck_name': df['deck_name'].unique()})
        agg_keywords['pct_removal'] = 0.0
        agg_keywords['pct_draw'] = 0.0
        agg_keywords['pct_ramp'] = 0.0
        agg_keywords['pct_counter'] = 0.0
        agg_keywords['pct_lifegain'] = 0.0
    
    # 5. CUERPO DE MESA: media power/toughness de criaturas (ponderado por copias)
    # Recalcular power y toughness para criaturas
    power_clean_2 = df['power'].astype(str).str.replace('*', '0', regex=False)
    toughness_clean_2 = df['toughness'].astype(str).str.replace('*', '0', regex=False)
    power_num_2 = pd.to_numeric(power_clean_2, errors='coerce')
    toughness_num_2 = pd.to_numeric(toughness_clean_2, errors='coerce')
    del power_clean_2, toughness_clean_2
    
    is_creature_mask = df['type'].astype(str).str.lower().str.contains('creature', na=False)
    df['power_weighted'] = (power_num_2 * df['quantity']).where(is_creature_mask, 0)
    df['toughness_weighted'] = (toughness_num_2 * df['quantity']).where(is_creature_mask, 0)
    df['creature_quantity'] = df['quantity'].where(is_creature_mask, 0)
    
    agg_creatures = df.groupby('deck_name', observed=False).agg(
        total_power_weighted=('power_weighted','sum'),
        total_toughness_weighted=('toughness_weighted','sum'),
        total_creature_copies=('creature_quantity','sum')
    ).reset_index()
    
    agg_creatures['avg_power'] = (agg_creatures['total_power_weighted'] / agg_creatures['total_creature_copies']).replace([np.inf, -np.inf], 0).fillna(0.0)
    agg_creatures['avg_toughness'] = (agg_creatures['total_toughness_weighted'] / agg_creatures['total_creature_copies']).replace([np.inf, -np.inf], 0).fillna(0.0)
    agg_creatures = agg_creatures[['deck_name', 'avg_power', 'avg_toughness']]
    
    df.drop(columns=['power_weighted', 'toughness_weighted', 'creature_quantity'], inplace=True)
    del power_num_2, toughness_num_2, is_creature_mask
    
    # 6. MANA EFFICIENCY SCORE: (power + toughness) / cmc ponderado por curva
    # Ya tenemos parte de esto en card_power_score, pero agregamos versión específica
    power_clean_3 = df['power'].astype(str).str.replace('*', '0', regex=False)
    toughness_clean_3 = df['toughness'].astype(str).str.replace('*', '0', regex=False)
    power_num_3 = pd.to_numeric(power_clean_3, errors='coerce')
    toughness_num_3 = pd.to_numeric(toughness_clean_3, errors='coerce')
    del power_clean_3, toughness_clean_3
    
    valid_stats = power_num_3.notna() & toughness_num_3.notna()
    cmc_valid = df['cmc'].fillna(1.0)
    p_t_sum_2 = power_num_3 + toughness_num_3
    mana_efficiency = pd.Series(0.0, index=df.index, dtype='float64')
    mask_valid = valid_stats & (cmc_valid > 0)
    mana_efficiency.loc[mask_valid] = (p_t_sum_2 / cmc_valid).loc[mask_valid]
    
    df['mana_efficiency_score'] = mana_efficiency.fillna(0.0) * df['quantity']
    
    agg_mana_eff = df.groupby('deck_name', observed=False).agg(
        total_mana_eff_weighted=('mana_efficiency_score','sum'),
        total_cards_eff=('quantity','sum')
    ).reset_index()
    agg_mana_eff['mana_efficiency'] = (agg_mana_eff['total_mana_eff_weighted'] / agg_mana_eff['total_cards_eff']).fillna(0.0)
    agg_mana_eff = agg_mana_eff[['deck_name', 'mana_efficiency']]
    
    df.drop(columns=['mana_efficiency_score'], inplace=True)
    del power_num_3, toughness_num_3, p_t_sum_2, mana_efficiency, valid_stats, cmc_valid, mask_valid
    
    # 7. IDENTIDAD DE COLOR más detallada: mono, 2-color, 3+
    df['color_count'] = df['color_identity'].astype(str).str.replace(r'[\[\]]', '', regex=True).str.split(',').str.len()
    df['color_category'] = pd.cut(df['color_count'], bins=[0, 1, 2, 10], labels=['mono', 'two_color', 'three_plus'], include_lowest=True).astype(str)
    # Contar categorías por deck (ponderado por cantidad)
    color_cat_counts = df.groupby(['deck_name', 'color_category'], observed=False)['quantity'].sum().reset_index()
    color_cat_pivot = color_cat_counts.pivot_table(index='deck_name', columns='color_category', values='quantity', fill_value=0).reset_index()
    total_per_deck = color_cat_pivot[['mono', 'two_color', 'three_plus']].sum(axis=1)
    color_cat_pivot['pct_mono'] = (color_cat_pivot['mono'] / total_per_deck * 100).fillna(0.0)
    color_cat_pivot['pct_two_color'] = (color_cat_pivot['two_color'] / total_per_deck * 100).fillna(0.0)
    color_cat_pivot['pct_three_plus'] = (color_cat_pivot['three_plus'] / total_per_deck * 100).fillna(0.0)
    agg_colors = color_cat_pivot[['deck_name', 'pct_mono', 'pct_two_color', 'pct_three_plus']].copy()
    
    # También mantener avg_colors original
    avg_colors_simple = df.groupby('deck_name', observed=False)['color_count'].mean().reset_index(name='avg_colors')
    agg_colors = agg_colors.merge(avg_colors_simple, on='deck_name', how='left')
    
    df.drop(columns=['color_count', 'color_category'], inplace=True)
    
    # 8. UNIQUE TYPES: versión simplificada sin lotes (más rápido y menos memoria)
    def get_unique_types_simple(group):
        """Procesa tipos únicos de forma eficiente sin crear copias grandes
        group es una Serie cuando se hace groupby()['type'], no un DataFrame
        """
        types_all = []
        for type_str in group.dropna().astype(str):
            if type_str and type_str.lower() != 'nan':
                types_all.extend(t.strip() for t in type_str.split() if t.strip())
        return len(set(types_all))
    
    agg_types = df.groupby('deck_name', observed=False)['type'].apply(get_unique_types_simple).reset_index(name='unique_types')
    
    # ============================================================================
    # MERGE DE TODAS LAS FEATURES
    # ============================================================================
    
    # Agregaciones básicas iniciales
    agg_features = agg_cmc.merge(agg_rarity, on='deck_name', how='left')
    
    # Merge con Power Score
    agg_features = agg_features.merge(deck_power_scores, on='deck_name', how='left')
    agg_features['power_score'] = agg_features['power_score'].fillna(0.0)
    del deck_power_scores
    
    # Merge todas las features adicionales
    agg_features = agg_features.merge(agg_types_pct, on='deck_name', how='left')
    agg_features = agg_features.merge(agg_keywords, on='deck_name', how='left')
    agg_features = agg_features.merge(agg_creatures, on='deck_name', how='left')
    agg_features = agg_features.merge(agg_mana_eff, on='deck_name', how='left')
    agg_features = agg_features.merge(agg_colors, on='deck_name', how='left')
    agg_features = agg_features.merge(agg_types, on='deck_name', how='left')
    
    # Fill NaN con 0 para todas las nuevas features
    new_features_cols = ['std_cmc', 'var_cmc', 'avg_rarity_weighted', 'pct_creatures', 'pct_instants', 
                         'pct_sorceries', 'pct_planeswalkers', 'pct_enchantments', 'pct_artifacts',
                         'pct_removal', 'pct_draw', 'pct_ramp', 'pct_counter', 'pct_lifegain',
                         'avg_power', 'avg_toughness', 'mana_efficiency', 'pct_mono', 'pct_two_color',
                         'pct_three_plus', 'unique_types']
    for col in new_features_cols:
        if col in agg_features.columns:
            agg_features[col] = agg_features[col].fillna(0.0)
    
    # Liberar memoria del DataFrame principal
    del df

    # Merge con información de decks (Tier, Year)
    features = agg_features.merge(df_decks[['Name','Tier','Year']], left_on='deck_name', right_on='Name', how='left')
    features.drop(columns=['Name'], inplace=True)
    
    # ============================================================================
    # CREAR TARGETS: Binaria, Regresión, y Multiclase (Tier 1-5)
    # ============================================================================
    
    # 1. Clasificación binaria (competitive: 1 si Tier 1 o 2, 0 caso contrario)
    features['competitive'] = features['Tier'].astype(str).str.contains(r'^[12]$', regex=True, na=False).astype(int)
    
    # 2. Clasificación multiclase por Tier (1-5) - RECOMENDACIÓN DEL EXPERTO
    tier_numeric = pd.to_numeric(features['Tier'], errors='coerce').clip(1, 5).fillna(3)
    features['tier_multiclass'] = tier_numeric.astype(int)
    # Nota: Los modelos de scikit-learn/XGBoost esperan clases 0-4, pero guardamos 1-5 para interpretación
    
    # 3. Regresión: Competitiveness Score
    # Score continuo basado en Tier con variación controlada para que el modelo aprenda
    # Tier 5 (mejor) → score alto, Tier 1 (peor) → score bajo
    # Agregamos ruido aleatorio pequeño para simular variabilidad real
    np.random.seed(42)  # Para reproducibilidad
    base_scores = tier_numeric.map({1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0})
    # Variación aleatoria pequeña (±5%)
    noise = np.random.uniform(-0.05, 0.05, size=len(base_scores))
    features['competitiveness_score'] = (base_scores + noise).clip(0.0, 1.0)
    
    # Eliminar columnas no numéricas que no son features
    features = features.drop(columns=['deck_name', 'Tier', 'Year'])
    
    return features

def split_train_test(features, tier_positive_regex, test_size, random_state):
    """
    Divide los datos en train/test para clasificación binaria y regresión.
    Excluye tier_multiclass (solo se calcula pero no se usa en pipelines de Airflow).
    """
    df = features.copy()
    
    # Columnas a excluir de las features X (targets y variables no usadas)
    cols_to_drop = ["competitive", "competitiveness_score", "tier_multiclass"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    # Clasificación binaria
    X_cls = df.drop(columns=cols_to_drop)
    y_cls = df["competitive"]

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_cls, y_cls, test_size=test_size, random_state=random_state, stratify=y_cls
    )

    # Regresión (Competitiveness Score continuo)
    X_reg = df.drop(columns=cols_to_drop)
    y_reg = df["competitiveness_score"]

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=test_size, random_state=random_state
    )

    # 🔧 Convertir Series → DataFrame con reset_index para asegurar índices consecutivos
    y_train_cls = pd.DataFrame(y_train_cls.values, columns=["competitive"], index=y_train_cls.index).reset_index(drop=True)
    y_test_cls = pd.DataFrame(y_test_cls.values, columns=["competitive"], index=y_test_cls.index).reset_index(drop=True)
    y_train_reg = pd.DataFrame(y_train_reg.values, columns=["competitiveness_score"], index=y_train_reg.index).reset_index(drop=True)
    y_test_reg = pd.DataFrame(y_test_reg.values, columns=["competitiveness_score"], index=y_test_reg.index).reset_index(drop=True)
    
    # También resetear índices en X para consistencia
    X_train_cls = X_train_cls.reset_index(drop=True)
    X_test_cls = X_test_cls.reset_index(drop=True)
    X_train_reg = X_train_reg.reset_index(drop=True)
    X_test_reg = X_test_reg.reset_index(drop=True)

    return (X_train_cls, X_test_cls, y_train_cls, y_test_cls,
            X_train_reg, X_test_reg, y_train_reg, y_test_reg)
