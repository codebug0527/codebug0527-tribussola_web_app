import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import json

def find_optimal_gmm_components(out: pd.DataFrame, max_components: int = 8, 
                               use_uncertainty: bool = True):
    """
    Encontra o número ótimo de componentes usando BIC e Silhouette Score.
    """
    X = out[['Zranking']].values
    
    bics = []
    aics = []
    silhouette_scores = []
    
    for n in range(1, max_components + 1):
        if n > len(X):  # Não pode ter mais clusters que pontos
            break
            
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
        
        if use_uncertainty and 's_Zranking' in out.columns:
            weights = 1 / (out['s_Zranking'] ** 2 + 1e-8)
            weights = weights / weights.sum()
            gmm.fit(X, weights=weights)
        else:
            gmm.fit(X)
        
        labels = gmm.predict(X)
        
        # Calcula métricas
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))
        
        # Silhouette score só faz sentido para 2+ clusters
        if n > 1 and len(np.unique(labels)) > 1:
            silhouette_scores.append(silhouette_score(X, labels))
        else:
            silhouette_scores.append(-1)  # Valor inválido
    
    # Encontra ótimos por diferentes critérios
    optimal_bic = np.argmin(bics) + 1
    optimal_aic = np.argmin(aics) + 1
    
    # Para silhouette, ignora valores inválidos (-1)
    valid_silhouette = [s for s in silhouette_scores if s >= 0]
    if valid_silhouette:
        optimal_silhouette = np.argmax(valid_silhouette) + 2  # +2 porque começa de 2 clusters
    else:
        optimal_silhouette = optimal_bic
    
    # Combina os critérios (prioriza BIC geralmente)
    optimal_n = optimal_bic  # Ou pode usar uma votação
    
    # Plot das métricas
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(bics) + 1), bics, 'o-', label='BIC')
    plt.axvline(optimal_bic, color='red', linestyle='--', alpha=0.7, label=f'Ótimo BIC: {optimal_bic}')
    plt.title('Bayesian Information Criterion')
    plt.xlabel('Número de Clusters')
    plt.ylabel('BIC (menor é melhor)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(aics) + 1), aics, 'o-', label='AIC', color='orange')
    plt.axvline(optimal_aic, color='red', linestyle='--', alpha=0.7, label=f'Ótimo AIC: {optimal_aic}')
    plt.title('Akaike Information Criterion')
    plt.xlabel('Número de Clusters')
    plt.ylabel('AIC (menor é melhor)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    valid_range = range(2, len(silhouette_scores) + 1)
    valid_scores = silhouette_scores[1:]  # Remove o primeiro elemento (1 cluster)
    plt.plot(valid_range, valid_scores, 'o-', label='Silhouette', color='green')
    if valid_silhouette:
        plt.axvline(optimal_silhouette, color='red', linestyle='--', alpha=0.7, 
                   label=f'Ótimo Silhouette: {optimal_silhouette}')
    plt.title('Silhouette Score')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Silhouette (maior é melhor)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Número ótimo de clusters:")
    print(f"  - Por BIC: {optimal_bic}")
    print(f"  - Por AIC: {optimal_aic}")
    if valid_silhouette:
        print(f"  - Por Silhouette: {optimal_silhouette}")
    print(f"  - Selecionado: {optimal_n}")
    
    return optimal_n

def gmm_cluster_auto(out: pd.DataFrame, use_uncertainty: bool = True, 
                    max_components: int = 8, random_state: int = 42):
    """
    Clustering com GMM encontrando automaticamente o número ótimo de clusters.
    """
    # Encontra número ótimo de clusters
    n_components = find_optimal_gmm_components(out, max_components, use_uncertainty)
    
    X = out[['Zranking']].values
    
    if use_uncertainty and 's_Zranking' in out.columns:
        # Usa incertezas como pesos
        weights = 1 / (out['s_Zranking'] ** 2 + 1e-8)
        weights = weights / weights.sum()
        
        gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=10)
        gmm.fit(X, weights=weights)
    else:
        gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=10)
        gmm.fit(X)
    
    labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)
    
    # Metadados do modelo
    cluster_meta = {
        "method": "GaussianMixture",
        "n_components": n_components,
        "uses_uncertainty": use_uncertainty,
        "means": gmm.means_.flatten().tolist(),
        "weights": gmm.weights_.tolist(),
        "covariances": gmm.covariances_.flatten().tolist(),
        "bic": gmm.bic(X),
        "aic": gmm.aic(X),
        "converged": gmm.converged_
    }
    
    return labels, cluster_meta, probabilities, gmm

# Função para integrar no seu código
def cluster_indices_gmm(out: pd.DataFrame, use_uncertainty: bool = True):
    """
    Substituição direta da função original de clustering.
    """
    try:
        labels, cluster_meta, probabilities, gmm = gmm_cluster_auto(
            out, use_uncertainty=use_uncertainty
        )
        
        return labels, cluster_meta, probabilities
        
    except Exception as e:
        print(f"[aviso] GMM falhou: {e}. Retornando None.", file=sys.stderr)
        return None, None, None