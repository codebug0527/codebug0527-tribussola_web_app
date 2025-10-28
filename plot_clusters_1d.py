import matplotlib.pyplot as plt
import numpy as np

def plot_clusters_1d(x, s, idx, title=None, save_path=None):
    """
    Plota clusters 1D com barras de erro horizontais.
    
    Args:
        x: array com valores (Zranking)
        s: array com incertezas (s_Zranking) 
        idx: array com labels dos clusters (0 = ruído/noise)
        title: título do gráfico (opcional)
        save_path: caminho para salvar a figura (opcional)
    """
    x = np.array(x).flatten()
    s = np.array(s).flatten()
    idx = np.array(idx).flatten()
    
    N = len(x)
    K = max(idx) if len(idx) > 0 else 0
    
    # Cria figura
    plt.figure(figsize=(12, 6), facecolor='white')
    
    # y fictício só para separar pontos
    y = np.zeros(N)
    jitter = 0.02 * (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0.1
    
    # Adiciona um pouco de jitter vertical para melhor visualização
    y += np.random.normal(0, jitter/3, N)
    
    # Plota cada ponto
    for i in range(N):
        k = idx[i]
        if k == -1 or k == 0:  # -1 ou 0 = ruído (DBSCAN usa -1, pode ser adaptado)
            color = [0.5, 0.5, 0.5]  # cinza
            marker = 'o'
            marker_size = 36
        else:
            color = plt.cm.tab10((k-1) % 10)  # cores distintas
            marker = 'o'
            marker_size = 48
        
        # Barras de erro horizontais (±s)
        plt.plot([x[i] - s[i], x[i] + s[i]], [y[i], y[i]], 
                color=color, linewidth=1.4, alpha=0.8)
        
        # Ponto central
        plt.scatter(x[i], y[i], s=marker_size, c=[color], 
                   marker=marker, edgecolors='black', linewidth=0.5, alpha=0.8)
    
    plt.xlabel('Zranking', fontsize=12)
    plt.yticks([])
    plt.ylim([-1, 1])
    plt.grid(True, alpha=0.3)
    
    if title is None:
        title = f'Clusters (K={K}; cinza = ruído)'
    plt.title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[info] Gráfico salvo em: {save_path}")
    
    plt.show()

# Versão alternativa que aceita DataFrame diretamente (mais conveniente)
def plot_clusters_1d_from_df(df, cluster_col='cluster', title=None, save_path=None):
    """
    Versão que aceita DataFrame diretamente (compatível com a saída do código anterior).
    
    Args:
        df: DataFrame com colunas 'Zranking', 's_Zranking' e cluster_col
        cluster_col: nome da coluna com os labels dos clusters
        title: título do gráfico
        save_path: caminho para salvar
    """
    if 'Zranking' not in df.columns or 's_Zranking' not in df.columns:
        raise ValueError("DataFrame deve conter colunas 'Zranking' e 's_Zranking'")
    
    if cluster_col not in df.columns:
        raise ValueError(f"DataFrame deve conter coluna '{cluster_col}'")
    
    x = df['Zranking'].values
    s = df['s_Zranking'].values
    idx = df[cluster_col].values
    
    plot_clusters_1d(x, s, idx, title, save_path)

# Função específica para GMM com probabilidades
def plot_gmm_clusters_1d(df, cluster_col='cluster', prob_col='cluster_probability', 
                        title=None, save_path=None):
    """
    Versão especializada para GMM que mostra probabilidades pelo tamanho dos pontos.
    
    Args:
        df: DataFrame com resultados do GMM
        cluster_col: coluna com labels dos clusters  
        prob_col: coluna com probabilidades (opcional)
        title: título do gráfico
        save_path: caminho para salvar
    """
    x = df['Zranking'].values
    s = df['s_Zranking'].values
    idx = df[cluster_col].values
    
    N = len(x)
    K = max(idx) if len(idx) > 0 else 0
    
    plt.figure(figsize=(14, 6), facecolor='white')
    
    # y fictício
    y = np.zeros(N)
    jitter = 0.02 * (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0.1
    y += np.random.normal(0, jitter/3, N)
    
    # Se temos probabilidades, usa para tamanho dos pontos
    if prob_col in df.columns:
        probs = df[prob_col].values
        sizes = 20 + 80 * probs  # tamanho proporcional à probabilidade
    else:
        sizes = 48 * np.ones(N)
    
    for i in range(N):
        k = idx[i]
        if k == -1 or k == 0:  # ruído
            color = [0.5, 0.5, 0.5]
            marker = 'o'
            edge_color = 'black'
        else:
            color = plt.cm.tab10((k-1) % 10)
            marker = 'o'
            edge_color = 'black'
        
        # Barras de erro
        plt.plot([x[i] - s[i], x[i] + s[i]], [y[i], y[i]], 
                color=color, linewidth=1.4, alpha=0.6)
        
        # Ponto (tamanho baseado na probabilidade se disponível)
        plt.scatter(x[i], y[i], s=sizes[i], c=[color], 
                   marker=marker, edgecolors=edge_color, linewidth=0.8, alpha=0.8)
    
    plt.xlabel('Zranking', fontsize=12)
    plt.yticks([])
    plt.ylim([-1, 1])
    plt.grid(True, alpha=0.3)
    
    if title is None:
        prob_text = " (tamanho = probabilidade)" if prob_col in df.columns else ""
        title = f'Clusters GMM (K={K}; cinza = ruído){prob_text}'
    plt.title(title, fontsize=14)
    
    # Legenda para probabilidades se disponível
    if prob_col in df.columns:
        plt.text(0.02, 0.98, 'Tamanho ∝ probabilidade do cluster', 
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[info] Gráfico GMM salvo em: {save_path}")
    
    plt.show()