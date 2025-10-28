##how_to_use plot_clusters_1d with clustering_solutions.py

# Após o clustering GMM, você pode plotar diretamente:

# Método 1: Com DataFrame completo
plot_clusters_1d_from_df(out, cluster_col='cluster', 
                        title='Clusters GMM com Incertezas',
                        save_path='clusters_gmm.png')

# Método 2: Versão GMM com probabilidades
plot_gmm_clusters_1d(out, cluster_col='cluster', 
                    prob_col='cluster_probability',
                    title='Clusters GMM - Tamanho = Confiança',
                    save_path='clusters_gmm_detailed.png')

# Método 3: Com arrays separados (como no MATLAB original)
plot_clusters_1d(out['Zranking'].values, 
                out['s_Zranking'].values, 
                out['cluster'].values,
                title='Clusters 1D')