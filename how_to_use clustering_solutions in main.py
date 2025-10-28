#how_to_use clustering_solutions in main

# No seu main(), substitua a chamada do clustering:
labels, cluster_meta, probabilities = cluster_indices_gmm(
    out, use_uncertainty=True
)

if labels is not None:
    out = out.copy()
    out["cluster"] = labels
    out["cluster_probability"] = np.max(probabilities, axis=1)
    
    # Salva resultados detalhados
    with open("gmm_clusters_meta.json", "w", encoding="utf-8") as f:
        json.dump(cluster_meta, f, ensure_ascii=False, indent=2)
    
    print(f"[info] GMM encontrou {cluster_meta['n_components']} clusters", file=sys.stderr)