# -*- coding: utf-8 -*-

def reduce_to_2d_tsne(embeddings, debug=False):
    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2,
        # early_exaggeration=12,
        learning_rate='auto',
        metric='cosine',
        init='pca',
        verbose=2 if debug else 0,
        n_iter=1000,
        random_state=42,
        method='barnes_hut',
        n_jobs=-1)
    embeddings_2d = tsne_model.fit_transform(embeddings)
    return embeddings_2d

def reduce_to_2d_tsne_cuml(embeddings, debug=False):
    from cuml.manifold import TSNE
    tsne_model = TSNE(n_components=2,
        # early_exaggeration=12,
        learning_rate_method='adaptive',
        metric='cosine',
        # init='pca',?
        verbose=debug,
        n_iter=1000,
        random_state=42,
        method='barnes_hut')
    embeddings_2d = tsne_model.fit_transform(embeddings)
    return embeddings_2d

def reduce_to_2d_umap(embeddings, debug=False):
    import warnings
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    import umap

    umap_model = umap.UMAP(n_components=2,
        n_neighbors=40,
        min_dist=7,
        spread=7,
        verbose=debug,
        random_state=42,
        metric='cosine')
    embeddings_2d = umap_model.fit_transform(embeddings)
    return embeddings_2d
