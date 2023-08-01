# -*- coding: utf-8 -*-

import multiprocessing

from vocab_coverage.utils import logger

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
        n_neighbors=30,
        min_dist=0.6,
        spread=1,
        verbose=debug,
        random_state=42,
        metric='cosine')
    embeddings_2d = umap_model.fit_transform(embeddings)
    return embeddings_2d

def reduce_to_2d_umap_cuml(embeddings, debug=False):
    import cuml
    umap_model = cuml.UMAP(n_components=2,
        n_neighbors=30,
        min_dist=1,
        spread=1,
        verbose=debug,
        random_state=42,
        metric='cosine')
    embeddings_2d = umap_model.fit_transform(embeddings)
    return embeddings_2d

def reduce_to_2d_umap_tsne(embeddings, debug=False):
    # Use UMAP reduce to 30 dimensions, then TSNE to 2
    import warnings
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    import umap
    umap_model = umap.UMAP(n_components=30,
        n_neighbors=30,
        min_dist=0.6,
        spread=1,
        verbose=debug,
        random_state=42,
        metric='cosine')
    embeddings_30 = umap_model.fit_transform(embeddings)
    # TSNE
    embeddings_2d = reduce_to_2d_tsne(embeddings_30, debug)
    return embeddings_2d


def reduce_to_2d_umap_tsne_cuml(embeddings, debug=False):
    # Use UMAP reduce to 30 dimensions, then TSNE to 2
    import warnings
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    import umap
    umap_model = umap.UMAP(n_components=30,
        n_neighbors=30,
        min_dist=0.6,
        spread=1,
        verbose=debug,
        random_state=42,
        metric='cosine')
    embeddings_30 = umap_model.fit_transform(embeddings)
    # TSNE
    embeddings_2d = reduce_to_2d_tsne_cuml(embeddings_30, debug)
    return embeddings_2d

def reduce_to_2d_umap_tsne_cuml_both(embeddings, debug=False):
    # Use UMAP reduce to 30 dimensions, then TSNE to 2
    import warnings
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    import cuml
    umap_model = cuml.UMAP(n_components=30,
        n_neighbors=30,
        min_dist=0.6,
        spread=1,
        verbose=debug,
        random_state=42,
        metric='cosine')
    embeddings_30 = umap_model.fit_transform(embeddings)
    # TSNE
    embeddings_2d = reduce_to_2d_tsne_cuml(embeddings_30, debug)
    return embeddings_2d

def with_timeout(func, timeout, *args, **kwargs):
    logger.debug(f"> call {func} with timeout: {timeout}s")
    
    current_process = multiprocessing.current_process()
    timeout_event = multiprocessing.Event()

    def monitor(timeout=300):
        if not timeout_event.wait(timeout):
            logger.debug("with_timeout():monitor(): timeout")
            if current_process.is_alive():
                logger.error(f"> {func} timeout: {timeout}s")
                current_process.terminate()
                current_process.join()
        else:
            logger.debug("with_timeout():monitor(): timeout_event.set()")
    
    monitor_process = multiprocessing.Process(target=monitor, args=(timeout,))
    result = func(*args, **kwargs)
    timeout_event.set()

    if monitor_process.is_alive():
        monitor_process.terminate()
        monitor_process.join()

    return result

def reduce_to_2d(embeddings, method='tsne', debug=False):
    if debug:
        logger.debug(f"> reducing the dimension of {embeddings.shape} to 2D by {method}...")
    reducer = None
    set_timeout = False
    if method == 'tsne':
        reducer = reduce_to_2d_tsne
        set_timeout = False
    elif method == 'tsne_cuml':
        reducer = reduce_to_2d_tsne_cuml
        set_timeout = True
    elif method == 'umap':
        reducer = reduce_to_2d_umap
        set_timeout = False
    elif method == 'umap_cuml':
        reducer = reduce_to_2d_umap_cuml
        set_timeout = True
    elif method == 'umap_tsne':
        reducer = reduce_to_2d_umap_tsne
        set_timeout = False
    elif method == 'umap_tsne_cuml':
        reducer = reduce_to_2d_umap_tsne_cuml
        set_timeout = True
    elif method == 'umap_tsne_cuml_both':
        reducer = reduce_to_2d_umap_tsne_cuml_both
        set_timeout = True
    else:
        raise ValueError(f'Unknown reduce_to_2d() method: {method}')
    
    if reducer is not None:
        if set_timeout:
            embeddings_2d = with_timeout(reducer, timeout=600, embeddings=embeddings, debug=debug)
        else:
            embeddings_2d = reducer(embeddings, debug)
    else:
        embeddings_2d = None
    return embeddings_2d
