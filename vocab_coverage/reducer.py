# -*- coding: utf-8 -*-

import importlib
import multiprocessing
import warnings
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import random

from vocab_coverage.utils import logger
from vocab_coverage import constants

def reduce_to_2d_tsne(embeddings, debug=False):
    TSNE = importlib.import_module('sklearn.manifold').TSNE
    if debug:
        logger.debug("> reduce_to_2d_tsne(%s): %s", embeddings.shape, TSNE)
    tsne_model = TSNE(n_components=2,
        perplexity=30,
        early_exaggeration=12,
        learning_rate='auto',
        metric='cosine',
        init='pca',
        verbose=2 if debug else 0,
        n_iter=1000,
        random_state=2218,
        method='barnes_hut',
        n_jobs=-1)
    embeddings_2d = tsne_model.fit_transform(embeddings)
    # for i in range(5):
    #     logger.debug(f"reducer[tsne]: {embeddings[i][:5]}...{embeddings[i][-3:]} => {embeddings_2d[i]}")
    return embeddings_2d

def reduce_to_2d_tsne_cuml(embeddings, debug=False):
    TSNE = importlib.import_module('cuml.manifold').TSNE
    if debug:
        logger.debug("> reduce_to_2d_tsne_cuml(%s): %s", embeddings.shape, TSNE)
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
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    UMAP = importlib.import_module('umap').UMAP
    if debug:
        logger.debug("> reduce_to_2d_umap(%s): %s", embeddings.shape, UMAP)
    umap_model = UMAP(n_components=2,
        n_neighbors=30,
        min_dist=0.6,
        spread=1,
        verbose=debug,
        random_state=42,
        metric='cosine')
    embeddings_2d = umap_model.fit_transform(embeddings)
    for i in range(5):
        logger.debug(f"reducer[umap-2d]: {embeddings[i][:5]}... => {embeddings_2d[i]}")
    return embeddings_2d

def reduce_to_2d_umap_cuml(embeddings, debug=False):
    UMAP = importlib.import_module('cuml.manifold').UMAP
    if debug:
        logger.debug("> reduce_to_2d_umap_cuml(%s): %s", embeddings.shape, UMAP)
    umap_model = UMAP(n_components=2,
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
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    UMAP = importlib.import_module('umap').UMAP
    if debug:
        logger.debug("> reduce_to_2d_umap_tsne(%s): %s", embeddings.shape, UMAP)
    umap_model = UMAP(n_components=30,
        n_neighbors=30,
        min_dist=0.6,
        spread=1,
        verbose=debug,
        random_state=42,
        metric='cosine')
    embeddings_30 = umap_model.fit_transform(embeddings)
    for i in range(5):
        logger.debug(f"reducer[umap-30]: {embeddings[i][:5]}... => {embeddings_30[i][:5]}...")
    # TSNE
    embeddings_2d = reduce_to_2d_tsne(embeddings_30, debug)
    return embeddings_2d


def reduce_to_2d_umap_tsne_cuml(embeddings, debug=False):
    # Use UMAP reduce to 30 dimensions, then TSNE to 2
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    UMAP = importlib.import_module('umap').UMAP
    if debug:
        logger.debug("> reduce_to_2d_umap_tsne_cuml(%s): %s", embeddings.shape, UMAP)
    umap_model = UMAP(n_components=30,
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
    warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
    UMAP = importlib.import_module('cuml.manifold').UMAP
    if debug:
        logger.debug("> reduce_to_2d_umap_tsne_cuml_both(%s): %s", embeddings.shape, UMAP)
    umap_model = UMAP(n_components=30,
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
    logger.debug("> call %s with timeout: %ss", func, timeout)

    current_process = multiprocessing.current_process()
    timeout_event = multiprocessing.Event()

    def monitor(timeout=300):
        if not timeout_event.wait(timeout):
            logger.debug("with_timeout():monitor(): timeout")
            if current_process.is_alive():
                logger.error("> %s timeout: %ss", func, timeout)
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

def reduce_to_2d(embeddings, method:str=constants.REDUCER_TSNE, shuffle:bool=True, debug:bool=False):
    if debug:
        logger.debug("> reducing the dimension of %s to 2D by [%s]...", embeddings.shape, method)
    reducer = None
    set_timeout = False
    reducers = {
        constants.REDUCER_TSNE: { 'reducer': reduce_to_2d_tsne, 'set_timeout': False },
        constants.REDUCER_TSNE_CUML: { 'reducer': reduce_to_2d_tsne_cuml, 'set_timeout': True },
        constants.REDUCER_UMAP: { 'reducer': reduce_to_2d_umap, 'set_timeout': False },
        constants.REDUCER_UMAP_CUML: { 'reducer': reduce_to_2d_umap_cuml, 'set_timeout': True },
        constants.REDUCER_UMAP_TSNE: { 'reducer': reduce_to_2d_umap_tsne, 'set_timeout': False },
        constants.REDUCER_UMAP_TSNE_CUML: { 'reducer': reduce_to_2d_umap_tsne_cuml, 'set_timeout': True },
        constants.REDUCER_UMAP_TSNE_CUML_BOTH: { 'reducer': reduce_to_2d_umap_tsne_cuml_both, 'set_timeout': True },
    }
    reducer = reducers[method]['reducer']
    set_timeout = reducers[method]['set_timeout']
    # shuffle
    if shuffle:
        original_embeddings = embeddings
        base_embeddings = list(enumerate(embeddings))
        random.shuffle(base_embeddings)
        embeddings = [x[1] for x in base_embeddings]
        embeddings = np.array(embeddings)
        if debug:
            for i in range(5):
                for j, em in enumerate(base_embeddings):
                    if em[0] == i:
                        logger.debug("reducer: (shuffle) [%s->%s]: original: %s... => %s...", i, j, original_embeddings[i][:5], embeddings[j][:5])
    # reduce
    if reducer is not None:
        if debug:
            logger.debug("> call reducer: %s, set_timeout: %s", reducer, set_timeout)
        if set_timeout:
            embeddings_2d = with_timeout(reducer, timeout=600, embeddings=embeddings, debug=debug)
        else:
            embeddings_2d = reducer(embeddings, debug)
    else:
        embeddings_2d = None
    # normalize
    if embeddings_2d is not None:
        embeddings_2d = MinMaxScaler().fit_transform(embeddings_2d)
    # restore the original order
    if shuffle:
        embeddings_2d = [(x[0], embeddings, embeddings_2d[i]) for i, x in enumerate(base_embeddings)]
        embeddings_2d.sort(key=lambda x: x[0])
        if debug:
            for i in range(5):
                logger.debug("reducer: (2D) [%s]: original: %s... => %s", i, original_embeddings[i][:5], embeddings_2d[i][2])
        embeddings_2d = [x[2] for x in embeddings_2d]
        embeddings_2d = np.array(embeddings_2d)
    return embeddings_2d
