# -*- coding: utf-8 -*-

from contextlib import contextmanager
import os
import shelve
import time
from vocab_coverage import constants
from vocab_coverage.utils import logger


class Cache:
    _instance = None # 单例模式

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(Cache, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), constants.FOLDER_CACHE)
        self.file = os.path.join(self.folder, constants.FILE_CACHE)
        self.lock = os.path.join(self.folder, constants.FILE_CACHE_LOCK)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder, exist_ok=True)
        if not os.path.exists(self.file):
            with self.open() as db:
                db['version'] = 1

    def set(self, key, value):
        with self.open() as db:
            db[key] = value

    def get(self, key):
        with self.open() as db:
            return db.get(key)
    
    def has(self, key):
        with self.open() as db:
            return key in db
    
    @contextmanager
    def open(self):
        # check lock
        start = time.time()
        waiting = False
        if os.path.exists(self.lock):
            logger.debug("waiting for lock %s...", self.lock)
            waiting = True
        while os.path.exists(self.lock):
            if time.time() - start > 600:
                raise TimeoutError(f"Waiting for lock {self.lock} timeout after 600 seconds")
            time.sleep(10)
        # lock
        with open(self.lock, 'w', encoding='utf-8') as f:
            f.write("locked")
            if waiting:
                logger.debug("aquired lock %s", self.lock)
        # open
        db = shelve.open(self.file)
        yield db
    
        # close
        db.close()

        # unlock
        os.remove(self.lock)
        if waiting:
            logger.debug("released lock %s", self.lock)

    @staticmethod
    def key(model_name, granularity, position, category):
        return f"{model_name}:{granularity}:{position}:{category}"

cache = Cache()
