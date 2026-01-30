import os
import random
import numpy as np
import datetime
import tensorflow as tf
from src.config import Config
import subprocess
import sys

class Environment:
    def __init__(self):
        self.seed = Config().seed

    def setup(self):
        # self.install_requirements()
        self.set_seed(self.seed)
        self.print_last_run_notebook()
        self.print_tf_version()

    @staticmethod
    def is_colab():
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False
        return is_colab

    @staticmethod
    def set_seed(_seed):
        os.environ['PYTHONHASHSEED'] = str(_seed)
        random.seed(_seed)
        np.random.seed(_seed)
        tf.random.set_seed(_seed)

    def get_seed(self):
        return str(self.seed)

    @staticmethod
    def print_tf_version():
        print('TensorFlow Version:', tf.__version__)

    @staticmethod
    def print_last_run_notebook():
        print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")

    # @staticmethod
    # def install_requirements(requirements_path="../../requirements.txt"):
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #     subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

