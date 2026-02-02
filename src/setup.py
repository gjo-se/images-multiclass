import os
import random
import numpy as np
import datetime
import tensorflow as tf
from configs.config import Config


class Environment:
    def __init__(self):
        self.seed = Config().seed

    def setup(self):
        # self.install_requirements()
        self.create_project_structure()
        self.set_seed(self.seed)
        self.print_last_run_notebook()
        self.print_tf_version()

    @staticmethod
    def get_project_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    def create_project_structure(self, base_dir=None):
        if base_dir is None:
            base_dir = self.get_project_root()
        folders = [
            "data/raw",
            "data/processed",
            "data/tfrecords",
            "data/features",
            "data/splits",
            "src",
            "configs",
            "experiments",
            "models/checkpoints",
            "models/logs"
        ]
        for folder in folders:
            path = os.path.join(base_dir, folder)
            os.makedirs(path, exist_ok=True)
            gitkeep_path = os.path.join(path, ".gitkeep")
            if not os.path.exists(gitkeep_path):
                with open(gitkeep_path, "w"):
                    pass

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
