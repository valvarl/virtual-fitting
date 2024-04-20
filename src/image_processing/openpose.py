import os
import subprocess

from config import Config as cfg


class OpenposeProcessor:
    def __init__(self):
        pass
    
    def __call__(self, imgs: list[str], dst):
        os.chdir(cfg.openpose)


        
        