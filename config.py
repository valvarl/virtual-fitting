from dataclasses import dataclass


@dataclass
class Config:
    densepose = './detectron2/projects/DensePose'
    densepose_output = 'image-densepose'
    denspose_checkpoint = './checkpoints/model_final_162be9.pkl'