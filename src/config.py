from dataclasses import dataclass


@dataclass
class Config:
    densepose = './detectron2/projects/DensePose'
    densepose_output = 'image-densepose'
    denspose_checkpoint = './checkpoints/model_final_162be9.pkl'
    
    openpose = '/usr/local/openpose'
    openpose_json = 'openpose_json'
    openpose_img = 'openpose_img'
    openpose_hand = False