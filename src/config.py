from dataclasses import dataclass


@dataclass
class Config:
    openpose = '/usr/local/openpose'
    openpose_json = 'openpose_json'
    openpose_img = 'openpose_img'
    openpose_hand = False
    
    human_parse = './CIHP_PGN'
    human_parse_output = 'image-parse-v3'
    
    densepose = './detectron2/projects/DensePose'
    densepose_output = 'image-densepose'
    denspose_checkpoint = './checkpoints/model_final_162be9.pkl'
