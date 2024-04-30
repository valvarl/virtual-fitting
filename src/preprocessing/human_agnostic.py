import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 16).astype(np.float32) +
                    (parse_array == 17).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 19).astype(np.float32))

    
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)
    
    agnostic_mask = Image.new('L', agnostic.size)
    agnostic_mask_draw = ImageDraw.Draw(agnostic_mask) 

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    r = int(length_a / 16) + 1
    
    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        agnostic_mask_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 255, 255)
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        agnostic_mask_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 255, width=r*10)
        
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        agnostic_mask_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 255, 255)

    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_mask_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 255, 255)
    
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')
    
    agnostic_mask_draw.line([tuple(pose_data[i]) for i in [2, 9]], 255, width=r*6)    
    agnostic_mask_draw.line([tuple(pose_data[i]) for i in [5, 12]], 255, width=r*6)    
    agnostic_mask_draw.line([tuple(pose_data[i]) for i in [9, 12]], 255, width=r*12)
    agnostic_mask_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 255, 255)

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    
    agnostic_mask_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 255, 255)
    agnostic_mask_draw.bitmap((0, 0), Image.fromarray(np.uint8(parse_lower * 255)), fill=0)
    agnostic_mask_draw.bitmap((0, 0), Image.fromarray(np.uint8(parse_head * 255)), fill=0)

    return agnostic, agnostic_mask
