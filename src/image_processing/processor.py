import glob
import json
import os
import os.path as osp
import shutil
import subprocess
import tempfile
import typing as tp

import numpy as np
from PIL import Image

from .cloth_mask import ClothMaskProcessor
from .human_agnostic import get_img_agnostic
from ..config import Config as cfg


class ImageProcessor:
    def __init__(self):
        self.image_cloth_pairs = []
        self.tempdir = tempfile.TemporaryDirectory()
        self.make_structure()
        
        self.cloth_mask_processor = ClothMaskProcessor()
        
    def __del__(self):
        del self.cloth_mask_processor
        self.tempdir.cleanup()
        
    def make_structure(self):
        os.makedirs(osp.join(self.tempdir.name, 'queue'))
        os.makedirs(osp.join(self.tempdir.name, 'queue/image'))
        os.makedirs(osp.join(self.tempdir.name, 'queue/cloth'))
        os.makedirs(osp.join(self.tempdir.name, 'process'))
        os.makedirs(osp.join(self.tempdir.name, 'process/image'))
        os.makedirs(osp.join(self.tempdir.name, 'process', cfg.openpose_json))
        os.makedirs(osp.join(self.tempdir.name, 'process', cfg.openpose_img))
        os.makedirs(osp.join(self.tempdir.name, 'process', cfg.human_parse_output))
        os.makedirs(osp.join(self.tempdir.name, 'process', cfg.densepose_output))
        os.makedirs(osp.join(self.tempdir.name, 'process/agnostic-v3.2'))
        os.makedirs(osp.join(self.tempdir.name, 'process/agnostic-mask'))
        os.makedirs(osp.join(self.tempdir.name, 'process/cloth'))
        os.makedirs(osp.join(self.tempdir.name, 'process/cloth-mask'))
        
    def clear_all(self):
        os.rmdir(osp.join(self.tempdir.name, 'queue'))
        os.rmdir(osp.join(self.tempdir.name, 'process'))
        self.make_structure()
    
    def add_pair_image_cloth(self, image: str, cloth: str):
        image_dst = osp.join(self.tempdir.name, 'queue/image')
        cloth_dst = osp.join(self.tempdir.name, 'queue/cloth')
        shutil.copy(image, image_dst)
        shutil.copy(cloth, cloth_dst)
        self.image_cloth_pairs.append((osp.join(image_dst, osp.basename(image)), 
                                       osp.join(cloth_dst, osp.basename(cloth))))
        
    def process(self):
        for image, cloth in self.image_cloth_pairs:
            self.resize_image(image, osp.join(self.tempdir.name, 'process/image'))
            self.resize_image(cloth, osp.join(self.tempdir.name, 'process/cloth'))
        
        images = [img for img, _ in self.image_cloth_pairs]
        cloth = [cloth for _, cloth in self.image_cloth_pairs]
        
        self._openpose(images, 
                       osp.join(self.tempdir.name, 'process', cfg.openpose_json), 
                       osp.join(self.tempdir.name, 'process', cfg.openpose_img))
        
        self._human_parse(images, osp.join(self.tempdir.name, 'process', cfg.human_parse_output))
        
        self._denspose(images, osp.join(self.tempdir.name, 'process', cfg.densepose_output))
        
        self._cloth_mask(cloth, osp.join(self.tempdir.name, 'process/cloth-mask'))

        self._human_agnostic(images, 
                             osp.join(self.tempdir.name, 'process/agnostic-v3.2'),
                             osp.join(self.tempdir.name, 'process/agnostic-mask'))
        
        return self.tempdir.name
        
    def _openpose(self, images: tp.List[str], json_dst: str, render_dst: str):
        with tempfile.TemporaryDirectory(dir=self.tempdir.name) as temp_dir:
            for image in images:
                shutil.copy(image, temp_dir)
                
            command = [
                'build/examples/openpose/openpose.bin', 
                '--image_dir', temp_dir, 
                '--write_json', osp.join(self.tempdir.name, 'process', cfg.openpose_json), 
                '--write_images', osp.join(self.tempdir.name, 'process', cfg.openpose_img), 
                '--disable_blending', '--display', '0', '--num_gpu', '1', '--num_gpu_start', '0'
            ]

            if cfg.openpose_hand:
                command.append('--hand')

            subprocess.run(command, cwd=cfg.openpose)
            
    def _human_parse(self, images: tp.List[str], dst):
        shutil.rmtree(osp.join(cfg.human_parse, 'output'), ignore_errors=True)
        files = glob.glob(osp.join(cfg.human_parse, 'datasets/images/*'))
        for f in files:
            os.remove(f)
         
        for image in images:
            shutil.copy(image, osp.join(cfg.human_parse, 'datasets/images'))
                
        command = [
            'python', 
            'inference_pgn.py'
        ]

        subprocess.run(command, cwd=cfg.human_parse)
        
        output_path = osp.join(cfg.human_parse, 'output/cihp_parsing_maps')
        for image in os.listdir(output_path):
            if not image.endswith('vis.png'):
                shutil.copy(osp.join(output_path, image), dst)
    
    def _denspose(self, images: tp.List[str], dst):
        with tempfile.TemporaryDirectory(dir=self.tempdir.name) as temp_dir:
            input_path = osp.join(temp_dir, 'input')
            output_path = osp.join(temp_dir, 'output')
            os.makedirs(input_path)
            os.makedirs(output_path)
        
            for image in images:
                shutil.copy(image, input_path)
            
            command = [
                'python', osp.join(cfg.densepose, 'apply_net.py'), 'show',
                osp.join(cfg.densepose, 'configs/densepose_rcnn_R_50_FPN_s1x.yaml'),
                cfg.denspose_checkpoint, input_path, 'dp_segm', '-v', '--output-dir', output_path
            ]

            subprocess.run(command)
            
            for image in os.listdir(output_path):
                shutil.copy(osp.join(output_path, image), dst)

    def _cloth_mask(self, images: tp.List[str], dst):
        self.cloth_mask_processor(images, dst)
        
    def _human_agnostic(self, images: tp.List[str], agnostic_dst, agnostic_mask_dst):
        data_path = osp.join(self.tempdir.name, 'process')

        for im_name in [osp.basename(i) for i in images]:

            # load pose image
            pose_name = im_name.replace('.jpg', '_keypoints.json')

            try:
                with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                    pose_label = json.load(f)
                    pose_data = pose_label['people'][0]['pose_keypoints_2d']
                    pose_data = np.array(pose_data)
                    pose_data = pose_data.reshape((-1, 3))[:, :2]
            except IndexError:
                print(pose_name)
                continue

            # load parsing image
            im = Image.open(osp.join(data_path, 'image', im_name))
            label_name = im_name.replace('.jpg', '.png')
            im_label = Image.open(osp.join(data_path, 'image-parse-v3', label_name))

            agnostic, agnostic_mask = get_img_agnostic(im, im_label, pose_data)
            agnostic.save(osp.join(agnostic_dst, im_name))
            agnostic_mask.save(osp.join(agnostic_mask_dst, im_name.replace('.jpg', '_mask.png')))


    @staticmethod
    def resize_image(path: str, dst):
        # Загрузить изображение
        img = Image.open(path)

        # Определить размеры изображения
        width, height = img.size

        # Обработка горизонтального изображения
        if width > height:
            # Вырезаем середину
            left = (width - height * 3 / 4) / 2
            right = left + height * 3 / 4
            img = img.crop((left, 0, right, height))
            # Масштабируем до нужных размеров
            img = img.resize((768, 1024))
        else:
            # Масштабирование, если высота больше 1024
            if height > 1024:
                new_height = 1024
                new_width = int((new_height / height) * width)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            width, height = img.size  # Обновляем размеры после масштабирования

            # Если ширина меньше 768, добавляем отзеркаленные края
            if width < 768:
                delta = (768 - width) // 2
                padding = (delta, 0, delta, 0)
                img = ImageOps.expand(img, padding, fill='black')
                img = ImageOps.mirror(img)
            elif width > 768:
                # Обрезаем до 768 по ширине, выравнивая по центру
                left = (width - 768) / 2
                right = left + 768
                img = img.crop((left, 0, right, height))
        
        img.save(osp.join(dst, osp.basename(path)))
