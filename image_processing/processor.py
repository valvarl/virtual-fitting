import os
import os.path as osp
import shutil
import subprocess
import tempfile

from PIL import Image

from config import Config as cfg


class ImageProcessor:
    def __init__(self):
        self.image_cloth_pairs = []
        self.tempdir = tempfile.TemporaryDirectory()
        self.make_structure()
        
    def __del__(self):
        os.rmdir(self.tempdir)
    
    def add_pair_image_cloth(self, image: str, cloth: str):
        image_dst = osp.join(self.tempdir, 'queue/image')
        cloth_dst = osp.join(self.tempdir, 'queue/cloth')
        shutil.copy(image, image_dst)
        shutil.copy(cloth, cloth_dst)
        self.image_cloth_pairs.append((osp.join(image_dst, os.basename(image)), 
                                       osp.join(cloth_dst, os.basename(cloth))))
        
    def process(self):
        for image, cloth in self.image_cloth_pairs:
            
        
    def clear_all(self):
        os.rmdir(osp.join(self.tempdir, 'queue'))
        os.rmdir(osp.join(self.tempdir, 'process'))
        self.make_structure()
    
    def make_structure(self):
        os.makedirs(osp.join(self.tempdir, 'queue'))
        os.makedirs(osp.join(self.tempdir, 'process'))
        os.makedirs(osp.join(self.tempdir, 'process/image'))
        os.makedirs(osp.join(self.tempdir, 'process/image-densepose'))
        os.makedirs(osp.join(self.tempdir, 'process/agnostic'))
        os.makedirs(osp.join(self.tempdir, 'process/agnostic-mask'))
        os.makedirs(osp.join(self.tempdir, 'process/cloth'))
        os.makedirs(osp.join(self.tempdir, 'process/cloth_mask'))
    
    def _denspose(self, images: list[str], dst):
        with tempfile.TemporaryDirectory(dir=self.tempdir) as temp_dir:
            input_path = osp.join(temp_dir, 'input')
            output_path = osp.join(temp_dir, 'output')
            os.makedirs(input_path)
            os.makedirs(output_path)
        
            for image in images:
                shutil.copy(image, input_path)
            
            # Строим команду для выполнения
            command = [
                'python', osp.join(cfg.densepose, 'apply_net.py'), 'show',
                osp.join(cfg.densepose, 'configs/densepose_rcnn_R_50_FPN_s1x.yaml'),
                cfg.denspose_checkpoint, input_path, 'dp_segm', '-v', '--output-dir', output_path
            ]

            # Запускаем внешнюю команду
            subprocess.run(command)
            
            for image in os.listdir(output_path):
                shutil.copy(image, dst)
                denspose_images.append(Image.open(osp.join(output_path, i)))
                
            print(dst)
           
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

        return img