import argparse
from distutils.dir_util import copy_tree
from virtual_fitting.image_processing.processor import ImageProcessor


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process image with cloth overlay')
    parser.add_argument('image', type=str, help='Path to the input image file')
    parser.add_argument('cloth', type=str, help='Path to the cloth overlay image file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    image_path = args.image
    cloth_path = args.cloth
    
    ip = ImageProcessor()
    ip.add_pair_image_cloth(image_path, cloth_path)
    output = ip.process()
    copy_tree(output, './output')
    