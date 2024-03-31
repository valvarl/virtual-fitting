from image_processing.processor import ImageProcessor

if __name__ == '__main__':
    ip = ImageProcessor()
    image = ip.load_image('assets/photo_2023-01-13_21-34-38.jpg')
    ip.denspose([image])