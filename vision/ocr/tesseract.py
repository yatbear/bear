# -*- coding: utf-8 -*-
# Optical Character Recognition

import requests
import builtins
from PIL import Image, ImageFilter
from StringIO import StringIO
from pytesseract import *

class Tesseract(object):
    
    def __init__(self):
        self.original_open = open
    
    def bin_open(self, filename, mode='rb'):
        return self.original_open(filename, mode)
    
    def retrieve_image(self, url):
        return Image.open(StringIO(requests.get(url).content))
    
    def open_image(self, path):
        return Image.open(path)
    
    def process_image(self, img):
        img.filter(ImageFilter.SHARPEN)
        try:
            builtins.open = self.bin_open
            bts = image_to_string(img)
        finally:
            builtins.open = self.original_open    
        
        return bts.decode('utf-8')

def main():
    tesser = Tesseract()
    url = 'https://www.realpython.com/images/blog_images/ocr/sample4.jpg'
    img = tesser.retrieve_image(url)
    text = tesser.process_image(img)
    print text

if __name__ == '__main__':
    main()