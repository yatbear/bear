#!usr/bin/env python
# -*- coding: utf-8 -*-

# Usage: python [files]
#
# Image Warper:
#       Extract SIFT features
#       Match features based on Lowe's ratio test
#       Warp image using Perspective Transformation
#
# Author: yatbear <sapphirejyt@gmail.com>
#         2016-03-14

import cv2
import numpy as np
from collections import OrderedDict

class Warper(object):
    
    def __init__(self, rtthres=0.7, min_match_count=4):
        '''
        Args:
            rtthres: threshold for Lowe's ratio test
            min_match_count: minimun number of matches
        '''
        self.RATIO_TEST_THRES = rtthres
        self.MIN_MATCH_COUNT = min_match_count
        
    def get_gray_img(self, img_path):
        ''' Read a gray image from the given path. '''
        gray = cv2.imread(img_path, 0)
        return gray
    
    def resize_img(self, src, dst):
        ''' 
        Resize the source image to the same size as the destination image.
        Args:
            src: 2-D matrix
            dst: 2-D matrix
        Returns:
            resized source image matrix
        '''
        h, w = dst.shape
        img = cv2.resize(src, (w, h))
        return img
    
    def extract_features(self, img):
        ''' 
        Extract SIFT featrues from input image.
        Returns:
            kp: a list of KeyPoint objects
            desc: a list of SIFT descriptors of the key points  
        '''
        sift = cv2.SIFT()
        kp, desc = sift.detectAndCompute(img, None)
        return kp, desc
        
    def get_matches(self, desc1, desc2):
        ''' 
        Match features with FLANN. 
        Select good matches based on Lowe's ratio test.
        Args: 
            desc1: a list of SIFT descriptors of image 1
            desc2: a list of SIFT descriptors of image 2
        Returns:
            a list of DMatch objects
        '''
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)

        # Filter matches using ratio test
        def select_good_matches(matches, thres):
            good = list()
            for (m1, m2) in matches:
                if m1.distance < thres * m2.distance:
                    good.append(m1)
            return good
        
        good = select_good_matches(matches, self.RATIO_TEST_THRES)   
        return good if len(good) >= self.MIN_MATCH_COUNT else matches
        
    def display_matches(self, img1, img2, kp1, kp2, matches):
        ''' 
        Display feature matches.
        Args: 
            img1: gray image 1
            img2: gray image 2
            kp1: key points of image 1
            kp2: key points of image 2
            matches: a list of DMatch objects
        '''
        # Concatenate two gray images side by side
        img3 = np.concatenate((img1, img2), axis=1)
        n = img1.shape[1] # width of the left image
        
        # Draw matching pairs
        for m in matches:
            (x1, y1) = kp1[m.queryIdx].pt # left
            (x2, y2) = kp2[m.trainIdx].pt # right
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x2 += n # offset by the width of the image on the left
            # Mark SIFT features
            cv2.circle(img3, (x1, y1), 6, (0, 0, 255), thickness=2)
            cv2.circle(img3, (x2, y2), 6, (0, 0, 255), thickness=2)
            # Draw a line between two matched points
            cv2.line(img3, (x1, y1), (x2, y2), (0, 255, 0))  
        
        # Resize for display
        img3 = cv2.resize(img3, (0, 0), fx=0.8, fy=0.8) 
        self.display_img(img3, 'SIFT feature matching')
        
    def display_img(self, img, title=''):
        ''' Display the given image. '''
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def warp(self, img1, img2, kp1, kp2, matches):
        ''' 
        Warp image 1 towards image 2 using perspective transformation.
        Return:
            Warped image 
        '''
        # Get points of interest for matching
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # Find a perspective transformation
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
        h, w = img1.shape
        # Warp the image 
        warped = cv2.warpPerspective(img1, H, (w, h))
        return warped
        
    def save_img(self, img, filename='test.png'):
        ''' Write image to file. '''
        cv2.imwrite(filename, img)
    
    def driver(self, path1, path2, show_matches=False):
        ''' 
        Driver function.
        Args:
            path1: path to image file 1
            path2: path to image file 2
            show_matches: if display SIFT matching results
        '''
        # Read images
        img1 = self.get_gray_img(path1)
        img2 = self.get_gray_img(path2)
        img1 = self.resize_img(img1, img2)
        
        # Pad zeros to image borders
        pad_width = 65
        npad = ((pad_width, pad_width), (pad_width, pad_width))
        img1 = np.pad(img1, pad_width=npad, mode='constant', constant_values=0)
        img2 = np.pad(img2, pad_width=npad, mode='constant', constant_values=0)
        
        # Calculate SIFT features
        kp1, desc1 = self.extract_features(img1)
        kp2, desc2 = self.extract_features(img2)
        
        # Match SIFT features
        matches = self.get_matches(desc1, desc2)
        if show_matches:
            self.display_matches(img1, img2, kp1, kp2, matches)
        
        # Projective Transformation  
        warped = self.warp(img1, img2, kp1, kp2, matches)
        warped = self.resize_img(warped, img2)
        img3 = np.concatenate((warped, img2), axis=1)
        self.display_img(img3, 'Warped')

        # Stitch the warped image and image 2 together 
        img = cv2.addWeighted(warped, 0.5, img2, 0.5, 0)
        self.display_img(img, 'Stitched')
        self.save_img(img, 'result.png')
    
def main():
    path1, path2 = 'graf1.png', 'graf2.png'
    warper = Warper(rtthres=0.7, min_match_count=4)
    warper.driver(path1, path2, show_matches=True)
    
if __name__ == '__main__':
    main()    