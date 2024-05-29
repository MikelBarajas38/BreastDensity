import csv
import pandas as pd
import cv2
import numpy as np
import os
import argparse
import multiprocessing

import preprocessing as pp

def process_image(image_path, id):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if args.ROI:
        ROI = pp.get_ROI(img)
        img = ROI
    else:
        if args.CLAHE:
            img = pp.apply_clahe(img)
        
        img = pp.clean_img(img)

    cv2.imwrite(f'{args.out}/{id}.jpg', img)

def main():

    try: 
        os.mkdir(args.out) 
    except OSError as error: 
        print(error)

    density_df = pd.read_csv('density_info.csv', sep=',')

    ids = density_df['id']
    image_paths = density_df['image_path']

    args_list = list(zip(image_paths, ids))

    if args.multiprocessing:
        with multiprocessing.Pool() as pool:
            pool.starmap(process_image, args_list)
    else:
        for i in range(len(image_paths)):
            print(f'Processing image {i+1}/{len(image_paths)}')
            process_image(image_paths[i], ids[i])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default = 'img', help='your output path')
    parser.add_argument('--ROI', action='store_true', help='store ROI')
    parser.add_argument('--CLAHE', action='store_true', help='apply CLAHE')
    parser.add_argument('--multiprocessing', action='store_true', help='use multiprocessing')
    args = parser.parse_args()
    main()