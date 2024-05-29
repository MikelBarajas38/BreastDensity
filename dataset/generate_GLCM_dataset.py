import csv
import pandas as pd
import cv2
import numpy as np
import argparse

from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

import preprocessing as pp

n_samples = 50

def get_n_samples_per_density(density_df, n_samples=n_samples):

    densities = density_df['breast density'].unique()

    samples = []

    for density in densities:
        temp = density_df[density_df['breast density'] == density]

        if len(temp) < n_samples:
            n_samples = len(temp)

        temp = temp.sample(n_samples).reset_index(drop=True)
        samples.append(temp)

    return pd.concat(samples, ignore_index=True)

def generate_GLCM_dataset_separate_past(density_df):
    #density_df = get_n_samples_per_density(density_df)
    print(density_df)

    image_path = density_df['image_path']
    breast_density = density_df['breast density']
    type = density_df['image view']

    GLCM_df = pd.DataFrame()

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    data_per_image_CC = []
    data_per_image_MLO = []

    for i in range(len(density_df)):
        temp = {}

        img = cv2.imread(image_path[i], cv2.IMREAD_GRAYSCALE)
        ROI = pp.get_ROI(img)

        GLCM = graycomatrix(ROI, [1], angles, symmetric=True, normed=True)

        GLCM_asm = graycoprops(GLCM, 'ASM')[0]
        GLCM_corr = graycoprops(GLCM, 'correlation')[0]
        GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
        GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
        GLCM_contr = graycoprops(GLCM, 'contrast')[0]

        for angle in range(len(angles)):
            temp[f'ASM_{angle}'] = GLCM_asm[angle]
            temp[f'correlation_{angle}'] = GLCM_corr[angle]
            temp[f'dissimilarity_{angle}'] = GLCM_diss[angle]
            temp[f'homogeneity_{angle}'] = GLCM_hom[angle]
            temp[f'contrast_{angle}'] = GLCM_contr[angle]

        entropy = shannon_entropy(ROI)
        temp[f'entropy'] = entropy

        density = breast_density[i]
        temp[f'density'] = density

        if type[i] == 'CC':
            data_per_image_CC.append(pd.DataFrame(temp, index=[0]))
        else:
            data_per_image_MLO.append(pd.DataFrame(temp, index=[0]))

    GLCM_df_CC = pd.concat(data_per_image_CC, ignore_index=True)
    GLCM_df_MLO = pd.concat(data_per_image_MLO, ignore_index=True)

    GLCM_df_CC.to_csv(f'{args.out}/GLCM_dataset_CC.csv', index=False)
    GLCM_df_MLO.to_csv(f'{args.out}/GLCM_dataset_MLO.csv', index=False)

    print(f'GLCM info saved to {args.out}/GLCM_dataset_CC.csv & {args.out}/GLCM_dataset_MLO.csv')


def generate_GLCM_dataset(density_df):
    
    # density_df = get_n_samples_per_density(density_df)
    # print(density_df)

    id = density_df['id']
    image_path = density_df['image_path']
    type = density_df['image view']
    breast_density = density_df['breast density']

    GLCM_df = pd.DataFrame() # use the constructor below

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    data_per_image = []

    for i in range(len(density_df)):

        print(f'Processing image {i+1}/{len(density_df)}')

        temp = {}

        img = cv2.imread(image_path[i], cv2.IMREAD_GRAYSCALE)
        ROI = pp.get_ROI(img)

        GLCM = graycomatrix(ROI, [1], angles, symmetric=True, normed=True)

        GLCM_asm = graycoprops(GLCM, 'ASM')[0]
        GLCM_corr = graycoprops(GLCM, 'correlation')[0]
        GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
        GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
        GLCM_contr = graycoprops(GLCM, 'contrast')[0]

        temp['id'] = id[i]
        temp['image_path'] = image_path[i]

        for angle in range(len(angles)):
            temp[f'ASM_{angle}'] = GLCM_asm[angle]
            temp[f'correlation_{angle}'] = GLCM_corr[angle]
            temp[f'dissimilarity_{angle}'] = GLCM_diss[angle]
            temp[f'homogeneity_{angle}'] = GLCM_hom[angle]
            temp[f'contrast_{angle}'] = GLCM_contr[angle]

        entropy = shannon_entropy(ROI)
        temp[f'entropy'] = entropy

        density = breast_density[i]
        temp['view'] = type[i]
        temp['density'] = density

        data_per_image.append(pd.DataFrame(temp))

    GLCM_df = pd.concat(data_per_image, ignore_index=True) # TODO: just use the constructor lol
    GLCM_df.to_csv(f'{args.out}/GLCM_dataset.csv', index=False)
    print(f'GLCM info saved to {args.out}/GLCM_dataset.csv')

def main():
    density_df = pd.read_csv('density_info.csv', sep=',')
    generate_GLCM_dataset(density_df)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default = 'GLCM', help='your output path')
    args = parser.parse_args()
    main()