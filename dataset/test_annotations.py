import csv
import pandas as pd
import cv2

import preprocessing as pp

def test1():

    density_df = pd.read_csv('density_info.csv', sep=',')

    density_df = density_df[density_df['breast density'] == 4]
    density_df = density_df.sample(frac=1).reset_index(drop=True)

    image_path = density_df['image_path']
    breast_density = density_df['breast density']


    for i in range(len(image_path[:5])):
        print(f'image path: {image_path[i]}')

        img = cv2.imread(image_path[i])

        height, width = img.shape[:2]
        new_height = height // 8
        new_width = width // 8
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imshow(f'image (density = {breast_density[i]})', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test2():
    density_df = pd.read_csv('density_info.csv', sep=',')

    # choose 1 image of each density
    density_df = density_df.drop_duplicates(subset='breast density', keep='first').reset_index(drop=True)

    image_path = density_df['image_path']
    breast_density = density_df['breast density']

    for i in range(len(image_path)):

        print(f'image path: {image_path[i]}')
        
        img = cv2.imread(image_path[i], cv2.IMREAD_GRAYSCALE)

        height, width = img.shape[:2]
        new_height = height // 8
        new_width = width // 8
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imshow(f'image (density = {breast_density[i]})', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ROI = pp.get_ROI(img)

        cv2.imshow('ROI', ROI)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_n_samples_from_class_c(df, c, n):
    df = df[df['breast density'] == c]
    df = df.sample(n).reset_index(drop=True)
    return df


def test3(density):
    density_df = pd.read_csv('density_info.csv', sep=',')

    density_df = get_n_samples_from_class_c(density_df, density, 5)

    image_path = density_df['image_path']
    breast_density = density_df['breast density']

    for i in range(len(image_path)):
        print(f'image path: {image_path[i]}')

        img = cv2.imread(image_path[i], cv2.IMREAD_GRAYSCALE)

        height, width = img.shape[:2]
        new_height = height // 8
        new_width = width // 8
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imshow(f'image (density = {breast_density[i]})', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ROI = pp.get_ROI(img)

        cv2.imshow('ROI', ROI)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test4():
    density_df = pd.read_csv('density_info.csv', sep=',')

    random_df = density_df.sample(5).reset_index(drop=True)

    image_path = random_df['image_path']
    ids = random_df['id']

    for i in range(len(image_path)):

        print(f'id: {ids[i]}, image path: {image_path[i]}')
        
        img = cv2.imread(image_path[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(f'img/{ids[i]}.jpg', cv2.IMREAD_GRAYSCALE)

        height, width = img.shape[:2]
        new_height = height // 8
        new_width = width // 8
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imshow(f'1', img)
        cv2.imshow(f'2', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ROI = pp.get_ROI(img)
        ROI2 = cv2.imread(f'roi/{ids[i]}.jpg', cv2.IMREAD_GRAYSCALE)

        cv2.imshow('1', ROI)
        cv2.imshow('2', ROI2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        clean_img = pp.clean_img(img)
        clean_img2 = cv2.imread(f'pp/{ids[i]}.jpg', cv2.IMREAD_GRAYSCALE)

        cv2.imshow('1', clean_img)
        cv2.imshow('2', clean_img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # test1()
    # test2()
    #for i in range(1, 5):
    #    test3(i)
    test4()

if __name__ == '__main__':
    main()