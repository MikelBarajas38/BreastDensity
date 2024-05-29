import csv
import pandas as pd

other_files = ['CBIS-DDSM/csv/mass_case_description_test_set.csv', 
               'CBIS-DDSM/csv/mass_case_description_train_set.csv',
                'CBIS-DDSM/csv/calc_case_description_test_set.csv',
                'CBIS-DDSM/csv/calc_case_description_train_set.csv']

dicom_info_columns = ['file_path', 'image_path', 'SeriesDescription']
other_columns = ['image file path', 'patient_id', 'image view', 'breast density']

def main():

    dicom_info = pd.read_csv('CBIS-DDSM/csv/dicom_info.csv', sep=',', usecols=dicom_info_columns)
    dicom_info = dicom_info[dicom_info.SeriesDescription == 'full mammogram images']
    dicom_info['file_path'] = dicom_info['file_path'].apply(lambda x: x.split('/')[2])

    file_df = []

    for file in other_files:
        
        other_df = pd.read_csv(file, sep=',', usecols=other_columns)
        other_df['image file path'] = other_df['image file path'].apply(lambda x: x.split('/')[2])
        other_df = other_df.reindex(columns=other_columns)

        merged_df = pd.merge(dicom_info, other_df, left_on='file_path', right_on='image file path', how='left')
        merged_df = merged_df.dropna()

        merged_df = merged_df.drop('image file path', axis=1)
        merged_df['breast density'] = merged_df['breast density'].apply(lambda x: int(x))

        file_df.append(merged_df)

    final_df = pd.concat(file_df, ignore_index=True)
    final_df.to_csv('density_info.csv', index=True, index_label='id')

if __name__ == '__main__':
    main()