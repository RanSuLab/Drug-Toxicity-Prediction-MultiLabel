import csv as csv
import numpy as np
import pandas as pd


# modified @ 25th, Jul., 2020


def import_data(file_name):
    # open file given the file name and
    # return a 2-dimension list
    file_data = []
    row_count = 0
    print('Loading file: ' + file_name)
    try:
        csv_file = open(file_name, 'rt')
        # csv_file = file(file_name, 'rb')  # Opens a file for reading only in binary format
        lines = csv.reader(csv_file)  # Each row read from the csv file is returned as a list of strings.
        for line in lines:
            file_data.append(line)
            row_count += 1
        csv_file.close()
    except Exception as e:
        print(e)
        if not file_data == []:
            print("Current Data: row " + str(row_count))
            print(file_data[-1])
    else:
        col_count = file_data[0].__len__()
        print(file_name + ' Loaded Successfully')
        print('Data Size: ' + str(row_count) + ' rows, ' + str(col_count) + ' cols')
        return file_data


def import_matrix(file_name, has_title_line=False):
    # import the shuffled data as matrix
    # attention, feature_set and label_set shall imported separately
    data = import_data(file_name)  # data is a list of strings
    if has_title_line:
        field_names = np.array(data[0])  # feature names
        data_matrix = np.mat(data[1:], dtype=np.float64)
        return field_names, data_matrix
    else:
        data_matrix = np.mat(data[0:], dtype=np.float64)  # numpy.mat:interpret the input as a matrix.
        return data_matrix


def generate_ordered_data_MLRF(sorted_feature_names, features_needs_to_be_sorted, organ_choice):
    # MLRF, MLReliefF
    # columns : sequence, optional. Columns to write.
    # Write specific columns in a csv file
    # feature_names = open('output/sorted_feature_names.csv', 'w')
    # feature_names_writer = csv.writer(feature_names)
    # feature_names_writer.writerow(sorted_feature_names)
    features_needs_to_be_sorted.to_csv('./output/' + organ_choice + '-MLReliefF-sorted-feature.csv',
                                       index=False, columns=sorted_feature_names)
    return


def generate_ordered_data_MLFS(sorted_feature_names, features_needs_to_be_sorted, organ_choice):
    # MLFS, Multi Label F Statistic
    features_needs_to_be_sorted.to_csv('./output/' + organ_choice + '-MLFStatistic-sorted-feature.csv',
                                       index=False, columns=sorted_feature_names)
    return


def generate_data_file(path_dataset, path_data, label_num, organ_choice):
    """
    data shall be two parts, feature set named data_name_feature and label set named data_name_label
    data shall locate in './datasets/'. shuffled data will be saved in './dataset/organ/'
    the name of features or labels will be save in data_name_feature_names and data_name_label_names separately
    """
    # read file into dataframe
    dataset = pd.read_csv(path_dataset)

    # save the names in feature_names.csv and label_names.csv
    label_names = dataset.columns.tolist()[:label_num]
    feature_names = dataset.columns.tolist()[label_num:]
    feature_names_df = pd.DataFrame(feature_names)
    feature_names_df.T.to_csv(path_data + organ_choice + '_feature_names.csv', index=False, header=False)
    label_names_df = pd.DataFrame(label_names)
    label_names_df.T.to_csv(path_data + organ_choice + '_label_names.csv', index=False, header=False)

    # save feature_data_and_names.csv
    feature_all = dataset.iloc[:, label_num:]
    feature_all.to_csv(path_data + organ_choice + '_feature_data_and_names.csv', index=False)

    # save feature data and label data separately
    feature = dataset.iloc[:, label_num:]
    feature.to_csv(path_data + organ_choice + '_feature_data.csv', index=False, header=None)
    label = dataset.iloc[:, :label_num]
    label.to_csv(path_data + organ_choice + '_label_data.csv', index=False, header=None)
