import numpy as np
import LoadingData as ld
import matplotlib.pyplot as plt
import pandas as pd


# modified @ 6th, Oct., 2020


def multi_label_f_statistic(feature_set, label_set):
    # rank features using Multi Label F Statistic
    # Reference:
    # Ding, C.H., Huang, H., Kong, D., & Zhao, H.. (2012). Multi-label ReliefF and
    # F-statistic feature selections for image annotation. CVPR.

    # feature_set and label_set shall ba matrix, each sample a row
    # return a 2 dimension narray, index is the stored_index for labels in decreasing order,
    # feature_score is the sorted scores

    print('MLFS')
    nrow, ncol = feature_set.shape
    nlabel = label_set.shape[1]

    feature_score = np.zeros(ncol)

    numerator = np.mat(np.zeros((1, ncol)))
    denominator = 0
    for label in range(nlabel):
        for row in range(nrow):
            numerator += (label_set[row, label] * feature_set[row, :])
            denominator += (label_set[row, label])
    sample_mean_all = numerator * 1.0 / denominator

    sample_mean_label = dict()
    sample_list = dict()
    for label in range(nlabel):
        sample_list[label] = []
        for row in range(nrow):
            if label_set[row, label] == 1:
                sample_list[label].append(row)
        if feature_set[sample_list[label], :].__len__() > 0:
            sample_mean_label[label] = feature_set[sample_list[label], :].mean(axis=0)
        else:
            sample_mean_label[label] = np.mat(np.zeros((1, ncol)))

    '''
    scatter_between = np.mat(np.zeros((ncol, ncol)))
    for label in range(nlabel):
        for row in range(nrow):
            scatter_between += (label_set[row, label] * (sample_mean_label[label] - sample_mean_all).transpose() *
                                (sample_mean_label[label] - sample_mean_all))

    scatter_within = np.mat(np.zeros((ncol, ncol)))
    for label in range(nlabel):
        for row in range(nrow):
            scatter_within += (label_set[row, label] * (feature_set[row, :] - sample_mean_label[label]).transpose() *
                               (feature_set[row, :] - sample_mean_label[label]))

    print('Ranking Features:')
    for col in range(ncol):
        if (nlabel - 1) != 0 and scatter_within[col, col] != 0:
            feature_score[col] = (nrow - nlabel) * 1.0 / (nlabel - 1) * scatter_between[col, col] / scatter_within[col, col]
        elif (nlabel - 1) == 0 and scatter_within[col, col] != 0:
            feature_score[col] = (nrow - nlabel) * 1.0 / 1.0 * scatter_between[col, col] / scatter_within[col, col]
        elif (nlabel - 1) != 0 and scatter_within[col, col] == 0:
            feature_score[col] = (nrow - nlabel) * 1.0 / (nlabel - 1) * scatter_between[col, col] / 1.0
        else:
            feature_score[col] = (nrow - nlabel) * 1.0 / 1.0 * scatter_between[col, col] / 1.0
        print('Feature_' + str(col) + '(' + str(ncol) + '):' + str(feature_score[col]))
    '''

    print('Ranking Features:')

    for col in range(ncol):
        sw = 0.0
        for label in range(nlabel):
            if sample_list[label].__len__() != 0:
                sw += feature_set[sample_list[label], :][:, col].var(axis=0)
            else:
                sw += 0
        sb = 0.0
        for label in range(nlabel):
            sb += (sample_list[label].__len__() * (sample_mean_label[label][0, col] - sample_mean_all[0, col]) ** 2)

        if sw != 0:
            feature_score[col] = 1.0 * (nrow - nlabel) * sb / (nlabel - 1) / sw
        else:
            feature_score[col] = 0

    sorted_index = feature_score.argsort().tolist()  # in ascending order
    sorted_index.reverse()
    sorted_index = np.array(sorted_index)
    feature_score = feature_score[sorted_index]
    return sorted_index, feature_score


'''
Special line if __name__ == “__main__”: indicates the code inside this if statement should only be executed 
when you run the file which contains those code block. It will not executed if the program is imported as a module.
We use if __name__ == “__main__” block to prevent (certain) code from being run when the module is imported.
'''
if __name__ == '__main__':
    organ_choice = "kidney"
    label_num = 8
    path_data = "./dataset/" + organ_choice + "/"
    ld.generate_data_file(path_data + "kidney mlsmote result.csv", path_data, label_num, organ_choice)

    features = ld.import_matrix(path_data + organ_choice + '_feature_data.csv')
    labels = ld.import_matrix(path_data + organ_choice + '_label_data.csv')
    feature_names = np.array(ld.import_data(path_data + organ_choice + '_feature_names.csv')[0])
    label_names = np.array(ld.import_data(path_data + organ_choice + '_label_names.csv')[0])
    features_needs_to_be_sorted = pd.read_csv(path_data + organ_choice + '_feature_data_and_names.csv')

    # apply multi label f-statistic
    index, scores = multi_label_f_statistic(features, labels)
    print('Sorted Index:')
    print(index)
    print('Sorted Scores:')
    print(scores)
    print('Sorted Feature Names:')
    sorted_feature_names = feature_names[index]
    print(sorted_feature_names)

    # save feature score
    feature_score = np.vstack((index, sorted_feature_names, scores))  # Stack arrays in sequence vertically (row wise).
    pd_data = pd.DataFrame(feature_score)
    pd_data.to_csv('./' + organ_choice + '-feature-score.csv', index=False, header=False)

    # creates a figure
    plt.plot(scores)
    plt.xlabel('feature in descending order')
    plt.ylabel('feature score')
    plt.title(organ_choice + ' : Multi Label F Statistic', fontsize='large', fontweight='bold')
    # plt.show()
    plt.savefig("./" + organ_choice + "-MultiLabelFStatistic.jpg")

    # Save results to csv file
    # ld.generate_ordered_data_MLFS(feature_names[index], features_needs_to_be_sorted, organ_choice)
    features_sorted = features_needs_to_be_sorted[sorted_feature_names]
    label_all = pd.DataFrame(labels)
    label_all.columns = label_names.tolist()
    result = pd.concat([label_all, features_sorted], axis=1)  # down rows (axis=0) or along columns (axis=1)
    pd_data2 = pd.DataFrame(result)
    pd_data2.to_csv('./' + organ_choice + '-MLFS-sorted-feature.csv', index=False)



