# -*- coding: utf-8 -*-
"""
Method: Multilabel Synthetic Minority Over-sampling Technique (MLSMOTE).
Description: multilabel datasets (MLDs), a new algorithm aimed to produce synthetic instances for imbalanced MLDs
Reference: MLSMOTE: Approaching imbalanced multilabel learning through synthetic instance generation(2015)
"""

# Importing required Library
import numpy as np
import random
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle


def MLSMOTE(x, y, k_neighbors=5, criterion_rate=1.2, maxstep=740):
    """
    Give the augmented data using MLSMOTE algorithm
    Synthetic Minority Over-sampling Technique on multi-label dataset
    resample only the samples with minority classes

    Feature Vector Generation: (new_X)
    it uses the same SMOTE algorithm to generate the feature vector for the newly generated data.
    it finds the 5 nearest neighbors to the sample points. then draws a line to each of them.
    Then create samples on the lines with class == minority class.

    Every label whose IRPL(l) > MIR is considered as a tail label and all the instance of
    the data which contain that label is considered as minority instance data.


    Parameters
    ----------
    x : pandas.DataFrame, feature vector dataframe
    y : pandas.DataFrame, label set DataFrame
        y(i,j)=0 if i-th sample don't have j-th label, else 1

    k_neighbor: int, number of nearest neighbors

    Returns
    ----------
    X_new : ndarray, shape (n_samples_new, n_features)
            Synthetically generated samples.
    y_new : ndarray, shape (n_samples_new, n_labels)
            Target values for synthetic samples.

    """
    n_samples, n_labels = y.shape  # samples num, labels num

    # computer k-nn, 5 nearest neighbor of all the instance(of each element in X)
    neigh = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean', algorithm='auto')
    neigh.fit(x)

    x_new, y_new = x, y

    full_label_name = y.columns.values.tolist()  # full set of labels names
    rand_seed = 2  # Not actually random. Users would get the same result.
    j = 0
    while True:
        # check the condition to terminate the current iteration
        label_count = np.sum(y_new, axis=0)  # count the number of 1 in each column
        print(label_count)
        print("step-{}: num_majority / num_minority: {}".format(j, np.max(label_count) / np.min(label_count)))
        if np.max(label_count) / np.min(label_count) < criterion_rate:
            break

        irpl, mean_ir = get_IRPL_and_meanIR(y)  # calculate IR per label and meanIR
        count = 0  # store label index

        # search which label is majority or minority
        for each_label in full_label_name:  # loop through labels
            print("step " + str(j) + " loop labels: " + str(each_label))
            if irpl[count] > mean_ir:
                # y.iloc[:, count] give tail label column of the given target dataframe.
                # the index number of the tail label is [count]
                # min_bag = get_all_instances_of_tail_label(each_label)  # return instances of tail label.
                # minBag: bags of minority labels samples
                min_bag_indx = (y.iloc[:, count] == 1)  # give the Boolean index of the tail label instances
                minority_x, minority_y = x[min_bag_indx], y[min_bag_indx]  # bags of minority label samples
                distances, indices = neigh.kneighbors(minority_x, k_neighbors)  # neigh instances indices and distances
                # print(indices)
                # print(distances)

                for neighIndx in indices:
                    print("neighIndx:" + str(neighIndx))  # distance smallest items
                    # A for loop is used for iterating over sample in min_bag
                    # para neighIndx contains reference data point and neighbour data points
                    rand_seed = rand_seed + j + count + 2  # change random seed
                    random.seed(rand_seed)
                    np.random.seed(rand_seed)

                    # find out reference data point and neighbour data points
                    reference = neighIndx[0]
                    print("reference point:" + str(reference))
                    neighbours = np.random.choice(neighIndx[1:], 1)[0]  # get rand neighbour, [0] convert list to int
                    print("neighbours point:" + str(neighbours))

                    # ## Synthetic Data, feature set and label set generation
                    # neighIndx[0] is sample(first element in indices), reference is refNeigh, neighbours is neighbours
                    if np.all(y.iloc[neighIndx[0]] == y.iloc[neighbours]):
                        print("method 1")
                        # sample label set == neighbour label set
                        ratio = random.random()  # generates a random float uniformly in the semi-open range [0.0, 1.0)
                        # Feature Vector Generation
                        temp_x = [(1 - ratio) * x.iloc[neighIndx[0]] + ratio * x.iloc[neighbours]]
                        # Label Set Generation
                        temp_y = [y.iloc[neighIndx[0]]]  # same label set
                    elif (np.sum((y.iloc[neighIndx[0]] == y.iloc[neighbours]))).astype(int) != 0:
                        print("method 2")
                        # sample label set != neighbour label set
                        c = (np.sum((y.iloc[neighIndx[0]] == y.iloc[neighbours]))).astype(float) / n_labels
                        ratio = random.random() * c
                        # Feature Vector Generation
                        temp_x = [(1 - ratio) * x.iloc[neighIndx[0]] + ratio * x.iloc[neighbours]]
                        # Label Set Generation
                        temp_y = (1 - ratio) * y.iloc[neighIndx[0]] + ratio * y.iloc[neighbours]
                        temp_y = [(temp_y > 0.5).astype(int)]
                    else:
                        print("method 3")
                        # reference label set == neighbour label set == all labels are 0
                        ratio = random.random()
                        # Feature Vector Generation
                        temp_x = [(1 - ratio) * x.iloc[neighIndx[0]] + ratio * x.iloc[neighbours]]
                        # Label Set Generation
                        nn_df = y[y.index.isin(neighIndx)]  # all neigh samples labels set
                        ser = nn_df.sum(axis=0, skipna=True)  # the number of occurrences of each label
                        # temp_y = np.array([1 if val > 2 else 0 for val in ser])  # label_generation_method = "Ranking"
                        temp_y = np.array([1 if val > 0 else 0 for val in ser])  # label_generation_method = "Union"
                        temp_y = [temp_y]

                    # print("Label Set Generation: \n"+str(temp_y))

                    temp_x = np.array(temp_x)  # shape:(1,feat_num)
                    temp_y = np.array(temp_y)  # shape:(1,feat_num)

                    x_new = np.concatenate((x_new, temp_x))
                    y_new = np.concatenate((y_new, temp_y))

            count = count + 1

        j += 1
        if j > maxstep:
            break

    x_new, y_new = shuffle(x_new, y_new, random_state=12)
    return x_new, y_new


def singleLabelSMOTE(X, k_neighbors=5, rate=3):
    """
    Synthetic Minority Over-sampling Technique on single-label dataset

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)

    k_neighbors : int

    rate : float

    Returns
    -------
    X_new : ndarray

    """
    N = X.shape[0]
    N_new = int(N * rate)

    # computer k-nn
    X = shuffle(X)
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(X)

    X_new = X

    j = 0

    for x in X:
        knn = neigh.kneighbors(x.reshape(1, -1), k_neighbors)
        for _ in range(int(np.floor(rate))):
            m = random.randint(1, k_neighbors - 1)
            r = random.random()
            k = knn[1][0, m]
            sx = [(1 - r) * x + r * X[k]]
            X_new = np.concatenate((X_new, sx))
            j += 1

    if j >= N_new:
        return X_new
    else:
        X = shuffle(X)
        for x in X:
            knn = neigh.kneighbors(x.reshape(1, -1), k_neighbors)
            m = random.randint(1, k_neighbors - 1)
            r = random.random()
            k = knn[1][0, m]
            sx = [(1 - r) * x + r * X[k]]
            X_new = np.concatenate((X_new, sx))
            j += 1
            if j >= N_new:
                break
        return X_new


def get_IRPL_and_meanIR(label):
    """
       Imbalance ratio per label(irpl): It’s calculated individually for each label.The higher is the IRPL the larger
                                      would be the imbalance, allowing to know which labels are in minority or majority.
       Mean Imbalance ratio(mir): It is defined as the average of IRPL of all the labels.

       abbreviation:
       IR: imbalance ratio
       IRPL/IRL: imbalance ratio per label
       mir/mean_IR: mean imbalance ratio

       args
       y: pandas.DataFrame, the target vector dataframe
    """
    columns = label.columns
    n = len(columns)  # labels num
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = label[columns[column]].value_counts()[1]
    print("per label num: " + str(irpl))  # returns object containing counts of unique values
    irpl = max(irpl) / irpl  # irpl
    print("Imbalance ratio per label(irpl): " + str(irpl))
    print("Max Imbalance ratio(MaxIR): " + str(max(irpl)))
    mir = np.average(irpl)  # mir
    print("Mean Imbalance ratio(mir): " + str(mir))
    return irpl, mir


if __name__ == "__main__":
    """
        main function to use the MLSMOTE
    """
    # -------------------------------DATA LOAD-------------------------------------#
    # read file into dataframes
    dataset = pd.read_csv('datasets/liver.csv')  # dataset to oversample

    # ------------------------GET LABELS AND FEATURES-----------------------------#
    # extract labels from data, split the features-X and labels-y
    X = dataset.iloc[:, 12:]  # features
    Y = dataset.iloc[:, :12]  # labels
    print(X.shape)
    print(Y.shape)

    # --------------------------MLSMOTE ALGORITHM----------------------------------#
    X_new, Y_new = MLSMOTE(X, Y)
    print(X_new.shape)
    print(Y_new.shape)
    X_new = pd.DataFrame(X_new, columns=X.columns)
    Y_new = pd.DataFrame(Y_new, columns=Y.columns)

    result = pd.concat([Y_new, X_new], axis=1)  # down rows (axis=0) or along columns (axis=1)
    pd_data = pd.DataFrame(result)
    pd_data.to_csv('liver mlsmote result.csv', index=False)

    # --------------------------IMBALANCE COMPARISON----------------------------------#
    print("--------Original Dataset---------")
    get_IRPL_and_meanIR(Y)
    print("--------Applying MLSMOTE---------")
    get_IRPL_and_meanIR(Y_new)
