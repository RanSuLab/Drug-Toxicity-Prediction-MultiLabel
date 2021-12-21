# keras 2.4.3,keras applications1.0.8,keras preprocessing 1.1.2
# tensorflow 2.2.0,tensorflow-estimator 2.2.0,tensorboard 2.2.2,tensorboard plugin wit 1.7.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import seaborn as sns
import math
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler

from mlearn.models import RethinkNet
from mlearn.utils import load_data
from mlearn.criteria import pairwise_f1_score, pairwise_accuracy_score
from mlearn.criteria.sparse_criteria import sparse_pairwise_accuracy_score, sparse_pairwise_f1_score

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, auc, multilabel_confusion_matrix, roc_curve, confusion_matrix
from numpy import interp

# debugging:_SymbolicException: Inputs to eager execution function cannot be Keras symbolic tensors
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.random.set_seed(1234)
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")


def get_eval_para(true_value, prediction_value):
    cnf_matrix = multilabel_confusion_matrix(true_value, prediction_value)
    # print(cnf_matrix)

    TN = cnf_matrix[:, 0, 0]
    TP = cnf_matrix[:, 1, 1]
    FN = cnf_matrix[:, 1, 0]
    FP = cnf_matrix[:, 0, 1]
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    ACC = accuracy_score(true_value, prediction_value)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = metrics.recall_score(true_value, prediction_value, average='macro')

    # Specificity or true negative rate
    TNR = np.mean(TN / (TN + FP))

    # Precision or positive predictive value
    F1_score = f1_score(true_value, prediction_value, average='macro')

    return ACC, TPR, TNR, F1_score, cnf_matrix


def rethinkNet_example(X, Y, organ_choice):
    # K-Folds cross-validator
    kf = KFold(n_splits=5)

    # l2 parameter
    param = 10 ** -6

    feat_num = X.shape[1]
    label_num = Y.shape[1]

    ACC = []
    SPE = []
    TPR = []
    F1_score = []
    ACC2 = []

    scoring_fn = pairwise_accuracy_score
    Pairwise_Acc = []

    Label_Acc = [[] for i in range(Y.shape[1])]  # expect to access the performance of EACH LABEL
    mean_Label_Acc = []  # k fold average
    Average_Label_Accuracy = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fold = 1
    for train_index, test_index in kf.split(X, Y):
        print(str(fold)+" fold:")
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Initialize the classifier to be use
        clf = RethinkNet(n_features=feat_num, n_labels=label_num, scoring_fn=sparse_pairwise_accuracy_score,
                         l2w=param, architecture="arch_002")
        # train model
        clf.train(X_train, Y_train)
        # predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_probability(X_test)

        # train set
        trn_pred = clf.predict(X_train)
        ACC_trn, TPR_trn, SPE_trn, F1_score_trn, cnf_matrix_trn = get_eval_para(Y_train, trn_pred)
        print("ACC_trn: " + str(ACC_trn))
        ACC2.append(ACC_trn)

        ACC_tst, TPR_tst, SPE_tst, F1_score_tst, cnf_matrix = get_eval_para(Y_test, y_pred)
        print("ACC_tst: " + str(ACC_tst))
        ACC.append(ACC_tst)
        TPR.append(TPR_tst)
        SPE.append(SPE_tst)
        F1_score.append(F1_score_tst)

        # calculate pairwise_accuracy_score
        test_pred = clf.predict(X_test).todense()
        test_avg_score = np.mean(scoring_fn(Y_test, test_pred.astype(int)))
        Pairwise_Acc.append(test_avg_score)

        # compute ROC curve and area the curve
        y_prob = y_prob.toarray()
        fpr, tpr, thresholds = roc_curve(Y_test.ravel(), y_prob.ravel())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # calculate Average_Label_Accuracy and Label_Acc
        y_pred = y_pred.toarray()
        each_label_value = []
        each_estimators_predict = []

        # transposition, save each label true value in a list
        for i in range(len(Y_test[0])):  # row num
            t = []
            for j in range(len(Y_test)):
                t.append(Y_test[j][i])
            each_label_value.append(t)

        # transposition, save each label predict value in a list
        for i in range(len(y_pred[0])):  # row num
            t = []
            for j in range(len(y_pred)):
                t.append(y_pred[j][i])
            each_estimators_predict.append(t)

        lbl_acc_sum = 0
        for i in range(0, label_num, 1):
            lbl_acc = accuracy_score(each_label_value[i], each_estimators_predict[i])  # each label accuracy in one fold
            Label_Acc[i].append(lbl_acc)
            lbl_acc_sum = lbl_acc_sum + lbl_acc

        Average_Label_Accuracy.append(lbl_acc_sum / label_num)

        fold = fold + 1

    # calculate mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # k-fold mean value of label accuracy
    for i in range(0, label_num, 1):
        mean_Label_Acc.append(np.mean(Label_Acc[i]))

    print("-------------- result ------------------")
    print("train:" + str(np.mean(ACC2)))
    print("test:" + str(np.mean(ACC)))
    print(organ_choice, 'ACC:', np.mean(ACC), 'TPR:', np.mean(TPR), 'SPE', np.mean(SPE), 'F1_score:', np.mean(F1_score),
          'AUC:', mean_auc, 'Pairwise_Acc:', np.mean(Pairwise_Acc), 'Ave_Label_Acc:', np.mean(Average_Label_Accuracy))

    # Writing CSV Files
    columns_name = ['organ_choice', 'ACC', 'TPR', 'spe', 'f1', 'auc', 'pairAcc', 'aveLabAcc',
                    'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8',
                    'label9', 'label10', 'label11', 'label12']
    list_1 = [organ_choice, np.mean(ACC), np.mean(TPR), np.mean(SPE), np.mean(F1_score), mean_auc,
              np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    pd_data = pd.DataFrame(np.array(list_1).reshape(1, 20), columns=columns_name)
    output_path = "./" + organ_choice + "_NET002_result.csv"
    pd_data.to_csv(output_path, index=False)

    # Plot ROC curve with matplotlib
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)  # Draw Diagonal Lines
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='firebrick', label=r'Att-RethinkNet(Area=%0.4f)' % mean_auc, lw=1, alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("./" + organ_choice + "-NET002-roc-curve.pdf")
    # plt.show()


def cooccurrence_matrix(label, categories, fold):
    lbl_num = label.shape[1]

    # Compute cooccurrence matrix
    cooccurrence_matrix = np.dot(label.transpose(), label)
    # print('\ncooccurrence_matrix:\n{0}'.format(cooccurrence_matrix))

    # Compute the Jaccard coefficient
    jaccard_matrix = np.zeros(shape=(lbl_num, lbl_num))
    for i in range(lbl_num):
        for j in range(lbl_num):
            lbl_occur_union = cooccurrence_matrix[i][i] + cooccurrence_matrix[j][j] - cooccurrence_matrix[i][j]
            jaccard_matrix[i][j] = np.true_divide(cooccurrence_matrix[i][j], lbl_occur_union)
    # print('\njaccard_matrix:\n{0}'.format(jaccard_matrix))

    # Compute normalized similarities matrix
    norm_similarity_matrix = np.zeros(shape=(lbl_num, lbl_num))
    for i in range(lbl_num):
        quadratic_sum = 0
        for j in range(lbl_num):
            quadratic_sum = quadratic_sum + math.pow(jaccard_matrix[i][j], 2)
        norm_numerator = math.sqrt(quadratic_sum)
        norm_similarity_matrix[i, :] = np.true_divide(jaccard_matrix[i, :], norm_numerator)
    # print('\nnorm_similarity_matrix:\n{0}'.format(norm_similarity_matrix))

    # draw
    df_cm_per = pd.DataFrame(jaccard_matrix, index=categories, columns=categories)
    fig = plt.figure(figsize=(12, 8))

    # annot: If True, write the data value in each cell. annot_kws={"fontsize": "small/medium/large"}
    Heatmap = sns.heatmap(df_cm_per, annot=True, fmt=".2f", annot_kws={"fontsize": "medium"}, cmap='OrRd',
                          linewidths=0.5,
                          linecolor='white', square=True, mask=False, cbar_kws={"orientation": "vertical"}, cbar=True)

    plt.xticks(rotation=90, fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()  # fix the xlabels cutoff. X-axis text is too lengthy
    plt.savefig("Jaccard Matrix for fold " + str(fold+1) + ".pdf")
    # plt.show()
    plt.close()

    return norm_similarity_matrix


def get_evaluation_single(true_value, prediction_value):
    cnf_matrix = confusion_matrix(true_value, prediction_value)
    # print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    ACC = accuracy_score(true_value, prediction_value)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = metrics.recall_score(true_value, prediction_value, average='macro')

    # Specificity or true negative rate
    TNR = np.mean(TN / (TN + FP))

    # Precision or positive predictive value
    F1_score = metrics.f1_score(true_value, prediction_value, average='macro')

    return ACC, TPR, TNR, F1_score, cnf_matrix


def get_evaluation_multi(true_value, prediction_value):
    cnf_matrix = multilabel_confusion_matrix(true_value, prediction_value)
    # print(cnf_matrix)

    TN = cnf_matrix[:, 0, 0]
    TP = cnf_matrix[:, 1, 1]
    FN = cnf_matrix[:, 1, 0]
    FP = cnf_matrix[:, 0, 1]
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    ACC = accuracy_score(true_value, prediction_value)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = metrics.recall_score(true_value, prediction_value, average='macro')

    # Specificity or true negative rate
    TNR = np.mean(TN / (TN + FP))

    # Precision or positive predictive value
    F1_score = metrics.f1_score(true_value, prediction_value, average='macro')

    return ACC, TPR, TNR, F1_score


def pairwise_accuracy(true_value, prediction_value):
    f1 = 1.0 * ((true_value > 0) & (prediction_value > 0)).sum(axis=1)
    f2 = 1.0 * ((true_value > 0) | (prediction_value > 0)).sum(axis=1)
    f1[f2 == 0] = 1.0
    f1[f2 > 0] /= f2[f2 > 0]
    return f1


def jackknife_resampling(data):
    """
    Performs jackknife resampling on numpy arrays.
    Jackknife resampling is a technique to generate 'n' deterministic samples
    of size 'n-1' from a measured sample of size 'n'. Basically, the i-th
    sample, (1<=i<=n), is generated by means of removing the i-th measurement
    of the original sample. Like the bootstrap resampling, this statistical
    technique finds applications in estimating variance, bias, and confidence
    intervals.
    Parameters
    ----------
    data : numpy.ndarray
        Original sample (1-D array) from which the jackknife resamples will be
        generated.
    Returns
    -------
    resamples : numpy.ndarray
        The i-th row is the i-th jackknife sample, i.e., the original sample
        with the i-th measurement deleted.
    References
    ----------
    .. [1] McIntosh, Avery. "The Jackknife Estimation Method".
        <http://people.bu.edu/aimcinto/jackknife.pdf>
    .. [2] Efron, Bradley. "The Jackknife, the Bootstrap, and other
        Resampling Plans". Technical Report No. 63, Division of Biostatistics,
        Stanford University, December, 1980.
    .. [3] Cowles, Kate. "Computing in Statistics: The Jackknife, Lecture 11".
        <http://homepage.stat.uiowa.edu/~kcowles/s166_2009/lect11.pdf>.
        September, 2009.
    .. [4] Jackknife resampling <https://www.stat.berkeley.edu/~hhuang/STAT152/Jackknife-Bootstrap.pdf>
    .. [5] Jackknife resampling <https://en.wikipedia.org/wiki/Jackknife_resampling>
    """
    n = data.shape[0]
    if n <= 0:
        raise ValueError("data must contain at least one measurement.")

    resamples = np.empty([n, n-1])

    for i in range(n):
        resamples[i] = np.delete(data, i)

    return resamples


def delete_group_jackknife_resampling(data, n_delete=1):
    """
    An alternate the jackknife method of deleting one observation at a time is to delete d.
    this function only delete d in order, does not fulfill a combination C(n,d).
    this function can: verify whether jackknife_resampling function is running

    TODO:
    C(n,d）
    delete_d_jackknife_resampling
    It is natural to extend the delete-1 jackknife by omitting more than one observation.
    Instead of leaving out one observation at a time, we leave out d observations.
    Therefore, the size of a delete-d jackknife sample is (n-d), and there are C(n,d) jackknife samples.
    In the simplest case jackknife resampling is generated by sequentially deleting single cases
    from the original sample (delete‐one jackknife).
    A more generalized jackknife technique uses resampling that is based on multiple case deletion (delete‐d jackknife).

    References
    ----------
        https://www.stat.berkeley.edu/~hhuang/STAT152/Jackknife-Bootstrap.pdf
    """
    n = data.shape[0]  # number of data points
    if n <= 0:
        raise ValueError("data must contain at least one measurement.")

    # combins = [c for c in combinations(range(n), n_delete)]  # C(n,d)
    n_sub = n // n_delete  # number of data subset(model)(In Python 3, a//b floor division, a/b "true" (float) division)
    start = 0  # start of current block of data
    resamples = np.empty([n_sub, n-n_delete])

    for i in range(0, n_sub):
        end = np.minimum(start + n_delete, n)
        r = range(start, end)

        resamples[i] = np.delete(data, r)
        start = end

    return resamples


def prediction_model(X, y, categories, threshold, organ_choice):
    # K-Folds cross-validator
    fold_num = 5
    kf = KFold(n_splits=fold_num)
    label_num = y.shape[1]

    ACC = []
    SPE = []
    TPR = []
    F1_score = []

    Label_Acc = [[] for i in range(label_num)]  # expect to access the performance of EACH LABEL
    Average_Label_Accuracy = []
    mean_Label_Acc = []
    Pairwise_Acc = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    integrative_prediction_score_save = [[] for i in range(fold_num)]

    # define undersample strategy
    rus = RandomUnderSampler(random_state=222, sampling_strategy='majority')  # resample the majority class

    fold = 0
    for train_index, test_index in kf.split(X, y):
        print("K-fold cross validation: fold " + str(fold+1))
        # print("TRAIN:", train_index, "Validation:", validation_index)
        # Try to change the data structure from pandas to numpy instead, using .values
        X_train, X_test = X.values[train_index], X.values[test_index]
        Y_train, Y_test = y.values[train_index], y.values[test_index]

        # occurrence matrix that describes the presentation states of training samples
        normalized_similarity_matrix = cooccurrence_matrix(Y_train, categories, fold)

        clf = KNeighborsClassifier(n_neighbors=5)

        tst_sample_num = X_test.shape[0]

        prediction_score = np.zeros([label_num, tst_sample_num])
        for lbl in range(label_num):
            # # fit and apply the transform
            # X_train_rus, Y_train_rus = rus.fit_resample(X_train, Y_train[:, lbl])
            # print("Y_train_rus" + str(Y_train_rus))
            # print('Resampled dataset shape %s' % collections.Counter(Y_train_rus))

            # to get sample indices from RandomUnderSampler in imblearn
            # As the undersampling is solely based on the y_vector,
            # one can add a counter-variable instead of the the x-vector/array
            counter = range(0, len(Y_train[:, lbl]))
            counter = np.array(counter).reshape(-1, 1)
            train_index_rus, Y_train_rus = rus.fit_resample(counter, Y_train[:, lbl])
            print('Resampled dataset shape %s' % collections.Counter(Y_train_rus))

            train_index_res = jackknife_resampling(train_index_rus)
            # train_index_res = delete_group_jackknife_resampling(train_index_rus, 5)
            # print("train_index_res:"+str(train_index_res))  # train set index after jackknife resample
            n_sub = train_index_res.shape[0]

            sub_prediction_score = np.zeros([tst_sample_num, n_sub])  # n sub prediction models for each label
            for idx in range(n_sub):
                idx_res = train_index_res[idx].astype(np.int)
                X_train_res = X.values[idx_res]
                Y_train_res = y.values[idx_res]

                clf.fit(X_train_res, Y_train_res[:, lbl])  # the i-th label
                y_score = clf.predict_proba(X_test)
                # print("y_score:" + str(y_score))
                y_pred = clf.predict(X_test)
                # print("y_pred:" + str(y_pred))

                # store sub prediction score
                for sample in range(tst_sample_num):
                    # sub_prediction_score[sample][idx] = y_score[sample][1]  # higher than a threshold (would occur)
                    sub_prediction_score[sample][idx] = y_pred[sample]
            # print("sub_prediction_score:" + str(sub_prediction_score))

            for sample2 in range(tst_sample_num):
                prediction_score[lbl][sample2] = np.mean(sub_prediction_score[sample2, :])
            # print("prediction_score(lbl):" + str(prediction_score))
        print("prediction_score(model):" + str(prediction_score))

        # without normalized similarity matrix (without weight)
        # for row in range(0, prediction_score.shape[0]):
        #     for col in range(0, prediction_score.shape[1]):
        #         if prediction_score[row, col] > threshold:
        #             prediction_score[row, col] = 1
        #         else:
        #             prediction_score[row, col] = 0
        # print(prediction_score)

        # with normalized similarity matrix (with weight)
        integrative_prediction_score = np.dot(normalized_similarity_matrix, prediction_score)  # matrix multiplication
        print("integrative_prediction_score:" + str(integrative_prediction_score))
        integrative_predict = np.zeros((integrative_prediction_score.shape[0], integrative_prediction_score.shape[1]))
        for row in range(0, integrative_prediction_score.shape[0]):
            for col in range(0, integrative_prediction_score.shape[1]):
                if integrative_prediction_score[row, col] > threshold:
                    integrative_predict[row, col] = 1
                else:
                    integrative_predict[row, col] = 0
        # print("integrative_predict:" + str(integrative_predict))
        # print("Y_test:" + str(Y_test))

        # save each fold integrative_prediction_score
        integrative_prediction_score_save[fold] = integrative_prediction_score  # add each fold in a list
        # print("integrative_prediction_score_save:"+str(integrative_prediction_score_save))

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test.ravel(), integrative_predict.ravel())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        ACC_tst, TPR_tst, SPE_tst, F1_score_tst = get_evaluation_multi(Y_test, integrative_predict.T)
        Pairwise_Acc_tst = np.mean(pairwise_accuracy(Y_test, integrative_predict.T))
        ACC.append(ACC_tst)
        TPR.append(TPR_tst)
        SPE.append(SPE_tst)
        F1_score.append(F1_score_tst)
        Pairwise_Acc.append(Pairwise_Acc_tst)

        each_label_value = []  # each_label_value[0] represents the first label true value
        each_estimators_predict = integrative_predict.tolist()

        # transposition, save each label true value in a list
        for i in range(len(Y_test[0])):  # row num
            t = []
            for j in range(len(Y_test)):
                t.append(Y_test[j][i])
            each_label_value.append(t)

        lbl_acc_sum = 0
        for i in range(0, label_num, 1):
            lbl_pred = each_estimators_predict[i]  # the i-th label prediction, each label predict value
            lbl_acc = accuracy_score(each_label_value[i], lbl_pred)  # each label accuracy in one fold
            Label_Acc[i].append(lbl_acc)
            lbl_acc_sum = lbl_acc_sum + lbl_acc

        Average_Label_Accuracy.append(lbl_acc_sum / label_num)

        fold = fold + 1

    # Draw Diagonal Lines
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    # calculate mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='Blue',
             label=r'Integrative model(Area=%0.4f)' % mean_auc, lw=1, alpha=.8)

    # k-fold mean value of label accuracy
    for i in range(0, label_num, 1):
        mean_Label_Acc.append(np.mean(Label_Acc[i]))

    print("--------------test set result------------------")
    print("test:" + str(ACC))
    print(organ_choice, 'ACC:', np.mean(ACC), 'TPR:', np.mean(TPR), 'SPE', np.mean(SPE), 'F1_score:', np.mean(F1_score),
          'AUC:', mean_auc, 'Pairwise_Acc:', np.mean(Pairwise_Acc), 'Ave_Label_Acc:', np.mean(Average_Label_Accuracy))

    # Writing CSV Files
    columns_name = ['organ_choice', 'ACC', 'TPR', 'spe', 'f1', 'auc', 'pairAcc', 'aveLabAcc',
                    'label1', 'label2', 'label3', 'label4', 'label5', 'label6',
                    'label7', 'label8', 'label9', 'label10', 'label11', 'label12']
    list_1 = [organ_choice, np.mean(ACC), np.mean(TPR), np.mean(SPE), np.mean(F1_score), mean_auc,
              np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    pd_data = pd.DataFrame(np.array(list_1).reshape(1, 20), columns=columns_name)
    output_path = "./" + organ_choice + "_integrative_model_test_result.csv"
    pd_data.to_csv(output_path, index=False)

    print("integrative_prediction_score_save:" + str(integrative_prediction_score_save))
    return integrative_prediction_score_save


if __name__ == '__main__':
    organ_choice = "liver"

    dataset = pd.read_csv('./data/liver mlsmote result.csv')
    print("dataset.shape: " + str(dataset.shape))
    # split the features-X and class labels-y
    X = dataset.iloc[:, 12:]  # features
    Y = dataset.iloc[:, :12]  # labels
    # Normalise the data
    X = (X - X.min()) / (X.max() - X.min())  # min-max normalization, X are mapped to the range 0 to 1.
    X = X.values
    Y = Y.values

    print("Running Att-RethinkNet ...")
    rethinkNet_example(X, Y, organ_choice)

    # ##---------------integrative methods---------------
    dataset2 = pd.read_csv('./data/liver-MLFS-sorted-feature.csv')
    print("dataset.shape: " + str(dataset2.shape))

    # split the features-X and class labels-y
    X2 = dataset2.iloc[:, 12:]
    y2 = dataset2.iloc[:, :12]

    # keep label names in a list
    categories = y2.columns.tolist()
    if organ_choice == "liver":
        categories = ['Cellular infiltration', 'Eosinophilic change', 'Hypertrophy', 'Increased mitosis',
                      'NOS lesion', 'Microgranuloma', 'Necrosis', 'Hepatodiaphragmatic nodule',
                      'Kupffer cell proliferation', 'Single cell necrosis', 'Swelling', 'Cytoplasmic vacuolization']
    elif organ_choice == "kidney":
        categories = ['Hyaline cast', "Lymphocyte cellular infiltration", 'Basophilic change', 'Cyst',
                      'Dilatation', 'Cystic dilatation', 'Necrosis', 'Regeneration']

    # Normalise the data
    X2 = (X2 - X2.min()) / (X2.max() - X2.min())

    threshold = 0.5
    features_select = X2.iloc[:, 0: 200]  # feature subsets
    integrative_prediction_score_save = prediction_model(features_select, y2, categories, threshold, organ_choice)

    # ## ------------------------plot------------------------
    # Plot them on canvas and give a name to x-axis and y-axis
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("./" + organ_choice + "-002integrative-roc-curve.pdf")
    plt.show()

