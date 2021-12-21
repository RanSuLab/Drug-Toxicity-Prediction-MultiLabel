# keras 2.4.3,keras applications1.0.8,keras preprocessing 1.1.2
# tensorflow 2.2.0,tensorflow-estimator 2.2.0,tensorboard 2.2.2,tensorboard plugin wit 1.7.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlearn.models import RethinkNet
from mlearn.utils import load_data
from mlearn.criteria import pairwise_f1_score, pairwise_accuracy_score
from mlearn.criteria.sparse_criteria import sparse_pairwise_accuracy_score, sparse_pairwise_f1_score

from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, auc, multilabel_confusion_matrix, roc_curve
from numpy import interp

# debugging:_SymbolicException: Inputs to eager execution function cannot be Keras symbolic tensors
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.random.set_seed(1234)
from sklearn.model_selection import KFold

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

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


def pairwise_accuracy(true_value, prediction_value):
    f1 = 1.0 * ((true_value > 0) & (prediction_value > 0)).sum(axis=1)
    f2 = 1.0 * ((true_value > 0) | (prediction_value > 0)).sum(axis=1)
    f1[f2 == 0] = 1.0
    f1[f2 > 0] /= f2[f2 > 0]
    return f1


def logistic_regression(X, y, k, c_value, base_clf):
    # K-Folds cross-validator
    kf = KFold(n_splits=5)

    XXX = X.iloc[:, 0:k]  # feature subsets
    label_num = y.shape[1]

    ACC = []
    SPE = []
    TPR = []
    F1_score = []
    ACC2 = []

    Label_Acc = [[] for i in range(y.shape[1])]  # expect to access the performance of EACH LABEL
    Average_Label_Accuracy = []
    mean_Label_Acc = []
    Pairwise_Acc = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in kf.split(XXX, y):
        # print("TRAIN:", train_index, "Validation:", validation_index)
        # Try to change the data structure from pandas to numpy instead, using .values
        X_train, X_test = XXX.values[train_index], XXX.values[test_index]
        Y_train, Y_test = y.values[train_index], y.values[test_index]

        if base_clf == 'br':
            clf = OneVsRestClassifier(LogisticRegression(C=c_value, random_state=1))
        elif base_clf == 'cc':
            clf = ClassifierChain(LogisticRegression(C=c_value, random_state=1))
        else:
            print("Sorry, base classifier went wrong. Please try again.")
            break

        # train
        clf.fit(X_train, Y_train)
        # y_prediction
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test.ravel(), y_prob.ravel())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # train set
        y_pred2 = clf.predict(X_train)
        ACC_trn, TPR_trn, SPE_trn, F1_score_trn, cnf_matrix2 = get_eval_para(Y_train, y_pred2)
        ACC2.append(ACC_trn)

        ACC_tst, TPR_tst, SPE_tst, F1_score_tst, cnf_matrix = get_eval_para(Y_test, y_pred)
        Pairwise_Acc_tst = np.mean(pairwise_accuracy(Y_test, y_pred))

        ACC.append(ACC_tst)
        TPR.append(TPR_tst)
        SPE.append(SPE_tst)
        F1_score.append(F1_score_tst)
        Pairwise_Acc.append(Pairwise_Acc_tst)

        each_label_value = []  # each_label_value[0] represents the first label true value
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
            lbl_pred = each_estimators_predict[i]  # the i-th label prediction, each label predict value
            lbl_acc = accuracy_score(each_label_value[i], lbl_pred)  # each label accuracy in one fold
            Label_Acc[i].append(lbl_acc)
            lbl_acc_sum = lbl_acc_sum + lbl_acc

        Average_Label_Accuracy.append(lbl_acc_sum / label_num)

    # calculate mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # compute the average AUC

    # k-fold mean value of label accuracy
    for i in range(0, label_num, 1):
        mean_Label_Acc.append(np.mean(Label_Acc[i]))

    print("--------------test set result------------------")
    print("train:" + str(ACC2))
    print("test:" + str(ACC))
    print(np.mean(ACC2))
    print(organ_choice, base_clf, 'feature number:', k, 'c:', c_value,
          'ACC:', np.mean(ACC), 'TPR:', np.mean(TPR), 'SPE', np.mean(SPE), 'F1_score:', np.mean(F1_score),
          'AUC:', mean_auc, 'Pairwise_Acc:', np.mean(Pairwise_Acc), 'Ave_Label_Acc:', np.mean(Average_Label_Accuracy))

    # Writing CSV Files
    # columns_name = ['organ_choice', 'clf', 'feat_num', 'C', 'ACC', 'TPR', 'spe', 'f1', 'auc', 'pairAcc', 'aveLabAcc',
    #                 'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10',
    #                 'label11', 'label12']
    # list_1 = [organ_choice, base_clf, k, c_value, np.mean(ACC), np.mean(TPR), np.mean(SPE),
    #           np.mean(F1_score), mean_auc, np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    # pd_data = pd.DataFrame(np.array(list_1).reshape(1, 23), columns=columns_name)
    # output_path = "./output/" + organ_choice + "_"+ base_clf +"_LogisticRegression_test_result_fitness.csv"
    # pd_data.to_csv(output_path, index=False)

    return mean_fpr, mean_tpr, mean_auc


def random_forest(X, y, k, n, d_value, base_clf):
    # K-Folds cross-validator
    kf = KFold(n_splits=5)

    XXX = X.iloc[:, 0:k]  # feature subsets
    label_num = y.shape[1]

    ACC = []
    SPE = []
    TPR = []
    F1_score = []
    ACC2 = []

    Label_Acc = [[] for i in range(y.shape[1])]  # expect to access the performance of EACH LABEL
    Average_Label_Accuracy = []
    mean_Label_Acc = []
    Pairwise_Acc = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in kf.split(XXX, y):
        # print("TRAIN:", train_index, "Validation:", validation_index)
        # Try to change the data structure from pandas to numpy instead, using .values
        X_train, X_test = XXX.values[train_index], XXX.values[test_index]
        Y_train, Y_test = y.values[train_index], y.values[test_index]

        if base_clf == 'br':
            clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=n, max_depth=d_value, random_state=1))
        elif base_clf == 'cc':
            clf = ClassifierChain(RandomForestClassifier(n_estimators=n, max_depth=d_value, random_state=1))
        else:
            print("Sorry, base classifier went wrong. Please try again.")
            break

        # train
        clf.fit(X_train, Y_train)
        # y_prediction
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test.ravel(), y_prob.ravel())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # train set
        y_pred2 = clf.predict(X_train)
        ACC_trn, TPR_trn, SPE_trn, F1_score_trn, cnf_matrix2 = get_eval_para(Y_train, y_pred2)
        ACC2.append(ACC_trn)

        ACC_tst, TPR_tst, SPE_tst, F1_score_tst, cnf_matrix = get_eval_para(Y_test, y_pred)
        Pairwise_Acc_tst = np.mean(pairwise_accuracy(Y_test, y_pred))

        ACC.append(ACC_tst)
        TPR.append(TPR_tst)
        SPE.append(SPE_tst)
        F1_score.append(F1_score_tst)
        Pairwise_Acc.append(Pairwise_Acc_tst)

        each_label_value = []  # each_label_value[0] represents the first label true value
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
            lbl_pred = each_estimators_predict[i]  # the i-th label prediction, each label predict value
            lbl_acc = accuracy_score(each_label_value[i], lbl_pred)  # each label accuracy in one fold
            Label_Acc[i].append(lbl_acc)
            lbl_acc_sum = lbl_acc_sum + lbl_acc

        Average_Label_Accuracy.append(lbl_acc_sum / label_num)

    # calculate mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # k-fold mean value of label accuracy
    for i in range(0, label_num, 1):
        mean_Label_Acc.append(np.mean(Label_Acc[i]))

    print("--------------test set result------------------")
    print("train:" + str(ACC2))
    print("test:" + str(ACC))
    print(np.mean(ACC2))
    print(organ_choice, base_clf, 'feature number:', k, 'n:', n, 'd:', d_value,
          'ACC:', np.mean(ACC), 'TPR:', np.mean(TPR), 'SPE', np.mean(SPE), 'F1_score:', np.mean(F1_score),
          'AUC:', mean_auc, 'Pairwise_Acc:', np.mean(Pairwise_Acc), 'Ave_Label_Acc:', np.mean(Average_Label_Accuracy))

    # Writing CSV Files
    # columns_name = ['organ_choice', 'clf', 'feat_num', 'n', 'd', 'ACC', 'TPR', 'spe', 'f1', 'auc', 'pairAcc', 'aveLabAcc',
    #                 'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10',
    #                 'label11', 'label12']
    # list_1 = [organ_choice, base_clf, k, n, d_value, np.mean(ACC), np.mean(TPR), np.mean(SPE),
    #           np.mean(F1_score), mean_auc, np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    # pd_data = pd.DataFrame(np.array(list_1).reshape(1, 24), columns=columns_name)
    # output_path = "./output/" + organ_choice + "_" + base_clf + "_randomForest_test_result_fitness.csv"
    # pd_data.to_csv(output_path, index=False)

    return mean_fpr, mean_tpr, mean_auc


def linear_svm(X, y, k, c_value, base_clf):
    # K-Folds cross-validator
    kf = KFold(n_splits=5)

    XXX = X.iloc[:, 0:k]  # feature subsets
    label_num = y.shape[1]

    ACC = []
    SPE = []
    TPR = []
    F1_score = []
    ACC2 = []

    Label_Acc = [[] for i in range(y.shape[1])]  # expect to access the performance of EACH LABEL
    Average_Label_Accuracy = []
    mean_Label_Acc = []
    Pairwise_Acc = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in kf.split(XXX, y):
        # print("TRAIN:", train_index, "Validation:", validation_index)
        # Try to change the data structure from pandas to numpy instead, using .values
        X_train, X_test = XXX.values[train_index], XXX.values[test_index]
        Y_train, Y_test = y.values[train_index], y.values[test_index]

        if base_clf == 'br':
            clf = OneVsRestClassifier(LinearSVC(C=c_value, random_state=1))
        elif base_clf == 'cc':
            clf = ClassifierChain(LinearSVC(C=c_value, random_state=1))
        else:
            print("Sorry, base classifier went wrong. Please try again.")
            break

        # train
        clf.fit(X_train, Y_train)
        # y_prediction
        y_pred = clf.predict(X_test)
        y_prob = clf.decision_function(X_test)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test.ravel(), y_prob.ravel())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # train set
        y_pred2 = clf.predict(X_train)
        ACC_trn, TPR_trn, SPE_trn, F1_score_trn, cnf_matrix2 = get_eval_para(Y_train, y_pred2)
        ACC2.append(ACC_trn)

        ACC_tst, TPR_tst, SPE_tst, F1_score_tst, cnf_matrix = get_eval_para(Y_test, y_pred)
        Pairwise_Acc_tst = np.mean(pairwise_accuracy(Y_test, y_pred))

        ACC.append(ACC_tst)
        TPR.append(TPR_tst)
        SPE.append(SPE_tst)
        F1_score.append(F1_score_tst)
        Pairwise_Acc.append(Pairwise_Acc_tst)

        each_label_value = []  # each_label_value[0] represents the first label true value
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
            lbl_pred = each_estimators_predict[i]  # the i-th label prediction, each label predict value
            lbl_acc = accuracy_score(each_label_value[i], lbl_pred)  # each label accuracy in one fold
            Label_Acc[i].append(lbl_acc)
            lbl_acc_sum = lbl_acc_sum + lbl_acc

        Average_Label_Accuracy.append(lbl_acc_sum / label_num)

    # calculate mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # k-fold mean value of label accuracy
    for i in range(0, label_num, 1):
        mean_Label_Acc.append(np.mean(Label_Acc[i]))

    print("--------------test set result------------------")
    print("train:" + str(ACC2))
    print("test:" + str(ACC))
    print(np.mean(ACC2))
    print(organ_choice, base_clf, 'feature number:', k, 'c:', c_value,
          'ACC:', np.mean(ACC), 'TPR:', np.mean(TPR), 'SPE', np.mean(SPE), 'F1_score:', np.mean(F1_score),
          'AUC:', mean_auc, 'Pairwise_Acc:', np.mean(Pairwise_Acc), 'Ave_Label_Acc:', np.mean(Average_Label_Accuracy))

    # Writing CSV Files
    # columns_name = ['organ_choice', 'clf', 'feat_num', 'C', 'ACC', 'TPR', 'spe', 'f1', 'auc', 'pairAcc', 'aveLabAcc',
    #                 'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10',
    #                 'label11', 'label12']
    # list_1 = [organ_choice, base_clf, k, c_value, np.mean(ACC), np.mean(TPR), np.mean(SPE),
    #           np.mean(F1_score), mean_auc, np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    # pd_data = pd.DataFrame(np.array(list_1).reshape(1, 23), columns=columns_name)
    # output_path = "./output/" + organ_choice + "_" + base_clf + "_LinearSVC_test_result_fitness.csv"
    # pd_data.to_csv(output_path, index=False)

    return mean_fpr, mean_tpr, mean_auc


def classification_roc(feature, label):
    # ## the best found parameters
    # logistic_regression
    k_lr_br = 58
    c_lr_br = 1000
    k_lr_cc = 66
    c_lr_cc = 100

    # svm
    k_svm_br = 82
    c_svm_br = 10
    k_svm_cc = 78
    c_svm_cc = 10

    # random_forest
    k_rf_br = 51
    n_rf_br = 10
    d_rf_br = 5
    k_rf_cc = 37
    n_rf_cc = 10
    d_rf_cc = 5

    # All Classifier
    mean_fpr_lr_br, mean_tpr_lr_br, mean_auc_lr_br = logistic_regression(feature, label, k_lr_br, c_lr_br, 'br')
    mean_fpr_lr_cc, mean_tpr_lr_cc, mean_auc_lr_cc = logistic_regression(feature, label, k_lr_cc, c_lr_cc, 'cc')

    mean_fpr_rf_br, mean_tpr_rf_br, mean_auc_rf_br = random_forest(feature, label, k_rf_br, n_rf_br, d_rf_br, 'br')
    mean_fpr_rf_cc, mean_tpr_rf_cc, mean_auc_rf_cc = random_forest(feature, label, k_rf_cc, n_rf_cc, d_rf_cc, 'cc')

    mean_fpr_svm_br, mean_tpr_svm_br, mean_auc_svm_br = linear_svm(feature, label, k_svm_br, c_svm_br, 'br')
    mean_fpr_svm_cc, mean_tpr_svm_cc, mean_auc_svm_cc = linear_svm(feature, label, k_svm_cc, c_svm_cc, 'cc')

    # Draw Diagonal Lines
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)

    # ##-----------------BinaryRelevance-----------------
    plt.plot(mean_fpr_lr_br, mean_tpr_lr_br, color='ForestGreen',
             label=r'BR+LR(Area=%0.4f)' % mean_auc_lr_br, lw=1, alpha=.8)

    plt.plot(mean_fpr_rf_br, mean_tpr_rf_br, color='Navy',
             label=r'BR+RF(Area=%0.4f)' % mean_auc_rf_br, lw=1, alpha=.8)

    plt.plot(mean_fpr_svm_br, mean_tpr_svm_br, color='SaddleBrown',
             label=r'BR+SVM(Area=%0.4f)' % mean_auc_svm_br, lw=1, alpha=.8)

    # ##-----------------ClassifierChain-----------------
    plt.plot(mean_fpr_lr_cc, mean_tpr_lr_cc, color='DarkCyan',
             label=r'CC+LR(Area=%0.4f)' % mean_auc_lr_cc, lw=1, alpha=.8)

    plt.plot(mean_fpr_rf_cc, mean_tpr_rf_cc, color='DarkOrange',
             label=r'CC+RF(Area=%0.4f)' % mean_auc_rf_cc, lw=1, alpha=.8)

    plt.plot(mean_fpr_svm_cc, mean_tpr_svm_cc, color='DarkMagenta',
             label=r'CC+SVM(Area=%0.4f)' % mean_auc_svm_cc, lw=1, alpha=.8)

    # # Plot them on canvas and give a name to x-axis and y-axis
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc='lower right')
    # plt.savefig("./output/" + organ_choice + "-roc-curve.pdf")
    # plt.show()


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

    # ##---------------tradition methods---------------
    dataset2 = pd.read_csv('./data/liver-MLFS-sorted-feature.csv')
    print("dataset.shape: " + str(dataset2.shape))

    # split the features-X and class labels-y
    X2 = dataset2.iloc[:, 12:]
    y2 = dataset2.iloc[:, :12]

    # Normalise the data
    X2 = (X2 - X2.min()) / (X2.max() - X2.min())

    classification_roc(X2, y2)

    # Plot them on canvas and give a name to x-axis and y-axis
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("./" + organ_choice + "-002tradition-roc-curve.pdf")
    plt.show()

