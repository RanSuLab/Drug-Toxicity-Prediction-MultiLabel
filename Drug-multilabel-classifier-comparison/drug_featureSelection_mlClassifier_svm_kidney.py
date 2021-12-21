"""
Topic：Drug Multilabel Classifier

Method: 1)Binary Relevance; 2)Classifier Chains.
1)Binary Relevance Classifier: A Binary Relevance Classifier has been implemented in which independent base classifiers
are implemented for each label. This uses a one-vs-all approach to generate the training sets for each base classifier.
2)Classifier Chains: A classifier chain model generates a chain of binary classifiers each of predicts the presence or
absence of a specific label. The in input to each classifier in the chain, however, includes the original descriptive
features plus the outputs of the classifiers so far in the chain. This allows label associations to be taken into
account.

Description: Consider BR=Binary Relevance Classifier,
                      CC=Classifier Chains.
Dataset: The drug dataset is formed by micro-array expression.
         The folder "data" in this project contains the dataset in the form of a CSV file.
"""
import pandas as pd
import numpy as np
import warnings

from numpy import interp
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, GridSearchCV
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, auc, multilabel_confusion_matrix, roc_curve
from sklearn import metrics, exceptions
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import LinearSVC, SVC

# warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=exceptions.UndefinedMetricWarning)


def platt_func(x):
    return 1 / (1 + np.exp(-x))


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


def pairwise_accuracy(true_value, prediction_value):
    f1 = 1.0 * ((true_value > 0) & (prediction_value > 0)).sum(axis=1)
    f2 = 1.0 * ((true_value > 0) | (prediction_value > 0)).sum(axis=1)
    f1[f2 == 0] = 1.0
    f1[f2 > 0] /= f2[f2 > 0]
    return f1


# ## Use GridSearchCV to Find the Best Model Parameters of the Binary Relevance Algorithm
def br_feature_selection_and_train(X, y, organ_choice, feat_select):
    # K-Folds cross-validator
    kf = KFold(n_splits=5)

    label_num = y.shape[1]

    # save minimum value from the duplicated/repeated values
    mydict_ACC = {}
    mydict_c = {}
    final_ACC = []

    # feature selection column by column
    for feature_selected in range(1, feat_select+1, 1):
        print("selected feature num:" + str(feature_selected))
        XX = X.iloc[:, :feature_selected]  # feature subsets

        # Parameter tuning. base models: LinearSVC()
        c_value_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        # c_value_list = [1]

        # according to acc, store max scores, other max scores vary with max_ACC
        max_ACC = 0  # Accuracy
        max_SPE = 0  # Specificity
        max_TPR = 0  # Sensitivity
        max_F1 = 0  # F1 Scores
        max_c_value = 0
        max_AUC = 0  # Area Under the ROC Curve
        max_Pairwise_Acc = 0  # pairwise accuracy
        max_Average_Label_Accuracy = 0  # average label accuracy
        max_Label_Acc_list = []  # each label accuracy

        # Using the custom GridSearchCV
        for c_value in c_value_list:
            ACC_list = []
            SPE_list = []
            TPR_list = []
            F1_score_list = []

            Label_Acc_list = [[] for i in range(y.shape[1])]  # expect to access the performance of EACH LABEL
            Average_Label_Accuracy_list = []
            mean_Label_Acc_list = []
            Pairwise_Acc_list = []

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            # cross-validation
            for train_index, test_index in kf.split(XX, y):
                # Try to change the data structure from pandas to numpy instead, using .values
                X_train, X_test = XX.values[train_index], XX.values[test_index]
                Y_train, Y_test = y.values[train_index], y.values[test_index]

                # Split into Train and Test Set
                x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
                # Initialize the classifier to be used
                clf = OneVsRestClassifier(LinearSVC(C=c_value, random_state=1))
                # train
                clf.fit(x_train, y_train)
                # y_prediction
                y_pred = clf.predict(x_val)
                y_prob = clf.decision_function(x_val)

                # Predict probabilities for Sklearn LinearSVC
                # a simple subclass of LinearSVC model predicting probabilities by Platt’s scaling.
                # f = np.vectorize(platt_func)
                # raw_predictions = clf.decision_function(x_val)
                # platt_predictions = f(raw_predictions)
                # y_prob = platt_predictions / platt_predictions.sum(axis=1)[:, None]

                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y_val.ravel(), y_prob.ravel())
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                ACC_val, TPR_val, SPE_val, F1_score_val, cnf_matrix = get_eval_para(y_val, y_pred)
                Pairwise_Acc_val = np.mean(pairwise_accuracy(y_val, y_pred))
                ACC_list.append(ACC_val)
                TPR_list.append(TPR_val)
                SPE_list.append(SPE_val)
                F1_score_list.append(F1_score_val)
                Pairwise_Acc_list.append(Pairwise_Acc_val)

                each_label_value = []  # each_label_value[0] represents the first label true value

                # transposition, save each label true value in a list
                for i in range(len(y_val[0])):  # row num
                    t = []
                    for j in range(len(y_val)):
                        t.append(y_val[j][i])
                    each_label_value.append(t)

                # access individual estimators
                # fp = clf.estimators_[0].predict(X_test)  # the first-class prediction
                # sp = clf.estimators_[1].predict(X_test)  # the second-class prediction
                # tp = clf.estimators_[2].predict(X_test)  # the third-class prediction
                lbl_acc_sum = 0
                for i in range(0, label_num, 1):
                    lbl_pred = clf.estimators_[i].predict(x_val)  # the i-th label prediction, each label predict value
                    lbl_acc = accuracy_score(each_label_value[i], lbl_pred)  # each label accuracy in one fold
                    Label_Acc_list[i].append(lbl_acc)
                    lbl_acc_sum = lbl_acc_sum + lbl_acc

                Average_Label_Accuracy_list.append(lbl_acc_sum / label_num)
                # print("label accuracy each fold:" + str(Label_Acc_list))
                # print("average label accuracy each fold:" + str(Average_Label_Accuracy_list))

            # calculate mean auc
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)

            # The average value of cross-validation loop
            mean_ACC = np.mean(ACC_list)
            mean_TPR = np.mean(TPR_list)
            mean_SPE = np.mean(SPE_list)
            mean_F1_score = np.mean(F1_score_list)
            mean_AUC = mean_auc
            mean_Pairwise_Acc = np.mean(Pairwise_Acc_list)
            mean_Average_Label_Accuracy = np.mean(Average_Label_Accuracy_list)
            for i in range(0, label_num, 1):
                # k-fold mean value of label accuracy
                mean_Label_Acc_list.append(np.mean(Label_Acc_list[i]))

            # save best parameters
            if mean_ACC > max_ACC:
                max_ACC = mean_ACC
                max_c_value = c_value
                max_TPR = mean_TPR
                max_SPE = mean_SPE
                max_F1 = mean_F1_score
                max_AUC = mean_AUC
                max_Pairwise_Acc = mean_Pairwise_Acc
                max_Average_Label_Accuracy = mean_Average_Label_Accuracy
                max_Label_Acc_list = mean_Label_Acc_list

        print(organ_choice, 'br', 'feature number:', feature_selected, 'c:', max_c_value,
              'ACC:', max_ACC, 'TPR:', max_TPR, 'SPE', max_SPE, 'F1_score:', max_F1, 'AUC:', max_AUC,
              'max_Pairwise_Acc:', max_Pairwise_Acc, 'Ave_Label_Acc:', max_Average_Label_Accuracy)

        # Writing CSV Files
        # columns=['organ_choice', 'clf', 'feat_num', 'C', 'ACC', 'TPR', 'spe', 'f1', 'auc']
        list_1 = [organ_choice, 'br', feature_selected, max_c_value, max_ACC, max_TPR, max_SPE, max_F1, max_AUC,
                  max_Pairwise_Acc, max_Average_Label_Accuracy] + max_Label_Acc_list
        pd_data = pd.DataFrame(np.array(list_1).reshape(1, 19))
        output_path = "./output/" + organ_choice + "_br_LinearSVC_validation_result.csv"
        pd_data.to_csv(output_path, mode='a+', header=False)

        # If the accuracy is not changing, only the smallest feature subset is retained
        final_ACC.append(max_ACC)
        if max_ACC in mydict_ACC.keys():
            print("key already exists")
        else:
            mydict_ACC[max_ACC] = feature_selected
        if max_ACC in mydict_c.keys():
            print("key already exists")
        else:
            mydict_c[max_ACC] = max_c_value

    final_feature_selected = mydict_ACC[max(final_ACC)]  # max acc and the smallest feature set
    final_c = mydict_c[max(final_ACC)]

    return final_feature_selected, final_c


# ## Use GridSearchCV to Find the Best Model Parameters of the Classifier Chains Algorithm
def cc_feature_selection_and_train(X, y, organ_choice, feat_select):
    # K-Folds cross-validator
    kf = KFold(n_splits=5)

    label_num = y.shape[1]

    # save minimum value from the duplicated/repeated values
    mydict_ACC = {}
    mydict_c = {}
    final_ACC = []

    # feature selection column by column
    for feature_selected in range(1, feat_select+1, 1):
        print("selected feature num:" + str(feature_selected))
        XX = X.iloc[:, :feature_selected]  # feature subsets

        # Parameter tuning. base models: LinearSVC()
        c_value_list = [0.001, 0.005, 0.01, 0.05, 0.1]
        # c_value_list = [1]

        # according to acc, store max scores, other max scores vary with max_ACC
        max_ACC = 0  # Accuracy
        max_SPE = 0  # Specificity
        max_TPR = 0  # Sensitivity
        max_F1 = 0  # F1 Scores
        max_c_value = 0
        max_AUC = 0  # Area Under the ROC Curve
        max_Pairwise_Acc = 0  # pairwise accuracy
        max_Average_Label_Accuracy = 0  # average label accuracy
        max_Label_Acc_list = []  # each label accuracy

        # Using the custom GridSearchCV
        for c_value in c_value_list:
            ACC_list = []
            SPE_list = []
            TPR_list = []
            F1_score_list = []

            Label_Acc_list = [[] for i in range(y.shape[1])]  # expect to access the performance of EACH LABEL
            Average_Label_Accuracy_list = []
            mean_Label_Acc_list = []
            Pairwise_Acc_list = []

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            # cross-validation
            for train_index, test_index in kf.split(XX, y):
                # Try to change the data structure from pandas to numpy instead, using .values
                X_train, X_test = XX.values[train_index], XX.values[test_index]
                Y_train, Y_test = y.values[train_index], y.values[test_index]

                # Split into Train and Test Set
                x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
                # Initialize the classifier to be used
                clf = ClassifierChain(LinearSVC(C=c_value, random_state=1))
                # train
                clf.fit(x_train, y_train)
                # y_prediction
                y_pred = clf.predict(x_val)
                y_prob = clf.decision_function(x_val)

                # Predict probabilities for Sklearn LinearSVC
                # a simple subclass of LinearSVC model predicting probabilities by Platt’s scaling.
                # f = np.vectorize(platt_func)
                # raw_predictions = clf.decision_function(x_val)
                # platt_predictions = f(raw_predictions)
                # y_prob = platt_predictions / platt_predictions.sum(axis=1)[:, None]

                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y_val.ravel(), y_prob.ravel())
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                ACC_val, TPR_val, SPE_val, F1_score_val, cnf_matrix = get_eval_para(y_val, y_pred)
                Pairwise_Acc_val = np.mean(pairwise_accuracy(y_val, y_pred))
                ACC_list.append(ACC_val)
                TPR_list.append(TPR_val)
                SPE_list.append(SPE_val)
                F1_score_list.append(F1_score_val)
                Pairwise_Acc_list.append(Pairwise_Acc_val)

                each_label_value = []  # each_label_value[0] represents the first label true value
                each_estimators_predict = []

                # transposition, save each label true value in a list
                for i in range(len(y_val[0])):  # row num
                    t = []
                    for j in range(len(y_val)):
                        t.append(y_val[j][i])
                    each_label_value.append(t)

                # transposition, save each label predict value in a list
                for i in range(len(y_pred[0])):  # row num
                    t = []
                    for j in range(len(y_pred)):
                        t.append(y_pred[j][i])
                    each_estimators_predict.append(t)

                # access individual estimators
                lbl_acc_sum = 0
                for i in range(0, label_num, 1):
                    lbl_pred = each_estimators_predict[i]  # the i-th label prediction, each label predict value
                    lbl_acc = accuracy_score(each_label_value[i], lbl_pred)  # each label accuracy in one fold
                    Label_Acc_list[i].append(lbl_acc)
                    lbl_acc_sum = lbl_acc_sum + lbl_acc

                Average_Label_Accuracy_list.append(lbl_acc_sum / label_num)
                # print("label accuracy each fold:" + str(Label_Acc_list))
                # print("average label accuracy each fold:" + str(Average_Label_Accuracy_list))

            # calculate mean auc
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)

            # The average value of cross-validation loop
            mean_ACC = np.mean(ACC_list)
            mean_TPR = np.mean(TPR_list)
            mean_SPE = np.mean(SPE_list)
            mean_F1_score = np.mean(F1_score_list)
            mean_AUC = mean_auc
            mean_Pairwise_Acc = np.mean(Pairwise_Acc_list)
            mean_Average_Label_Accuracy = np.mean(Average_Label_Accuracy_list)
            for i in range(0, label_num, 1):
                # k-fold mean value of label accuracy
                mean_Label_Acc_list.append(np.mean(Label_Acc_list[i]))

            # save best parameters
            if mean_ACC > max_ACC:
                max_ACC = mean_ACC
                max_c_value = c_value
                max_TPR = mean_TPR
                max_SPE = mean_SPE
                max_F1 = mean_F1_score
                max_AUC = mean_AUC
                max_Pairwise_Acc = mean_Pairwise_Acc
                max_Average_Label_Accuracy = mean_Average_Label_Accuracy
                max_Label_Acc_list = mean_Label_Acc_list

        print(organ_choice, 'cc', 'feature number:', feature_selected, 'c:', max_c_value,
              'ACC:', max_ACC, 'TPR:', max_TPR, 'SPE', max_SPE, 'F1_score:', max_F1, 'AUC:', max_AUC,
              'max_Pairwise_Acc:', max_Pairwise_Acc, 'Ave_Label_Acc:', max_Average_Label_Accuracy)

        # Writing CSV Files
        # columns=['organ_choice', 'clf', 'feat_num', 'C', 'ACC', 'TPR', 'spe', 'f1', 'auc']
        list_1 = [organ_choice, 'cc', feature_selected, max_c_value, max_ACC, max_TPR, max_SPE, max_F1, max_AUC,
                  max_Pairwise_Acc, max_Average_Label_Accuracy] + max_Label_Acc_list
        pd_data = pd.DataFrame(np.array(list_1).reshape(1, 19))
        output_path = "./output/" + organ_choice + "_cc_LinearSVC_validation_result.csv"
        pd_data.to_csv(output_path, mode='a+', header=False)

        # If the accuracy is not changing, only the smallest feature subset is retained
        final_ACC.append(max_ACC)
        if max_ACC in mydict_ACC.keys():
            print("key already exists")
        else:
            mydict_ACC[max_ACC] = feature_selected
        if max_ACC in mydict_c.keys():
            print("key already exists")
        else:
            mydict_c[max_ACC] = max_c_value

    final_feature_selected = mydict_ACC[max(final_ACC)]  # max acc and the smallest feature set
    final_c = mydict_c[max(final_ACC)]

    return final_feature_selected, final_c


# ## Binary Relevance Algorithm validation set
def br_svm(X, y, organ_choice, feat_select):
    # K-Folds cross-validator
    kf = KFold(n_splits=5)

    # best parameter
    feature_selected, c_value = br_feature_selection_and_train(X, y, organ_choice, feat_select)

    XXX = X.iloc[:, 0:feature_selected]  # feature subsets
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

        clf = OneVsRestClassifier(LinearSVC(C=c_value, random_state=1))
        # train
        clf.fit(X_train, Y_train)
        # y_prediction
        y_pred = clf.predict(X_test)
        y_prob = clf.decision_function(X_test)

        # Predict probabilities for Sklearn LinearSVC
        # a simple subclass of LinearSVC model predicting probabilities by Platt’s scaling.
        # f = np.vectorize(platt_func)
        # raw_predictions = clf.decision_function(X_test)
        # platt_predictions = f(raw_predictions)
        # y_prob = platt_predictions / platt_predictions.sum(axis=1)[:, None]

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

        # transposition, save each label true value in a list
        for i in range(len(Y_test[0])):  # row num
            t = []
            for j in range(len(Y_test)):
                t.append(Y_test[j][i])
            each_label_value.append(t)

        lbl_acc_sum = 0
        for i in range(0, label_num, 1):
            lbl_pred = clf.estimators_[i].predict(X_test)  # the i-th label prediction, each label predict value
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
    print(organ_choice, 'br', 'feature number:', feature_selected, 'c:', c_value,
          'ACC:', np.mean(ACC), 'TPR:', np.mean(TPR), 'SPE', np.mean(SPE), 'F1_score:', np.mean(F1_score),
          'AUC:', mean_auc, 'Pairwise_Acc:', np.mean(Pairwise_Acc), 'Ave_Label_Acc:', np.mean(Average_Label_Accuracy))

    # Writing CSV Files
    columns_name = ['organ_choice', 'clf', 'feat_num', 'C', 'ACC', 'TPR', 'spe', 'f1', 'auc', 'pairAcc', 'aveLabAcc',
                    'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8']
    list_1 = [organ_choice, 'br', feature_selected, c_value, np.mean(ACC), np.mean(TPR), np.mean(SPE),
              np.mean(F1_score), mean_auc, np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    pd_data = pd.DataFrame(np.array(list_1).reshape(1, 19), columns=columns_name)
    output_path = "./output/" + organ_choice + "_br_LinearSVC_test_result.csv"
    pd_data.to_csv(output_path, index=False)


# ## Classifier Chains Algorithm validation set
def cc_svm(X, y, organ_choice, feat_select):
    # K-Folds cross-validator
    kf = KFold(n_splits=5)

    # best parameter
    feature_selected, c_value = cc_feature_selection_and_train(X, y, organ_choice, feat_select)

    XXX = X.iloc[:, 0:feature_selected]  # feature subsets
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

        clf = ClassifierChain(LinearSVC(C=c_value, random_state=1))
        # train
        clf.fit(X_train, Y_train)
        # y_prediction
        y_pred = clf.predict(X_test)
        y_prob = clf.decision_function(X_test)

        # Predict probabilities for Sklearn LinearSVC
        # a simple subclass of LinearSVC model predicting probabilities by Platt’s scaling.
        # f = np.vectorize(platt_func)
        # raw_predictions = clf.decision_function(X_test)
        # platt_predictions = f(raw_predictions)
        # y_prob = platt_predictions / platt_predictions.sum(axis=1)[:, None]

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
    print(organ_choice, 'cc', 'feature number:', feature_selected, 'c:', c_value,
          'ACC:', np.mean(ACC), 'TPR:', np.mean(TPR), 'SPE', np.mean(SPE), 'F1_score:', np.mean(F1_score),
          'AUC:', mean_auc, 'Pairwise_Acc:', np.mean(Pairwise_Acc), 'Ave_Label_Acc:', np.mean(Average_Label_Accuracy))

    # Writing CSV Files
    columns_name = ['organ_choice', 'clf', 'feat_num', 'C', 'ACC', 'TPR', 'spe', 'f1', 'auc', 'pairAcc', 'aveLabAcc',
                    'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8']
    list_1 = [organ_choice, 'cc', feature_selected, c_value, np.mean(ACC), np.mean(TPR), np.mean(SPE),
              np.mean(F1_score), mean_auc, np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    pd_data = pd.DataFrame(np.array(list_1).reshape(1, 19), columns=columns_name)
    output_path = "./output/" + organ_choice + "_cc_LinearSVC_test_result.csv"
    pd_data.to_csv(output_path, index=False)


if __name__ == '__main__':
    """
    __name__ is a built-in variable which evaluates to the name of the current module.
    If you run the module directly, execute the function, if you import the module, don’t run it.
    """
    organ_choice = "kidney"
    dataset = pd.read_csv('dataset/kidney-MLFS-sorted-feature.csv')
    print("dataset.shape: " + str(dataset.shape))

    # split the features-X and class labels-y
    X = dataset.iloc[:, 8:]  # features
    y = dataset.iloc[:, :8].astype('int')  # labels
    feat_select = 200

    print("---------------------------------------------")
    print("X.shape: " + str(X.shape))
    print(X.head())  # print out the first 5 rows
    print("---------------------------------------------")
    print("y.shape: " + str(y.shape))
    print(y.head())
    print("---------------------------------------------")
    print("Descriptive stats:")
    print(X.describe())

    # Normalise the data
    X = (X - X.min()) / (X.max() - X.min())  # min-max normalization

    # feature selection and classification
    br_svm(X, y, organ_choice, feat_select)
    cc_svm(X, y, organ_choice, feat_select)


