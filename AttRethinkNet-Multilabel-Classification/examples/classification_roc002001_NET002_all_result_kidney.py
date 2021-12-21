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
from sklearn.metrics import accuracy_score, f1_score, auc, multilabel_confusion_matrix, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay
from numpy import interp

# debugging:_SymbolicException: Inputs to eager execution function cannot be Keras symbolic tensors
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.random.set_seed(1234)
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE


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


def show_confusion_matrix(Y_test, y_pred, organ, fold, lbl_num):
    # print("Y_test"+str(Y_test))
    # print("y_pred" + str(y_pred))
    fig, axes = plt.subplots(2, 4, figsize=(25, 10))
    axes = axes.ravel()
    label_name = ['Hyaline cast', 'Lymphocyte \n cellular infiltration', 'Basophilic change', 'Cyst',
                  'Dilatation', 'Cystic dilatation', 'Necrosis', 'Regeneration']

    font_size = {'size': '24'}  # Adjust to fit
    tick_size = 21
    for i in range(lbl_num):
        # Use rcParams to change all text in the plot
        plt.rcParams.update({'font.size': tick_size})

        # display_labels=[0, i], display_labels=['False', 'True']
        disp = ConfusionMatrixDisplay(confusion_matrix(Y_test[:, i], y_pred[:, i]), display_labels=[0, 1])

        # cmap='Greens', cmap='YlGnBu', cmap='YlGn', cmap='Blues', cmap='BuPu', cmap='viridis'
        disp.plot(ax=axes[i], values_format='.4g', cmap='YlGn')
        disp.ax_.set_title(label_name[i], fontdict=font_size)  # disp.ax_.set_title(f'class {i}')
        disp.ax_.tick_params(axis='both', labelsize=tick_size)
        disp.ax_.set_xlabel("Predicted label", fontdict=font_size)
        disp.ax_.set_ylabel("True label", fontdict=font_size)

        # only show the last row's text and the first column's text
        if i < 4:
            disp.ax_.set_xlabel('')
        if i % 4 != 0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    cbar1 = fig.colorbar(disp.im_, ax=axes)
    cbar1.ax.tick_params(labelsize=tick_size)  # change the font size of the ticks of my colorbar
    plt.savefig("./" + organ + "-confusion-matrix-fold" + str(fold) +".pdf")
    # plt.show()
    plt.close()
    # reset rcParams, recover matplotlib defaults after setting stylesheet
    plt.rcParams.update(plt.rcParamsDefault)


def show_tsne_lbl_figures(X_data, Y_data, organ, fold, lbl_num, str_name):
    """
       t-SNE is a technique for visualizing high-dimensional data in a low-dimensional space (2- or 3-dimensional).
       Only care about pairwise distances between points. It try to position the points on a plane such that the
       pairwise distances between them would minimize a certain criterion.
       To label the axes, I recommend writing something like "t-SNE dimension 1" and "t-SNE dimension 2".
       Sometimes people write "t-SNE 1" and "t-SNE 2" or some such, which is sloppy.
    """
    # save an image for each label, plot labels in multiple figures
    print('Begining TSNE......')
    tsne = TSNE(n_components=2, perplexity=50, n_iter=2000, init='pca', random_state=1, verbose=1)  # Fit and transform with a TSNE
    X_2d = tsne.fit_transform(X_data)  # Project the data in 2D
    print("Org data dimension is {}. Embedded data dimension is {}".format(X_data.shape[-1], X_2d.shape[-1]))
    print('Finished TSNE......')

    # Visualize the data
    # plt.scatter(X_2d[:, 0], X_2d[:, 1], s=10, c=Y_data[:, 4], edgecolors='none', label="t-SNE")  # no use, just try
    label_name = ['Hyaline cast', 'Lymphocyte cellular infiltration', 'Basophilic change', 'Cyst',
                  'Dilatation', 'Cystic dilatation', 'Necrosis', 'Regeneration']
    list_y = [0, 1]
    target_ids = range(len(list_y))
    colors = '#003399', '#FFB03A'
    label_font = {'size': '25'}
    for lbl in range(lbl_num):
        plt.figure(figsize=(6, 5))
        for i, c, label in zip(target_ids, colors, list_y):
            plt.scatter(X_2d[Y_data[:, lbl] == i, 0], X_2d[Y_data[:, lbl] == i, 1], c=c, label=label, s=10, alpha=0.8)
        plt.title(label_name[lbl], fontdict=label_font)
        plt.legend(loc='lower right', fontsize=14)
        plt.tick_params(labelsize=16)
        plt.savefig("./" + organ + "-tsne-" + label_name[lbl] + "-" + str_name + "-fold" + str(fold) + ".pdf", pad_inches=0)
        # plt.show()
        plt.close()


def show_tsne(X_data, Y_data, organ, fold, lbl_num, str_name):
    # add a subplot to the current figure, plot all the labels in the same figure
    print('Begining TSNE......')
    tsne = TSNE(n_components=2, perplexity=50, n_iter=2000, init='pca', random_state=1, verbose=1)  # Fit and transform with a TSNE
    X_2d = tsne.fit_transform(X_data)  # Project the data in 2D
    print("Org data dimension is {}. Embedded data dimension is {}".format(X_data.shape[-1], X_2d.shape[-1]))
    print('Finished TSNE......')

    # Visualize the data of each label
    fig, axes = plt.subplots(2, 4, figsize=(25, 12))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = axes.ravel()
    label_name = ['Hyaline cast', 'Lymphocyte \n cellular infiltration', 'Basophilic change', 'Cyst',
                  'Dilatation', 'Cystic dilatation', 'Necrosis', 'Regeneration']
    list_y = [0, 1]
    target_ids = range(len(list_y))
    colors = '#003399', '#FFB03A'
    label_font = {'size': '30'}
    for lbl in range(lbl_num):
        for i, c, label in zip(target_ids, colors, list_y):
            axes[lbl].scatter(X_2d[Y_data[:, lbl] == i, 0], X_2d[Y_data[:, lbl] == i, 1],
                              c=c, label=label, s=10, alpha=0.8)
            axes[lbl].set_title(label_name[lbl], fontdict=label_font)
            axes[lbl].legend(loc='lower right', fontsize=21, framealpha=0.6)
            axes[lbl].tick_params(labelsize=23)

    # plt.tight_layout()
    plt.savefig("./" + organ + "-subplot-tsne-" + str_name + "-fold" + str(fold) + ".pdf", pad_inches=0)
    # plt.show()
    plt.close()


def show_attention(attention_vector_final, organ, fold):
    # plot part. ylim = (0,1)
    df = pd.DataFrame(attention_vector_final, columns=['attention (%)'])
    # df.plot(kind='bar', rot=0, title='Attention Mechanism as a function of input dimensions.')
    df.plot(kind='bar', rot=0)
    plt.xlabel('Input dimension', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    # plt.legend(loc='upper right')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.legend(loc='lower right', bbox_to_anchor=(1, 1), fontsize=14)
    plt.savefig("./" + organ + "-attention-mechanism-fold" + str(fold) + ".pdf")
    # plt.show()
    plt.close()


def rethinkNet_example_001(X, Y, organ_choice):
    # K-Folds cross-validator
    kf = KFold(n_splits=5)

    # l2 parameter
    param = 10 ** -5

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
                         l2w=param, architecture="arch_001")
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
                    'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8']
    list_1 = [organ_choice, np.mean(ACC), np.mean(TPR), np.mean(SPE), np.mean(F1_score), mean_auc,
              np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    pd_data = pd.DataFrame(np.array(list_1).reshape(1, 16), columns=columns_name)
    output_path = "./" + organ_choice + "_NET001_result.csv"
    pd_data.to_csv(output_path, index=False)

    # Plot ROC curve with matplotlib (only plot arch001)
    plt.figure("arch001")
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)  # Draw Diagonal Lines
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='mediumblue', label=r'RethinkNet(Area=%0.4f)' % mean_auc, lw=1, alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("./" + organ_choice + "-NET001-roc-curve.pdf")
    # plt.show()
    plt.close("arch001")

    # Plot ROC curve with matplotlib (superimpose the two plots)
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)  # Draw Diagonal Lines
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='mediumblue', label=r'RethinkNet(Area=%0.4f)' % mean_auc, lw=1, alpha=.8)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc='lower right')
    # plt.savefig("./" + organ_choice + "-NET001-roc-curve.pdf")
    # plt.show()

    # # save tpr and fpr
    # pd_fpr = pd.DataFrame(mean_fpr)
    # pd_fpr.to_csv("./" + organ_choice + "-NET001-mean-fpr.txt", index=False, header=False, float_format='%.100f')
    # pd_tpr = pd.DataFrame(mean_tpr)
    # pd_tpr.to_csv("./" + organ_choice + "-NET001-mean-tpr.txt", index=False, header=False, float_format='%.100f')


def rethinkNet_example_002(X, Y, organ_choice):
    # K-Folds cross-validator
    kf = KFold(n_splits=5)

    # l2 parameter
    param = 10 ** -5

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
        print(str(fold) + " fold:")
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

        # -----show tsne subplot-----
        show_tsne(X_test, Y_test, organ_choice, fold, label_num, 'original')
        features = clf.model_get_output_features(X_test)
        show_tsne(features, Y_test, organ_choice, fold, label_num, 'NET002')

        # -----show tsne multiple figures-----
        show_tsne_lbl_figures(X_test, Y_test, organ_choice, fold, label_num, 'original')
        features = clf.model_get_output_features(X_test)
        show_tsne_lbl_figures(features, Y_test, organ_choice, fold, label_num, 'NET002')

        # -----show attention-----
        att_vector_final = clf.get_att(X_test)
        show_attention(att_vector_final, organ_choice, fold)

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
        # -----show confusion matrix-----
        show_confusion_matrix(Y_test, y_pred, organ_choice, fold, label_num)

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
                    'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8']
    list_1 = [organ_choice, np.mean(ACC), np.mean(TPR), np.mean(SPE), np.mean(F1_score), mean_auc,
              np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    pd_data = pd.DataFrame(np.array(list_1).reshape(1, 16), columns=columns_name)
    output_path = "./" + organ_choice + "_NET002_result.csv"
    pd_data.to_csv(output_path, index=False)

    # Plot ROC curve with matplotlib
    plt.figure()
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


def main():
    organ_choice = "kidney"

    dataset = pd.read_csv('./data/kidney mlsmote result.csv')
    print("dataset.shape: " + str(dataset.shape))
    # split the features-X and class labels-y
    X = dataset.iloc[:, 8:]  # features
    Y = dataset.iloc[:, :8]  # labels
    # Normalise the data
    X = (X - X.min()) / (X.max() - X.min())  # min-max normalization, X are mapped to the range 0 to 1.
    X = X.values
    Y = Y.values

    print("Running Att-RethinkNet ...")
    rethinkNet_example_002(X, Y, organ_choice)

    print("Running raw RethinkNet ...")
    rethinkNet_example_001(X, Y, organ_choice)

    # Plot them on canvas and give a name to x-axis and y-axis
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("./" + organ_choice + "-001002-roc-curve.pdf")
    plt.show()


if __name__ == '__main__':
    main()

