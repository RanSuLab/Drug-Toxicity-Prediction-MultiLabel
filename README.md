# Att-RethinkNet-Project
A multi-label learning model for predicting drug-induced toxicity in multi-organ based on toxicogenomics data


# Keywords
Drug-induced Toxicity, Multi-label, Prediction, Multi-organ, Gene Expression Data


# Dataset
[Toxygates](https://toxygates.nibiohn.go.jp/toxygates/), [Open TG-GATEs](https://toxico.nibiohn.go.jp/english/), [Pathology items](https://dbarchive.biosciencedbc.jp/en/open-tggates/download.html)


# Requires
R version 3.4.0
Python version 3.6.6

keras==2.4.3
tensorflow==2.2.0
numpy>=1.17
scipy==1.4.1
Cython>=0.29
scikit-learn==0.23.2
setuptools==50.0.3
absl-py==0.10.0
matplotlib==3.2.0
joblib==0.16.0
h5py==2.10.0
pandas==1.1.2
six
liac-arff
arff
tqdm
astor


# Installation
Install packages
```bash
pip install numpy Cython
```

Compile and install the C-extensions
Change Directory in CMD: change to project working directory using the "cd" command
```bash
python ./setup.py install
python ./setup.py build_ext -i
```

Run example in project directory
```bash
python ./examples/classification.py
```


# Orgainzation  
1.	AdvancedML-MLSMOTE 
    Method : Multilabel Synthetic Minority Over-sampling Technique (MLSMOTE)
    * **mlsmote-liver.py** Produce synthetic instances for imbalanced multilabel liver dataset. 
	* **mlsmote-kidney.py** Produce synthetic instances for imbalanced multilabel kidney dataset. 
2.	Drug-feature-sorted-by-score-MLFS 
    * **LoadingData.py** Provide basic operations to loading data.
	* **MultiLabelFStatistic-liver.py** Multi label F-Statistic (MLFS) algorithm in liver dataset.
	* **MultiLabelFStatistic-kidney.py** Multi label F-Statistic (MLFS) algorithm in kidney dataset. 
3.	Drug-multilabel-classifier-comparison
    a. Liver dataset:
	Classifier : Binary relevance (BR), Classifier chains (CC).
    * **drug_featureSelection_mlClassifier_lr_liver.py** Base classifier: Logistic regression (lr). 
	* **drug_featureSelection_mlClassifier_rf_liver.py** Base classifier: Random forest (rf). 
	* **drug_featureSelection_mlClassifier_svm_liver.py** Base classifier: Linear support vector machines (svm). 
    * **drug_FeatureSubsetSelected_roc_curve_liver.py** Selected optimal feature subsets. (optimal subset of relevant features)
	* **integrative-model-jackknife-knn-liver.py** Integrative model.
	b. Kidney dataset:
	Classifier : Binary relevance (BR), Classifier chains (CC).
    * **drug_featureSelection_mlClassifier_lr_kidney.py** Base classifier: Logistic regression (lr). 
	* **drug_featureSelection_mlClassifier_rf_kidney.py** Base classifier: Random forest (rf). 
	* **drug_featureSelection_mlClassifier_svm_kidney.py** Base classifier: Linear support vector machines (svm). 
    * **drug_FeatureSubsetSelected_roc_curve_kidney.py** Selected optimal feature subsets. (optimal subset of relevant features)
	* **integrative-model-jackknife-knn-kidney.py** Integrative model.
4.	AttRethinkNet-Multilabel-Classification
    |-- mlearn
       |-- models
	      |-- rethinknet
		     |-- rethinkNet.py (Att-RethinkNet framework)
    |-- examples (Run models and draw ROC curves, 002 means our proposed Att-RethinkNet, 001 means RethinkNet.)

 
# Use
Download datasets->MLSMOTE->MLFS->Feature selection and classification.


# Contact
Any questions can be directed to 313658560@qq.com