Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.

============= RESTART: D:\Sem4\ml\lab\assignments\done\Lab07\A1.py =============
Enter values for hyperparameters (comma separated)
50,100,200
None,10,20
2,5
1,2
3
2
Fitting 2 folds for each of 3 candidates, totalling 6 fits
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   0.9s
[CV] END max_depth=20, min_samples_leaf=2, min_samples_split=5, n_estimators=200; total time=   1.0s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   0.4s
[CV] END max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=100; total time=   0.5s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   1.0s
[CV] END max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200; total time=   0.9s
{'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 20}
0.4857142857142857
>>> 
============= RESTART: D:/Sem4/ml/lab/assignments/done/Lab07/A2.py =============
Select models (comma separated numbers):
1.SVM  2.DecisionTree  3.RandomForest  4.AdaBoost  5.NaiveBayes  6.MLP
1,2,3
          Model  Train Accuracy  Test Accuracy  Precision    Recall  F1 Score
0           SVM           0.638       0.428571   0.416628  0.428571  0.400706
1  DecisionTree           0.998       0.452381   0.446757  0.452381  0.442877
2  RandomForest           0.998       0.461905   0.451127  0.461905  0.445981
>>> 
============= RESTART: D:/Sem4/ml/lab/assignments/done/Lab07/A3.py =============
Select regressors (comma separated numbers):
1.LinearRegression  2.SVR  3.DecisionTree  4.RandomForest  5.AdaBoost  6.MLP
1,4,3
              Model  Train MSE  Test MSE  Train MAE  Test MAE  R2 Score
0  LinearRegression   0.001000  1.677951   0.002000  0.975621 -1.434939
1      RandomForest   0.074434  0.590879   0.229111  0.640280  0.142553
2      DecisionTree   0.001000  1.046131   0.002000  0.722024 -0.518081
>>> 
============= RESTART: D:/Sem4/ml/lab/assignments/done/Lab07/A4.py =============
Select clustering methods (comma separated numbers):
1.Hierarchical  2.DBSCAN
1,2
Enter number of clusters: 3
Enter linkage (ward/complete/average): ward
Enter eps value (e.g., 0.5): 0.5
Enter min_samples: 5
      Algorithm  Param1 Param2  Silhouette Score
0  Hierarchical     3.0   ward          0.041229
1        DBSCAN     0.5      5         -0.063186
>>> 
============= RESTART: D:/Sem4/ml/lab/assignments/done/Lab07/Q1.py =============
Traceback (most recent call last):
  File "D:/Sem4/ml/lab/assignments/done/Lab07/Q1.py", line 3, in <module>
    import shap
ModuleNotFoundError: No module named 'shap'
