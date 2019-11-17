# Code Explation

1. data

    original data from kaggle
    
1. deprecate

    Useless code file, for backup only
    
1. online

    model or some GridSearch online
    
    lgbm:
    
    - model_lgbm_1365: 直接用lgbm的api：https://www.kaggle.com/enzoamp/nfl-lightgbm/code
    
    - model_lgbm_multipleClassifier: LGBMClassifier + 平滑， https://www.kaggle.com/mrkmakr/lgbm-multiple-classifier
   
    - model_lgbm_regression_1412 LGBMClassifier + 平滑,V7 效果不好，cv:0.020105, lb:0.02159 ,猜测哪里有bug,(原文：cv:0.013140205432501861，lb:0.01384)

    NN：
    
    - model_NN_1362
    
    - online_NN:online test use
    
    - online_NN2:online test use
    
    RF:
    - model_RF_1372
    
    GridSearch
    
    - online_cv_lgbm
    
    - online_cv_rf
    
    Model Ensemble
    
    - online_ensemble: simple blend
        - to be optimized
    
    - online_stacking: The second layer predicts the cumlate probability of each category【deprecate:The results are not strictly increasing and do not meet the requirements of the topic】
    
    - online_stacking_cumlate:The second layer predicts the probability of each category, then cumlate all just like single model does
        - didn't perform well, maybe try attention stacking later

 1. reference
 
    Code reproduced from discussion area

1. test

    test only
    
1. model_xgb

    to be finished model
    
1. feature_filter/feature_filter_test
   
   FE use 

   