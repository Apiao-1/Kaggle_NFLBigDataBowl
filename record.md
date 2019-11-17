# 最佳成绩记录
## NN
1. https://www.kaggle.com/apiao1/location-eda-8eb410 
   V12 cv:0.012741 lb:.1362 ，11->12 增加Grass特征
   一样的V4 https://www.kaggle.com/apiao1/model-nn/output?scriptVersionId=23301598
   
1. https://www.kaggle.com/apiao1/location-eda-8eb410?scriptVersionId=23289148 
   V11 cv:0.012725 lb:.1362 表现次于1
   
## RF
1. https://www.kaggle.com/apiao1/model-rf?scriptVersionId=23306697
   V2 cv:0.012934402524660512 lb:.1372 特征同NN.2
   
1. https://www.kaggle.com/apiao1/model-rf/output?scriptVersionId=23400566
   V2 cv:0.01292363832402649 lb:.1373 
   
## lgbm 
预测多标签方法：
1. 直接用lgbm的api：https://www.kaggle.com/enzoamp/nfl-lightgbm/code
```         
'objective':'multiclass',
"metric": 'multi_logloss',
'num_class': 199,
```
https://www.kaggle.com/apiao1/nfl-001-lightgbm?scriptVersionId=23335929
V5 cv:0.50477(指标算错了) lb:.1386

加maxDepth,https://www.kaggle.com/apiao1/model-lgbm?scriptVersionId=23338069
V7 cv:0.013051045682717676 lb:.1376

https://www.kaggle.com/apiao1/model-lgbm?scriptVersionId=23465307
V22 cv:0.013031424252303903 lb:.1374

2. 用LGBMRegressor,得到预测值后加函数展开， https://www.kaggle.com/newbielch/lgbm-regression-view
https://www.kaggle.com/apiao1/model-lgbm-regression/notebook?scriptVersionId=23357454
V4 cv:0.01360 lb:0.01412(不及原文的成绩，原文的cv0.01349,lb0.01401) 应该过拟合很严重，调参应该有较好结果

3. LGBMClassifier + 平滑， https://www.kaggle.com/mrkmakr/lgbm-multiple-classifier
https://www.kaggle.com/apiao1/model-lgbm-multipleclassifier?scriptVersionId=23397574
V7 效果不好，cv:0.020105, lb:0.02159 ,猜测哪里有bug,(原文：cv:0.013140205432501861，lb:0.01384)
