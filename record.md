# 最佳成绩记录
## NN
1. https://www.kaggle.com/apiao1/location-eda-8eb410 
   V12 cv:0.012740 lb:.1362 ，11->12 增加Grass特征
   
1. https://www.kaggle.com/apiao1/location-eda-8eb410?scriptVersionId=23289148 
   V11 cv:0.012725 lb:.1362 表现次于1
   
## RF
1. https://www.kaggle.com/apiao1/model-rf?scriptVersionId=23306697
   V2 cv:0.012934 lb:.1372 特征同NN.2
   
## lgbm 
预测多标签方法：
1. 直接用lgbm的api：https://www.kaggle.com/enzoamp/nfl-lightgbm/code
```         
'objective':'multiclass',
"metric": 'multi_logloss',
'num_class': 199,
```
https://www.kaggle.com/apiao1/nfl-001-lightgbm?scriptVersionId=23335929
V5 cv:0.50477 lb:.1386

加maxDepth,https://www.kaggle.com/apiao1/nfl-001-lightgbm?scriptVersionId=23338069
V7 cv:0.50511 lb:.1376

2. 用LGBMRegressor,得到预测值后加函数展开， https://www.kaggle.com/newbielch/lgbm-regression-view
3. LGBMClassifier + 平滑， https://www.kaggle.com/mrkmakr/lgbm-multiple-classifier