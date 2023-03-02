# Databricks notebook source
import pandas as pd
from pyspark.mllib.evaluation import MulticlassMetrics

# COMMAND ----------

#load train and test data
test_data = spark.table('rashmi.df_with_ga_oct2')
train_data=spark.table('rashmi.df_with_ga_jul1')

# COMMAND ----------

display(train_data)

# COMMAND ----------

#convert to integer datatype
from pyspark.sql.types import IntegerType,FloatType
 
train_data=train_data.withColumn("target",train_data.target.cast(IntegerType()))
test_data=test_data.withColumn("target",test_data.target.cast(IntegerType()))

# COMMAND ----------

#rename target to label
test_data=test_data.withColumnRenamed('target','label')
train_data=train_data.withColumnRenamed('target','label')

# COMMAND ----------

#conversion to integer datatype
train_data=train_data.withColumn("avgnumrenew",train_data.avgnumrenew.cast(IntegerType()))
train_data=train_data.withColumn("avg_nbr_of_issues",train_data.avg_nbr_of_issues.cast(IntegerType()))
train_data=train_data.withColumn("order_amt",train_data.avg_nbr_of_issues.cast(IntegerType()))

# COMMAND ----------

#conversion to integer datatype
test_data=test_data.withColumn("avgnumrenew",test_data.avgnumrenew.cast(IntegerType()))
test_data=test_data.withColumn("avg_nbr_of_issues",test_data.avg_nbr_of_issues.cast(IntegerType()))
test_data=test_data.withColumn("order_amt",test_data.avg_nbr_of_issues.cast(IntegerType()))

# COMMAND ----------

#FEATURE ENGINEERING
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

train_data.dtypes

# COMMAND ----------

featur=["avgnumrenew",'order_amt',"avg_nbr_of_issues",'totalpageviews_lsixm','totals_timeOnSite_lsixm']

# COMMAND ----------

train_data=train_data.select(*['universal_id','avgnumrenew','order_amt','avg_nbr_of_issues','totalpageviews_lsixm','totals_timeOnSite_lsixm','label'])
test_data=test_data.select(*['universal_id','avgnumrenew','order_amt','avg_nbr_of_issues','totalpageviews_lsixm','totals_timeOnSite_lsixm','label'])

# COMMAND ----------

train_data=train_data.fillna(0)
test_data=test_data.fillna(0)

# COMMAND ----------

data1=train_data.filter(train_data.label==0).limit(50000)
data1_test=test_data.filter(test_data.label==0).limit(50000)

# COMMAND ----------

data2=train_data.filter(train_data.label==1).limit(50000)
data2_test=test_data.filter(test_data.label==1).limit(50000)

# COMMAND ----------

dataset=data1.union(data2)
dataset_test=data1_test.union(data2_test)

# COMMAND ----------

assembler=VectorAssembler(inputCols=['avgnumrenew','order_amt','avg_nbr_of_issues','totalpageviews_lsixm','totals_timeOnSite_lsixm'],outputCol='features')


# COMMAND ----------

output=assembler.transform(dataset)

# COMMAND ----------

# MAGIC %md Model: Random Forest Classsifier

# COMMAND ----------

from pyspark.ml.classification import (RandomForestClassifier,DecisionTreeClassifier,GBTClassifier)

# COMMAND ----------

rfc=RandomForestClassifier(labelCol='label',featuresCol='features')

# COMMAND ----------

output.printSchema()

# COMMAND ----------

final_data=output.select('features','label')

# COMMAND ----------

final_data.show()

# COMMAND ----------

# train_data,test_data=final_data.randomSplit([0.7,0.3])

# COMMAND ----------

rfc_model=rfc.fit(final_data)

# COMMAND ----------

rfc_model.featureImportances

# COMMAND ----------

# MAGIC %md Feature Importance

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
ax=sns.barplot(x=featur,y=rfc_model.featureImportances.toArray(),dodge=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC Relationship of feature with target

# COMMAND ----------

#Relation with target
for col in [col.lower() for col in featur]:
    dataset.groupby("label").agg({col: "mean"}).show()

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder

# COMMAND ----------

dtc=DecisionTreeClassifier(labelCol='label',featuresCol='features')
rfc=RandomForestClassifier(numTrees=150,labelCol='label',featuresCol='features')
gbt=GBTClassifier(labelCol='label',featuresCol='features')

# COMMAND ----------

pipeline=Pipeline(stages=[assembler,rfc])

# COMMAND ----------

rfc_model_pip=pipeline.fit(dataset)

# COMMAND ----------

pred=rfc_model_pip.transform(dataset_test)

# COMMAND ----------

selected = pred.select("universal_id", "features", "probability", "prediction")

# COMMAND ----------

display(selected)

# COMMAND ----------

# MAGIC %md 

# COMMAND ----------

# MAGIC %md Hyperparameter tuning

# COMMAND ----------

import random
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
#Hyperparamaeter tuning
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
paramGrid = ParamGridBuilder().addGrid(rfc_model.maxDepth, [2,3,4,5,6,7]).build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2) 
CV_model = crossval.fit(dataset)
prediction = CV_model.transform(dataset_test)
selected = prediction.select("universal_id", 'features', "probability", "prediction")
for row in selected.collect():
    print(row)

# COMMAND ----------

prediction.printSchema()

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
my_binary_eval=BinaryClassificationEvaluator(labelCol='label',rawPredictionCol='prediction')

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pyspark.sql.functions as F
f1_eval=MulticlassClassificationEvaluator(labelCol='label',metricName='f1')
acc_eval=MulticlassClassificationEvaluator(labelCol='label',metricName='accuracy')
prec_eval=MulticlassClassificationEvaluator(labelCol='label',metricName='precisionByLabel')
rec_eval=MulticlassClassificationEvaluator(labelCol='label',metricName='recallByLabel')
rfc_f1=f1_eval.evaluate(prediction)
rfc_acc=acc_eval.evaluate(prediction)
rfc_prec=prec_eval.evaluate(prediction)
rfc_rec=rec_eval.evaluate(prediction)
auc=my_binary_eval.evaluate(prediction)

# COMMAND ----------

print('****RFC Metrics****')
print('AUC:'+str(auc))
print('Recall:'+str(rfc_rec))
print('Precision:'+str(rfc_prec))
print('aCCUracy:'+str(rfc_acc))
print('f1 score:'+str(rfc_f1))

# COMMAND ----------

#on test dataset
import sklearn 
y_true = prediction.select(['label']).collect()
y_pred = prediction.select(['prediction']).collect()

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))

# COMMAND ----------


