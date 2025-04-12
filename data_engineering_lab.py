from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark
import pandas as pd

# 1. Load and Explore the Dataset
spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()
data = spark.read.csv("churn_lab.csv", header=True, inferSchema=True)
data.printSchema()  # Explore schema
data.show(5)      # Explore first 5 rows

# 2. Preprocess the Data
categorical_cols = [col for col, dtype in data.dtypes if dtype == "string" and col != 'churn']
if 'churn_index' not in data.columns:
    categorical_cols.append('churn')
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_index").fit(data) for col in categorical_cols]
else:
    indexers = []  
feature_cols = [col + "_index" if col in categorical_cols else col for col in data.columns if col != "churn"] 
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 3. Split Data
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# 4. Build Logistic Regression Model
lr = LogisticRegression(featuresCol="features", labelCol="churn_index", maxIter=10, regParam=0.01)

# 5. Use MLflow for Tracking
with mlflow.start_run(run_name="Churn Prediction Run") as run:
    pipeline = Pipeline(stages=indexers + [assembler, lr])
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)

    # Log parameters and metrics
    mlflow.log_param("maxIter", lr.getMaxIter())
    mlflow.log_param("regParam", lr.getRegParam())
    evaluator = BinaryClassificationEvaluator(labelCol="churn_index", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    mlflow.log_metric("AUC", auc)  
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="churn_index", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_acc.evaluate(predictions)
    mlflow.log_metric("accuracy", accuracy)  

    # Log model and artifacts
    mlflow.spark.log_model(model, "spark-model")
    
    # 6. Evaluate and Save Predictions 
    prediction_sample = predictions.select("Churn", "prediction").limit(10).toPandas()
    prediction_sample.to_csv("prediction_sample.csv", index=False)
    mlflow.log_artifact("prediction_sample.csv")

    # Set custom tags
    mlflow.set_tag("lab", "SparkML")
    mlflow.set_tag("model_type", "LogReg")

    # 7. Display MLflow Run URL and Model Performance
    print(f"MLflow Run URL: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
    print(f"Model Performance - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    