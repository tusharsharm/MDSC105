from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark
import os

# Start Spark session
spark = SparkSession.builder.appName("MLflowAdvancedSparkML").getOrCreate()

# Load data
data = spark.read.csv("churnprediction.csv", header=True, inferSchema=True)

# ML pipeline stages
gender_indexer = StringIndexer(inputCol="gender", outputCol="genderIndex")
label_indexer = StringIndexer(inputCol="churn", outputCol="label")

feature_cols = ["genderIndex", "age", "balance", "products", "isActive"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Logistic Regression with params
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.01)

pipeline = Pipeline(stages=[gender_indexer, label_indexer, assembler, lr])

# Train/Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# MLflow tracking
with mlflow.start_run(run_name="SparkML-LogReg-Churn"):

    # Log parameters
    mlflow.log_param("maxIter", 10)
    mlflow.log_param("regParam", 0.01)
    mlflow.set_tag("project", "Churn Prediction")
    mlflow.set_tag("framework", "Spark ML")

    # Fit model
    model = pipeline.fit(train_data)

    # Evaluate
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator()

    auc = evaluator.evaluate(predictions)
    mlflow.log_metric("AUC", auc)

    accuracy = predictions.filter(predictions.label == predictions.prediction).count() / predictions.count()
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.spark.log_model(model, "spark-ml-model")

    # Log a sample prediction output as artifact
    output_path = "/tmp/predictions.csv"
    predictions.select("customerID", "probability", "prediction", "label") \
        .limit(10).toPandas().to_csv(output_path, index=False)
    mlflow.log_artifact(output_path)

    print(f"Run complete with AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

spark.stop()
