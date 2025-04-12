# Import required libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import confusion_matrix
import mlflow
import mlflow.spark

# Start Spark session
spark = SparkSession.builder.appName("SparkML_MLflow_Tracking").getOrCreate()

# Load dataset
df = spark.read.csv("churnprediction.csv", header=True, inferSchema=True)

# Preprocessing steps
gender_indexer = StringIndexer(inputCol="gender", outputCol="genderIndex")
label_indexer = StringIndexer(inputCol="churn", outputCol="label")

features = ["genderIndex", "age", "balance", "products", "isActive"]
assembler = VectorAssembler(inputCols=features, outputCol="features")

# Define Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")

# Create pipeline
pipeline = Pipeline(stages=[gender_indexer, label_indexer, assembler, lr])

# Split the data
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Set tracking URI to local
mlflow.set_tracking_uri("http://localhost:5000")  # Default local URI

# Start MLflow run
with mlflow.start_run(run_name="Churn_Metrics_Monitoring"):

    # Log parameters
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("maxIter", 100)
    mlflow.log_param("features_used", ",".join(features))

    # Train model
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)

    # AUC
    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.evaluate(predictions)
    mlflow.log_metric("AUC", auc)

    # Accuracy
    total = predictions.count()
    correct = predictions.filter(predictions.label == predictions.prediction).count()
    accuracy = correct / total
    mlflow.log_metric("accuracy", accuracy)

    # Precision and Recall
    tp = predictions.filter((predictions.label == 1) & (predictions.prediction == 1)).count()
    fp = predictions.filter((predictions.label == 0) & (predictions.prediction == 1)).count()
    fn = predictions.filter((predictions.label == 1) & (predictions.prediction == 0)).count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Log model
    mlflow.spark.log_model(model, "spark-logreg-model")

    # Create and log confusion matrix
    pred_pd = predictions.select("label", "prediction").toPandas()
    cm = confusion_matrix(pred_pd["label"], pred_pd["prediction"])

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    # Save plot
    os.makedirs("artifacts", exist_ok=True)
    plot_path = "artifacts/confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)

    print(f"\nModel Trained and Logged to MLflow!")
    print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
