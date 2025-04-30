package com.wineml;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Wine Quality Prediction Model Inference
 * Loads a pre-trained model and runs prediction on test data
 */
public class Predict {
    
    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Usage: Predict <model-path> <test-data-path>");
            System.exit(1);
        }
        
        String modelPath = args[0];
        String testDataPath = args[1];
        
        // Initialize Spark Session
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction")
                .getOrCreate();
        
        // Define the schema for our CSV data
        StructType schema = DataTypes.createStructType(new StructField[] {
                DataTypes.createStructField("fixed_acidity", DataTypes.DoubleType, false),
                DataTypes.createStructField("volatile_acidity", DataTypes.DoubleType, false),
                DataTypes.createStructField("citric_acid", DataTypes.DoubleType, false),
                DataTypes.createStructField("residual_sugar", DataTypes.DoubleType, false),
                DataTypes.createStructField("chlorides", DataTypes.DoubleType, false),
                DataTypes.createStructField("free_sulfur_dioxide", DataTypes.DoubleType, false),
                DataTypes.createStructField("total_sulfur_dioxide", DataTypes.DoubleType, false),
                DataTypes.createStructField("density", DataTypes.DoubleType, false),
                DataTypes.createStructField("pH", DataTypes.DoubleType, false),
                DataTypes.createStructField("sulphates", DataTypes.DoubleType, false),
                DataTypes.createStructField("alcohol", DataTypes.DoubleType, false),
                DataTypes.createStructField("quality", DataTypes.DoubleType, false)
        });
        
        // Load test data
        Dataset<Row> testData = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv(testDataPath);
        
        // Convert quality to integer (1-10 classes)
        testData = testData.withColumn("quality", testData.col("quality").cast(DataTypes.IntegerType));
        
        System.out.println("Test Dataset Count: " + testData.count());
        
        // Load the pre-trained model
        System.out.println("Loading model from: " + modelPath);
        PipelineModel loadedModel = PipelineModel.load(modelPath);
        
        // Make predictions
        System.out.println("Running predictions on test data...");
        Dataset<Row> predictions = loadedModel.transform(testData);
        
        // Create an evaluator to measure F1 score
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        
        // Evaluate the model
        double f1Score = evaluator.evaluate(predictions);
        System.out.println("Test F1 Score: " + f1Score);
        
        // Show some example predictions
        System.out.println("Example predictions:");
        predictions.select("quality", "prediction").show(10);
        
        spark.stop();
    }
}
