package com.wineml;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Wine Quality Prediction Model Training
 * Trains multiple ML models in parallel on a 4-node Spark cluster
 */
public class Train {
    
    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("Usage: Train <training-data-path> <validation-data-path> <model-output-path>");
            System.exit(1);
        }
        
        String trainingDataPath = args[0];
        String validationDataPath = args[1];
        String modelOutputPath = args[2];
        
        // Initialize Spark Session
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Prediction Training")
                .getOrCreate();
        
        // Log info about the cluster
        System.out.println("Spark UI URL: " + spark.sparkContext().uiWebUrl().get());
        System.out.println("Number of executors: " + spark.sparkContext().statusTracker().getExecutorInfos().length);
        
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
        
        // Load training data
        Dataset<Row> trainingData = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv(trainingDataPath);
        
        // Load validation data
        Dataset<Row> validationData = spark.read()
                .option("header", "true")
                .schema(schema)
                .csv(validationDataPath);
        
        // Print dataset info
        System.out.println("Training Dataset Count: " + trainingData.count());
        System.out.println("Validation Dataset Count: " + validationData.count());
        
        // Convert quality to integer (1-10 classes)
        trainingData = trainingData.withColumn("quality", trainingData.col("quality").cast(DataTypes.IntegerType));
        validationData = validationData.withColumn("quality", validationData.col("quality").cast(DataTypes.IntegerType));
        
        // Feature columns (all columns except quality)
        String[] featureCols = {
                "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", 
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", 
                "density", "pH", "sulphates", "alcohol"
        };
        
        // Assemble features into a vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");
        
        // Scale features
        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(true);
        
        // Label indexer
        StringIndexer labelIndexer = new StringIndexer()
                .setInputCol("quality")
                .setOutputCol("indexedLabel")
                .setHandleInvalid("skip");
        
        // Feature indexer
        VectorIndexer featureIndexer = new VectorIndexer()
                .setInputCol("scaledFeatures")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(10);
        
        // Create an evaluator
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        
        // Initialize scores map
        Map<String, Double> modelScores = new HashMap<>();
        
        // 1. Logistic Regression with hyperparameter tuning
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures")
                .setMaxIter(20)
                .setFamily("multinomial");
        
        Pipeline lrPipeline = new Pipeline().setStages(new PipelineStage[] {
                assembler, scaler, labelIndexer, featureIndexer, lr
        });
        
        // Create parameter grid for cross validation
        ParamMap[] lrParamGrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[] {0.01, 0.1, 0.3})
                .addGrid(lr.elasticNetParam(), new double[] {0.0, 0.3, 0.8})
                .build();
        
        // Create cross validator
        CrossValidator lrCV = new CrossValidator()
                .setEstimator(lrPipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(lrParamGrid)
                .setNumFolds(3)
                .setParallelism(4); // Explicitly set parallelism
        
        // Train Logistic Regression model
        System.out.println("Training Logistic Regression model...");
        CrossValidatorModel lrModel = lrCV.fit(trainingData);
        
        // Get the pipeline model from the cross validator model
        PipelineModel lrPipelineModel = (PipelineModel) lrModel.bestModel();
        
        // Evaluate on validation data
        Dataset<Row> lrPredictions = lrModel.transform(validationData);
        double lrF1Score = evaluator.evaluate(lrPredictions);
        System.out.println("Logistic Regression F1 Score: " + lrF1Score);
        
        modelScores.put("LogisticRegression", lrF1Score);
        
        // 2. Random Forest with hyperparameter tuning
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures");
        
        Pipeline rfPipeline = new Pipeline().setStages(new PipelineStage[] {
                assembler, scaler, labelIndexer, featureIndexer, rf
        });
        
        // Create parameter grid for cross validation
        ParamMap[] rfParamGrid = new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[] {10, 20, 30})
                .addGrid(rf.maxDepth(), new int[] {5, 10, 15})
                .build();
        
        // Create cross validator
        CrossValidator rfCV = new CrossValidator()
                .setEstimator(rfPipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(rfParamGrid)
                .setNumFolds(3)
                .setParallelism(4); // Explicitly set parallelism
        
        // Train Random Forest model
        System.out.println("Training Random Forest model...");
        CrossValidatorModel rfModel = rfCV.fit(trainingData);
        
        // Get the pipeline model from the cross validator model
        PipelineModel rfPipelineModel = (PipelineModel) rfModel.bestModel();
        
        // Evaluate on validation data
        Dataset<Row> rfPredictions = rfModel.transform(validationData);
        double rfF1Score = evaluator.evaluate(rfPredictions);
        System.out.println("Random Forest F1 Score: " + rfF1Score);
        
        modelScores.put("RandomForest", rfF1Score);
        
        // Find the best model
        String bestModel = modelScores.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .get()
                .getKey();
        
        double bestScore = modelScores.get(bestModel);
        System.out.println("Best Model: " + bestModel + " with F1 Score: " + bestScore);
        
        // Save the best model
        System.out.println("Saving best model to: " + modelOutputPath);
        try {
            if (bestModel.equals("LogisticRegression")) {
                lrPipelineModel.write().overwrite().save(modelOutputPath);
            } else {
                rfPipelineModel.write().overwrite().save(modelOutputPath);
            }
            System.out.println("Model saved successfully");
        } catch (IOException e) {
            System.err.println("Error saving model: " + e.getMessage());
            e.printStackTrace();
        }
        
        spark.stop();
    }
}
