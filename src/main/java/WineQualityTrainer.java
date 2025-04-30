import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.LogisticRegression;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Wine Quality Prediction Model Trainer
 * Trains a model using Spark MLlib on 4 EC2 instances
 */
public class WineQualityTrainer {

    public static void main(String[] args) {
        // Initialize Spark
        SparkConf conf = new SparkConf()
                .setAppName("Wine Quality Prediction - Training")
                .setMaster("spark://ip-172-31-40-35.ec2.internal:7077");
        
        SparkSession spark = SparkSession.builder()
                .config(conf)
                .getOrCreate();

        try {
            // Load training dataset
            Dataset<Row> trainingDataRaw = spark.read()
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .option("delimiter", ";")
                    .csv("data/TrainingDataset.csv");
            
            System.out.println("Training dataset loaded successfully");
            
            // Rename columns to simpler names
            String[] cleanColNames = {"fixed_acidity", "volatile_acidity", "citric_acid", 
                                      "residual_sugar", "chlorides", "free_sulfur_dioxide", 
                                      "total_sulfur_dioxide", "density", "pH", "sulphates", 
                                      "alcohol", "quality"};
            
            for (int i = 0; i < trainingDataRaw.columns().length; i++) {
                trainingDataRaw = trainingDataRaw.withColumnRenamed(trainingDataRaw.columns()[i], cleanColNames[i]);
            }
            
            Dataset<Row> trainingData = trainingDataRaw;
            
            // Load validation dataset
            Dataset<Row> validationDataRaw = spark.read()
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .option("delimiter", ";")
                    .csv("data/ValidationDataset.csv");
                    
            System.out.println("Validation dataset loaded successfully");
            
            // Rename columns to match the training data
            for (int i = 0; i < validationDataRaw.columns().length; i++) {
                validationDataRaw = validationDataRaw.withColumnRenamed(validationDataRaw.columns()[i], cleanColNames[i]);
            }
            
            Dataset<Row> validationData = validationDataRaw;
            
            // Print schema after renaming
            System.out.println("Training data schema after renaming:");
            trainingData.printSchema();
            
            // Define feature columns (all except quality)
            String[] featureColumns = Arrays.copyOf(cleanColNames, cleanColNames.length - 1);
            String labelColumn = "quality";
            
            System.out.println("Features: " + Arrays.toString(featureColumns));
            System.out.println("Label: " + labelColumn);
            
            // Assemble features into a vector
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureColumns)
                    .setOutputCol("features");
            
            // Create logistic regression model
            LogisticRegression lr = new LogisticRegression()
                    .setMaxIter(10)
                    .setRegParam(0.01)
                    .setElasticNetParam(0.8)
                    .setLabelCol(labelColumn)
                    .setFeaturesCol("features");
            
            // Create pipeline
            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, lr});
            
            // Train model
            System.out.println("Training model...");
            PipelineModel model = pipeline.fit(trainingData);
            System.out.println("Model trained successfully");
            
            // Make predictions on validation dataset
            Dataset<Row> predictions = model.transform(validationData);
            
            // Select prediction and label columns
            predictions.select("prediction", labelColumn).show(10);
            
            // Evaluate model
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol(labelColumn)
                    .setPredictionCol("prediction")
                    .setMetricName("f1");
                    
            double f1 = evaluator.evaluate(predictions);
            System.out.println("F1 Score on validation data: " + f1);
            
            // Save model
            System.out.println("Saving model...");
            model.write().overwrite().save("model/wine-quality-model");
            System.out.println("Model saved successfully");
            
        } catch (Exception e) {
            System.err.println("Error during model training: " + e.getMessage());
            e.printStackTrace();
        } finally {
            spark.stop();
        }
    }
}
