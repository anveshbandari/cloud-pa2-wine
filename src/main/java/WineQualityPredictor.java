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

/**
 * Wine Quality Prediction Application
 * Simplified version that trains and predicts in one step
 */
public class WineQualityPredictor {

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: WineQualityPredictor <test-file-path>");
            System.exit(1);
        }
        
        String testFilePath = args[0];
        
        // Initialize Spark
        SparkConf conf = new SparkConf()
                .setAppName("Wine Quality Prediction")
                .setMaster("local[*]");  // local mode for prediction
                
        SparkSession spark = SparkSession.builder()
                .config(conf)
                .getOrCreate();
                
        try {
            // Load test data
            Dataset<Row> dataRaw = spark.read()
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .option("delimiter", ";")
                    .csv(testFilePath);
                    
            System.out.println("Data loaded successfully");
            
            // Rename columns to simpler names
            String[] cleanColNames = {"fixed_acidity", "volatile_acidity", "citric_acid", 
                                      "residual_sugar", "chlorides", "free_sulfur_dioxide", 
                                      "total_sulfur_dioxide", "density", "pH", "sulphates", 
                                      "alcohol", "quality"};
            
            for (int i = 0; i < dataRaw.columns().length; i++) {
                dataRaw = dataRaw.withColumnRenamed(dataRaw.columns()[i], cleanColNames[i]);
            }
            
            Dataset<Row> data = dataRaw;
            
            // Split data for training and prediction
            Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3}, 1234L);
            Dataset<Row> trainingData = splits[0];
            Dataset<Row> testData = splits[1];
            
            // Define feature columns (all except quality)
            String[] featureColumns = new String[cleanColNames.length - 1];
            System.arraycopy(cleanColNames, 0, featureColumns, 0, featureColumns.length);
            
            // Assemble features into a vector
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(featureColumns)
                    .setOutputCol("features");
            
            // Create logistic regression model
            LogisticRegression lr = new LogisticRegression()
                    .setMaxIter(10)
                    .setRegParam(0.01)
                    .setElasticNetParam(0.8)
                    .setLabelCol("quality")
                    .setFeaturesCol("features");
            
            // Create pipeline
            Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, lr});
            
            // Train model
            System.out.println("Training model...");
            PipelineModel model = pipeline.fit(trainingData);
            System.out.println("Model trained successfully");
            
            // Make predictions
            Dataset<Row> predictions = model.transform(testData);
            
            // Select example rows to display
            predictions.select("prediction", "quality").show(5);
            
            // Evaluate model
            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("quality")
                    .setPredictionCol("prediction")
                    .setMetricName("f1");
                    
            double f1 = evaluator.evaluate(predictions);
            System.out.println("F1 Score on test data: " + f1);
            
        } catch (Exception e) {
            System.err.println("Error during prediction: " + e.getMessage());
            e.printStackTrace();
        } finally {
            spark.stop();
        }
    }
}
