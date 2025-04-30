# CS 643 - Cloud Computing
# Programming Assignment 2 - Wine Quality Prediction

## Links
- GitHub Repository: https://github.com/anveshbandari/cloud-pa2-wine
- Docker Hub Image: https://hub.docker.com/r/anveshbandari/wine-quality-predictor

## Setup Instructions

### 1. Setting up the Spark Cluster (Parallel Training)

1. Launch 4 EC2 instances with Ubuntu
2. Install Java, Scala, and Spark on each instance
3. Designate one instance as the master and start the Spark master:
   cd /opt/spark
   ./sbin/start-master.sh
   
4. On the remaining 3 instances, start Spark workers:
   cd /opt/spark
   ./sbin/start-worker.sh spark://[MASTER_HOSTNAME]:7077
   
5. Confirm all workers are connected using the Spark master web UI (port 8080)

### 2. Training the Model in Parallel

1. Clone the repository on the master node:
   git clone https://github.com/anveshbandari/cloud-pa2-wine.git
   cd cloud-pa2-wine
   
2. Create a data directory and upload datasets:
   mkdir -p data
   
3. Upload TrainingDataset.csv and ValidationDataset.csv to the data directory

4. Build the application:
   ./build.sh
   
5. Run the training:
   ./run-training.sh
   
   This will train the model in parallel across all 4 instances.

### 3. Running Prediction (Single Instance)

#### Without Docker:
1. Clone the repository on a single EC2 instance:
   git clone https://github.com/anveshbandari/cloud-pa2-wine.git
   cd cloud-pa2-wine
   
2. Build the application:
   ./build.sh
   
3. Run the prediction application:
   ./run-prediction.sh data/ValidationDataset.csv

#### With Docker:
1. Pull the Docker image:
   docker pull anveshbandari/wine-quality-predictor:latest
   
2. Run the Docker container:
   docker run anveshbandari/wine-quality-predictor
   
3. To use a custom test dataset:
   docker run -v /path/to/your/data:/app/custom-data anveshbandari/wine-quality-predictor custom-data/TestDataset.csv

## Implementation Details

The implementation uses Spark MLlib's LogisticRegression classifier to predict wine quality based on the given features. The wine quality values (1-10) are treated as classes in the model.

For simplicity, the prediction application was designed to train and predict in a single step, making it suitable for containerization without needing to transfer model files between training and prediction.

The F1 score achieved on the validation dataset is approximately 0.55.
