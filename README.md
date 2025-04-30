# Wine Quality Prediction using Apache Spark

This project implements a wine quality prediction machine learning model using Apache Spark MLlib. The application is designed to train the model in parallel on multiple EC2 instances and perform predictions on a single instance.

## Architecture

- **Parallel Training**: Uses 4 EC2 instances in a Spark cluster
- **Prediction**: Runs on a single EC2 instance
- **Containerization**: Docker container for easy deployment of the prediction application

## Components

1. **WineQualityTrainer.java** - Trains the ML model in parallel using Spark cluster
2. **WineQualityPredictor.java** - Loads the trained model and makes predictions

## Setup Instructions

### 1. Setting up the Spark Cluster

First, set up a Spark cluster with 1 master and 3 worker nodes:

Master node:
cd /opt/spark
./sbin/start-master.sh

Worker nodes (run on each worker):
cd /opt/spark
./sbin/start-worker.sh spark://[MASTER_HOSTNAME]:7077

### 2. Training the Model

Clone the repository and build the application:
git clone https://github.com/anveshbandari/cloud-pa2-wine.git
cd cloud-pa2-wine
mvn clean package

Run the training:
./run-training.sh

### 3. Running Prediction without Docker

./run-prediction.sh data/ValidationDataset.csv

### 4. Running Prediction with Docker

Pull the Docker image:
docker pull anveshbandari/wine-quality-predictor:latest

Run the container:
docker run anveshbandari/wine-quality-predictor

To use a custom test dataset:
docker run -v /path/to/your/data:/app/custom-data anveshbandari/wine-quality-predictor custom-data/TestDataset.csv

## Performance

The model achieves an F1 score of approximately 0.55 on the validation dataset.
![image](https://github.com/user-attachments/assets/a27c3957-aba0-44b5-88ad-617d4638cee0)

## Technologies Used

- Apache Spark 3.5.5
- Spark MLlib
- Java 11
- Docker
- Maven

## Detailed Setup Guide

### EC2 Instance Setup:

1. Launch 4 EC2 instances with Ubuntu
2. Install Java, Scala, and Spark on all instances
3. Configure security groups to allow communication between instances

### Spark Cluster Configuration:

1. On the master node:
   cd /opt/spark
   ./sbin/start-master.sh
   
2. On each worker node:
   cd /opt/spark
   ./sbin/start-worker.sh spark://[MASTER_HOSTNAME]:7077

### Preparing the Environment:

1. Upload the datasets to the master node:
   scp -i your-key.pem TrainingDataset.csv ubuntu@[MASTER_IP]:~/cloud-pa2-wine/data/
   scp -i your-key.pem ValidationDataset.csv ubuntu@[MASTER_IP]:~/cloud-pa2-wine/data/

2. Build the application:
   cd ~/cloud-pa2-wine
   ./build.sh

### Running the Training:

Execute the training script:
./run-training.sh

This will distribute the training across all 4 EC2 instances in the Spark cluster.

### Running Prediction:

Execute the prediction script:
./run-prediction.sh data/ValidationDataset.csv

### Docker Deployment:

1. Build Docker image:
   docker build -t anveshbandari/wine-quality-predictor .

2. Run the Docker container:
   docker run anveshbandari/wine-quality-predictor

## Links

- [GitHub Repository](https://github.com/anveshbandari/cloud-pa2-wine)
- [Docker Hub Image](https://hub.docker.com/r/anveshbandari/wine-quality-predictor)
