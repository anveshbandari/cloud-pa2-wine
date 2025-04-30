#!/bin/bash

# Script to set up EMR cluster with 4 EC2 instances for wine quality prediction

# Set your AWS region
AWS_REGION="us-east-1"

# Set S3 bucket for logs and data
S3_BUCKET="s3://wine-quality-prediction-bucket"

# Set EC2 key name - make sure this key pair exists in your AWS account
EC2_KEY_NAME="wine-prediction-key"

# Create S3 bucket if it doesn't exist
aws s3 mb $S3_BUCKET --region $AWS_REGION

# Upload datasets to S3
aws s3 cp temp/TrainingDataset.csv $S3_BUCKET/data/
aws s3 cp temp/ValidationDataset.csv $S3_BUCKET/data/

# Upload the JAR file to S3
aws s3 cp target/wine-quality-prediction-1.0-SNAPSHOT.jar $S3_BUCKET/jars/

# Create EMR cluster with 4 instances (1 master + 3 core)
CLUSTER_ID=$(aws emr create-cluster \
    --name "Wine Quality Prediction Cluster" \
    --release-label emr-6.4.0 \
    --applications Name=Spark \
    --log-uri "$S3_BUCKET/logs/" \
    --ec2-attributes KeyName=$EC2_KEY_NAME \
    --instance-type m5.xlarge \
    --instance-count 4 \
    --use-default-roles \
    --region $AWS_REGION \
    --bootstrap-actions Path="s3://aws-bigdata-blog/artifacts/aws-blog-emr-jupyter/install-python-libraries.sh" \
    --configurations file://emr-config.json \
    --output text \
    --query 'ClusterId')

echo "EMR Cluster created with ID: $CLUSTER_ID"
echo "Waiting for cluster to be ready..."

# Wait for cluster to be ready
aws emr wait cluster-running --cluster-id $CLUSTER_ID --region $AWS_REGION

echo "Cluster is ready. Submitting training job..."

# Submit Spark job for training
aws emr add-steps \
    --cluster-id $CLUSTER_ID \
    --steps Type=Spark,Name="Wine Quality Training",ActionOnFailure=CONTINUE,Args=[--class,com.wineml.Train,$S3_BUCKET/jars/wine-quality-prediction-1.0-SNAPSHOT.jar,$S3_BUCKET/data/TrainingDataset.csv,$S3_BUCKET/data/ValidationDataset.csv,$S3_BUCKET/model/] \
    --region $AWS_REGION

echo "Training job submitted. Check EMR console for status."
echo "After training completes, you can run the prediction job."

# Instruction to download the model for local prediction (run this after training completes)
echo "To download the trained model for local prediction:"
echo "aws s3 cp $S3_BUCKET/model/ ./model/ --recursive"
