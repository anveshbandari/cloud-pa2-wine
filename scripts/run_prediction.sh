#!/bin/bash

# Script to run wine quality prediction with and without Docker

# Usage:
# ./run_prediction.sh <test-data-path> <model-path> [--docker]

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <test-data-path> <model-path> [--docker]"
    exit 1
fi

TEST_DATA_PATH=$1
MODEL_PATH=$2
USE_DOCKER=false

if [ "$#" -eq 3 ] && [ "$3" == "--docker" ]; then
    USE_DOCKER=true
fi

# Set your Docker username
DOCKER_USERNAME="anveshbandari"

if [ "$USE_DOCKER" = true ]; then
    echo "Running prediction using Docker..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Pull the latest image if not already available
    docker pull $DOCKER_USERNAME/wine-quality-prediction
    
    # Create absolute paths for test data and model
    ABS_TEST_DATA_PATH=$(realpath $TEST_DATA_PATH)
    ABS_MODEL_PATH=$(realpath $MODEL_PATH)
    
    # Create a temporary directory to mount
    TEMP_DIR=$(mktemp -d)
    
    # Copy files to the temp directory
    mkdir -p $TEMP_DIR/data
    cp $ABS_TEST_DATA_PATH $TEMP_DIR/data/test.csv
    
    # If model path is a directory, copy the entire directory
    if [ -d "$ABS_MODEL_PATH" ]; then
        mkdir -p $TEMP_DIR/model
        cp -r $ABS_MODEL_PATH/* $TEMP_DIR/model/
    else
        # If model path is a file, copy the file
        cp $ABS_MODEL_PATH $TEMP_DIR/model/
    fi
    
    # Run Docker container with volume mounted
    docker run --rm -v $TEMP_DIR:/data $DOCKER_USERNAME/wine-quality-prediction
    
    # Clean up
    rm -rf $TEMP_DIR
else
    echo "Running prediction without Docker..."
    
    # Check if Spark is installed
    if ! which spark-submit > /dev/null; then
        echo "Spark is not installed or not in PATH. Please install Spark or use Docker option."
        exit 1
    fi
    
    # Run Spark submit directly
    spark-submit --class com.wineml.Predict \
                 --master local[*] \
                 target/wine-quality-prediction-1.0-SNAPSHOT.jar \
                 $MODEL_PATH $TEST_DATA_PATH
fi

echo "Prediction completed!"
