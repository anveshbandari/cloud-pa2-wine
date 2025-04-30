#!/bin/bash
mkdir -p model
/opt/spark/bin/spark-submit \
  --master spark://ip-172-31-40-35.ec2.internal:7077 \
  --class WineQualityTrainer \
  --deploy-mode client \
  --conf spark.ui.showConsoleProgress=false \
  --conf spark.log.level=ERROR \
  target/wine-quality-prediction-1.0-SNAPSHOT-jar-with-dependencies.jar 2>&1 | grep -v "INFO"
