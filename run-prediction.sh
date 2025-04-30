#!/bin/bash
/opt/spark/bin/spark-submit \
  --master local[*] \
  --class WineQualityPredictor \
  target/wine-quality-prediction-1.0-SNAPSHOT-jar-with-dependencies.jar \
  data/ValidationDataset.csv
