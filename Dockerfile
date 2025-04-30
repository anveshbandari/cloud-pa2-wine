FROM openjdk:11-jre-slim

WORKDIR /app

# Copy JAR file and data
COPY target/wine-quality-prediction-1.0-SNAPSHOT-jar-with-dependencies.jar /app/wine-quality-predictor.jar
COPY data /app/data

ENTRYPOINT ["java", "-cp", "/app/wine-quality-predictor.jar", "WineQualityPredictor"]
CMD ["data/ValidationDataset.csv"]
