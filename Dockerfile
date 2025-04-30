# Use a base image with Java and Spark pre-installed
FROM bitnami/spark:3.1.2

# Set environment variables
ENV SPARK_HOME=/opt/bitnami/spark
ENV PATH=$PATH:$SPARK_HOME/bin

USER root

# Install AWS CLI and other dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install awscli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create directories for application
RUN mkdir -p /app/data /app/model

# Copy the JAR file
COPY target/wine-quality-prediction-1.0-SNAPSHOT.jar /app/wine-quality-prediction.jar

# Set working directory
WORKDIR /app

# Configure Spark to use local
ENV MASTER=local[*]

# Create a volume for input/output data
VOLUME ["/data"]

# Set default command
ENTRYPOINT ["spark-submit", "--class", "com.wineml.Predict", "/app/wine-quality-prediction.jar"]

# Default arguments (can be overridden)
CMD ["/data/model", "/data/test.csv"]
