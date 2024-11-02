import logging
import os
import socket
from io import StringIO

import boto3
import pyspark
import pandas as pd

from dotenv import load_dotenv

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, lag, avg, stddev, expr

# Configure the logging
logging.basicConfig(filename='pyspark_engine.log', level=logging.WARN,
                    format='%(asctime)s - %(levelname)s: %(message)s')

BUCKET_NAME = "data-engineer-assignment-dimamed"


def main():
    def is_possible_cloud() -> bool:
        # Common hostnames/IP ranges for cloud providers could be checked here
        cloud_keywords = ['amazon', 'aws', 'google', 'azure', 'cloud', 'compute', 'ec2', 'gcp', 'cloudapp', 'heroku']
        hostname = socket.gethostname().lower()
        for keyword in cloud_keywords:
            if keyword in hostname:
                return True

        # Check for known cloud-specific environment variables
        cloud_env_vars = ['AWS_EXECUTION_ENV', 'GOOGLE_CLOUD_PROJECT', 'AZURE_FUNCTIONS_ENVIRONMENT']
        for var in cloud_env_vars:
            if os.getenv(var):
                return True

        return False

    def write_to_s3(s3_client, spark_df: pyspark.sql.DataFrame, csv_file_name_full_path: str,
                    bucket_name: str = BUCKET_NAME) -> None:
        try:
            # Convert DataFrame to CSV format in-memory
            csv_buffer = StringIO()
            pandas_df: pd.DataFrame = spark_df.toPandas()
            pandas_df.to_csv(csv_buffer, index=False, header=True)

            s3_client.put_object(
                Bucket=bucket_name,
                Key=csv_file_name_full_path,
                Body=csv_buffer.getvalue()
            )
            logging.info(f"File {csv_file_name_full_path} uploaded successfully to {bucket_name}.")
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise e

    is_running_on_cloud = is_possible_cloud()
    if not is_running_on_cloud:
        logging.info("The script is not running on a cloud environment, loading local .env file")
        load_dotenv()
        # Retrieve AWS credentials from environment variables
        AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        AWS_REGION = os.getenv('AWS_REGION', 'eu-central-1')  # Default to 'us-west-2' if not specified

        # Set up boto3 client for S3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    else:
        logging.info("The script is running on a cloud environment, using IAM role for authentication")
        s3_client = boto3.client('s3')

    spark = ((SparkSession.builder
              .appName('vi-data-engineering-home-assignment')
              .config("spark.driver.host", "localhost"))
             # .config('spark.jars', f"{os.path.join(rootpath.detect(), 'hadoop-bare-naked-local-fs-0.1.0.jar')}")
             # .config("spark.jars.packages", "org.apache.spark:spark-hadoop-cloud_2.13-3.5.3")
             # .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
             # .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.6,"
             #         "com.amazonaws:aws-java-sdk-bundle:2.29.3")
             .master("local[*]")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext._jsc.hadoopConfiguration().set("spark.hadoop.fs.file.impl",
                                                      "org.apache.hadoop.fs.LocalFileSystem")
    # if is_running_on_cloud:
    #     spark._jsc.hadoopConfiguration().set("com.amazonaws.services.s3.enableV4", "true")
    #     spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.amazonaws.com")
    #     spark._jsc.hadoopConfiguration().set("fs.s3a.access.key", os.getenv('AWS_ACCESS_KEY_ID'))
    #     spark._jsc.hadoopConfiguration().set("fs.s3a.secret.key", os.getenv('AWS_SECRET_ACCESS_KEY'))
    #     spark._jsc.hadoopConfiguration().set("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    #     spark._jsc.hadoopConfiguration().set("fs.AbstractFileSystem.s3a.impl", "org.apache.hadoop.fs.s3a.S3A")

    # Load the CSV file
    df_source = spark.read.csv('stocks_data.csv', header=True, inferSchema=True)

    # 1. ** Compute the Average Daily Return of All Stocks **
    # Compute daily returns for each stock
    window_spec = Window.partitionBy('ticker').orderBy('Date')
    df = df_source.withColumn('previous_close', lag('close').over(window_spec))
    df = df.withColumn('daily_return', (col('close') - col('previous_close')) / col('previous_close'))

    # Compute the average daily return across all stocks
    average_daily_return = df.groupBy('Date').agg(avg('daily_return').alias('average_return'))
    average_daily_return.show()

    # 2. ** Which Stock Was Traded with the Highest Worth? **
    # Calculate trading value for each row
    df = df_source.withColumn('trading_value', col('close') * col('volume'))

    # Find the ticker with the highest average trading value
    average_trading_value = df.groupBy('ticker').agg(avg('trading_value').alias('value'))
    highest_worth_stock = average_trading_value.orderBy(col('value').desc()).limit(1)
    highest_worth_stock.show()

    # 3. ** Which Stock Was the Most Volatile? **
    # Calculate standard deviation of daily returns for each ticker
    window_spec = Window.partitionBy('ticker').orderBy('Date')
    df = df_source.withColumn('previous_close', lag('close').over(window_spec))
    df = df.withColumn('daily_return', (col('close') - col('previous_close')) / col('previous_close'))
    volatility = df.groupBy('ticker').agg(
        (stddev('daily_return') * expr('sqrt(252)')).alias('standard_deviation'))

    # Find the most volatile stock
    most_volatile_stock = volatility.orderBy(col('standard_deviation').desc()).limit(1)
    most_volatile_stock.show()

    # 4. ** Top Three 30-Day Return Dates **
    # Calculate 30-day returns
    df = df_source.withColumn('30_day_return',
                              (col('close') - lag(col('close'), 30).over(window_spec)) / lag(col('close'), 30).over(
                                  window_spec))

    # Get top three 30-day return dates
    top_30_day_returns = df.orderBy(col('30_day_return').desc()).select('ticker', 'Date').limit(3)
    top_30_day_returns.show()


    # Save to S3
    write_to_s3(s3_client, average_daily_return, 'result-data/'+'average_daily_return.csv')
    write_to_s3(s3_client, highest_worth_stock, 'result-data/'+'highest_worth_stock.csv')
    write_to_s3(s3_client, most_volatile_stock, 'result-data/'+'most_volatile_stock.csv')
    write_to_s3(s3_client, top_30_day_returns, 'result-data/'+'top_30_day_returns.csv')


if __name__ == "__main__":
    main()
