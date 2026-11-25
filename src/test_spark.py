import os

from pyspark.sql import SparkSession


python_executable = os.path.join(
    os.getcwd(),
    ".venv",
    "Scripts",
    "python.exe"
)

os.environ["PYSPARK_PYTHON"] = python_executable
os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

print("Python executable:")
print(python_executable)

print("\nStarting Spark...")

spark = (
    SparkSession.builder
    .appName("MarketingMLPOC")
    .master("local[*]")
    .config(
        "spark.pyspark.python",
        python_executable
    )
    .config(
        "spark.pyspark.driver.python",
        python_executable
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("Spark started successfully!")

data = [
    (1, "Test", 100),
    (2, "Test", 200),
    (3, "Test", 300),
]

df = spark.createDataFrame(
    data,
    ["id", "name", "value"]
)

print("\nTest DataFrame:")
df.show()

print(f"Row count: {df.count()}")

spark.stop()

print("\nSpark stopped successfully!")