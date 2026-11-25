import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession


print("=" * 60)
print("SPARK PYTHON WORKER TEST")
print("=" * 60)

print(f"Python: {sys.executable}")

spark = (
    SparkSession.builder
    .appName("SparkPythonWorkerTest")
    .master("local[*]")
    .config(
        "spark.pyspark.python",
        sys.executable
    )
    .config(
        "spark.pyspark.driver.python",
        sys.executable
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:

    df = spark.range(100)

    result = (
        df.rdd
        .map(lambda x: x.id * 2)
        .sum()
    )

    print("\nPython worker test result:", result)

    if result != 9900:
        raise RuntimeError(
            f"Unexpected result: {result}"
        )

    print("\nSUCCESS: Spark Python workers are working.")

finally:

    spark.stop()