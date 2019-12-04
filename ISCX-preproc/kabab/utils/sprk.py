from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext


def get_spark_session():
    # load Spark session
    spark = SparkSession.builder.master("local[64]").appName("PySparkShell").getOrCreate()
    conf = SparkConf().setAppName("PySparkShell").setMaster("local[64]")
    sc = SparkContext.getOrCreate(conf)
    sqlContext = SQLContext(sc)
    return spark, sc, sqlContext


def read_csv(spark, infile):
    return spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(infile)


def write_csv(df, outfile):
    df.write.csv(outfile, header=True)


