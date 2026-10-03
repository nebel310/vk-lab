import time
from pyspark.sql import SparkSession, functions as F
from pyspark import StorageLevel

spark = (SparkSession.builder
    .appName("lakehouse-bench")
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate())

spark.sparkContext.setLogLevel("WARN")

jvm = spark._jvm
hconf = spark._jsc.hadoopConfiguration()

def fs_path(p):  return jvm.org.apache.hadoop.fs.Path(p)
def rm(p):
    try:
        path = fs_path(p)
        fs = path.getFileSystem(hconf)
        if fs.exists(path): fs.delete(path, True)
    except Exception as e:
        print("rm err:", e)

def dir_size(p):
    path = fs_path(p)
    fs = path.getFileSystem(hconf)
    cs = fs.getContentSummary(path)
    return cs.getLength(), cs.getFileCount()

# ---------- данные ----------
print("=== Генерация датасета (15M строк) ===")
N = 15_000_000
t0 = time.perf_counter()
df = (spark.range(0, N)
    .withColumn("ss_item_sk",         (F.col("id") % 18000).cast("long"))
    .withColumn("ss_customer_sk",     (F.col("id") % 100000).cast("long"))
    .withColumn("ss_store_sk",        (F.col("id") % 12).cast("long"))
    .withColumn("ss_quantity",        (F.col("id") % 100).cast("int"))
    .withColumn("ss_wholesale_cost",  F.round(F.rand(1) * 100, 2))
    .withColumn("ss_list_price",      F.round(F.rand(2) * 200, 2))
    .withColumn("ss_sales_price",     F.round(F.rand(3) * 150, 2))
    .withColumn("ss_ext_sales_price", F.round(F.rand(4) * 15000, 2))
    .withColumn("ss_net_paid",        F.round(F.rand(5) * 10000, 2))
    .withColumn("ss_net_profit",      F.round(F.rand(6) * 5000, 2))
    .drop("id")
)
df.persist(StorageLevel.MEMORY_AND_DISK)
print(f"rows = {df.count()}, gen = {time.perf_counter()-t0:.1f}s")

# ---------- запись ----------
BENCH = [
    ("parquet", "snappy"),
    ("parquet", "zstd"),
    ("parquet", "gzip"),
    ("orc",     "snappy"),
    ("orc",     "zstd"),
    ("orc",     "zlib"),
]

print("\n=== Запись ===")
write_results = {}
for fmt, codec in BENCH:
    path = f"s3a://warehouse/spark_bench/{fmt}_{codec}"
    rm(path)
    t0 = time.perf_counter()
    (df.write.mode("overwrite")
       .option("compression", codec)
       .format(fmt)
       .save(path))
    wt = time.perf_counter() - t0
    size, files = dir_size(path)
    write_results[(fmt, codec)] = (size, files, wt)
    print(f"  {fmt:7s} {codec:7s}  size={size/1e6:8.2f} MB  files={files}  write={wt:.2f}s")

# ---------- агрегация: 3 прогона ----------
print("\n=== Агрегация ===")
agg_results = {}
for fmt, codec in BENCH:
    path = f"s3a://warehouse/spark_bench/{fmt}_{codec}"
    spark.read.format(fmt).load(path).createOrReplaceTempView(f"t_{fmt}_{codec}")
    sql = f"SELECT ss_store_sk, count(*) AS n, sum(ss_net_paid) AS s FROM t_{fmt}_{codec} GROUP BY ss_store_sk"
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        spark.sql(sql).collect()
        times.append(time.perf_counter() - t0)
    agg_results[(fmt, codec)] = times
    print(f"  {fmt:7s} {codec:7s}  agg_first={times[0]:.3f}s  agg_next={(times[1]+times[2])/2:.3f}s")

# ---------- итог ----------
print("\n=== ИТОГ ===")
print(f"{'format':8s} {'codec':7s} {'size_mb':>9s} {'files':>6s} {'write_s':>8s} {'agg_first':>10s} {'agg_next':>9s}")
print("-" * 62)
for (fmt, codec), (size, files, wt) in write_results.items():
    at = agg_results[(fmt, codec)]
    print(f"{fmt:8s} {codec:7s} {size/1e6:9.2f} {files:6d} {wt:8.2f} {at[0]:10.3f} {(at[1]+at[2])/2:9.3f}")

spark.stop()
print("\ndone.")