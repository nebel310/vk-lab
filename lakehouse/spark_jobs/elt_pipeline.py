"""
ELT-пайплайн для локального Lakehouse.
Bronze (сырьё) -> Silver (очистка/нормализация/обогащение) -> Gold (витрина).

Стек: Spark 3.5 + MinIO (S3A).
Данные: синтетические продажи магазинов, TPC-DS-подобная схема store_sales.
"""
import time
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark import StorageLevel

# =========================
# Инициализация Spark
# =========================
spark = (SparkSession.builder
    .appName("lakehouse-elt")
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

def fs_path(p): return jvm.org.apache.hadoop.fs.Path(p)

def rm(p):
    try:
        path = fs_path(p)
        fs = path.getFileSystem(hconf)
        if fs.exists(path): fs.delete(path, True)
    except Exception as e:
        print("rm err:", e)

def dir_info(p):
    path = fs_path(p)
    fs = path.getFileSystem(hconf)
    cs = fs.getContentSummary(path)
    return cs.getLength(), cs.getFileCount()

def log_section(name):
    print("\n" + "=" * 60)
    print(f"  {name}")
    print("=" * 60)

# =========================
# BRONZE — сырые данные с мусором
# =========================
log_section("BRONZE: генерация сырых продаж (с мусором)")

N = 10_000_000
rng = 42

raw = (spark.range(0, N)
    .withColumn("sale_id", F.col("id").cast("long"))
    .withColumn("store_id", (F.col("id") % 12 + 1).cast("int"))           # 1..12
    .withColumn("item_id",  (F.col("id") % 18000 + 1).cast("int"))
    .withColumn("quantity", ((F.col("id") % 20) - 2).cast("int"))          # может быть отрицательным или 0
    .withColumn("unit_price", F.round(F.rand(1) * 150, 2))
    .withColumn("discount_pct", F.round(F.rand(2) * 0.3, 3))
    .withColumn("sale_ts", F.to_timestamp(
        F.from_unixtime(F.lit(1704067200) + F.col("id") * 15)))
    .withColumn("customer_email", F.when(F.rand(3) < 0.05, F.lit(None))
                                    .otherwise(F.concat(F.lit("user"), F.col("id"), F.lit("@example.com"))))
    .drop("id")
)

# подмешиваем дубликаты (~1%)
dupes = raw.sample(False, 0.01, seed=7)
bronze = raw.unionByName(dupes)
bronze.persist(StorageLevel.MEMORY_AND_DISK)

print(f"строк в bronze (с дублями и мусором): {bronze.count()}")
bronze.printSchema()

t0 = time.perf_counter()
rm("s3a://warehouse/elt/bronze/sales")
(bronze.write.mode("overwrite").format("parquet")
    .option("compression", "snappy")
    .save("s3a://warehouse/elt/bronze/sales"))
size, files = dir_info("s3a://warehouse/elt/bronze/sales")
print(f"BRONZE записан: {size/1e6:.1f} MB, files={files}, write={time.perf_counter()-t0:.1f}s")

# справочник магазинов
stores = spark.createDataFrame([
    (1,  "Магазин на Тверской",     "Москва",            "Центр"),
    (2,  "Магазин на Невском",      "Санкт-Петербург",   "Северо-Запад"),
    (3,  "Магазин в Казани",        "Казань",            "Поволжье"),
    (4,  "Магазин в Екатеринбурге", "Екатеринбург",      "Урал"),
    (5,  "Магазин в Новосибирске",  "Новосибирск",       "Сибирь"),
    (6,  "Магазин в Сочи",          "Сочи",              "Юг"),
    (7,  "Магазин в Краснодаре",    "Краснодар",         "Юг"),
    (8,  "Магазин во Владивостоке", "Владивосток",       "Дальний Восток"),
    (9,  "Магазин в Самаре",        "Самара",            "Поволжье"),
    (10, "Магазин в Ростове",       "Ростов-на-Дону",    "Юг"),
    (11, "Магазин в Уфе",           "Уфа",               "Урал"),
    (12, "Магазин в Перми",         "Пермь",             "Урал"),
], schema="store_id int, store_name string, city string, region string")

rm("s3a://warehouse/elt/bronze/stores")
stores.write.mode("overwrite").format("parquet").save("s3a://warehouse/elt/bronze/stores")
print("BRONZE stores записан")

# =========================
# SILVER — очистка, нормализация, обогащение
# =========================
log_section("SILVER: очистка + нормализация + обогащение")

t0 = time.perf_counter()

bronze_df = spark.read.format("parquet").load("s3a://warehouse/elt/bronze/sales")
stores_df = spark.read.format("parquet").load("s3a://warehouse/elt/bronze/stores")

clean = (bronze_df
    # 1. дедупликация по sale_id
    .dropDuplicates(["sale_id"])
    # 2. фильтр мусора: quantity > 0, unit_price > 0, email не NULL
    .filter(F.col("quantity") > 0)
    .filter(F.col("unit_price") > 0)
    .filter(F.col("customer_email").isNotNull())
    # 3. нормализация
    .withColumn("sale_ts",  F.to_timestamp("sale_ts"))
    .withColumn("sale_date", F.to_date("sale_ts"))
    .withColumn("customer_email", F.lower(F.trim("customer_email")))
    # 4. расчётные поля
    .withColumn("gross_amount", F.round(F.col("quantity") * F.col("unit_price"), 2))
    .withColumn("discount_amount", F.round(F.col("gross_amount") * F.col("discount_pct"), 2))
    .withColumn("net_amount", F.round(F.col("gross_amount") - F.col("discount_amount"), 2))
)

# 5. обогащение join'ом со справочником
silver = (clean
    .join(stores_df, on="store_id", how="inner")
    .select("sale_id", "store_id", "store_name", "city", "region",
            "item_id", "quantity", "unit_price", "discount_pct",
            "gross_amount", "discount_amount", "net_amount",
            "sale_ts", "sale_date", "customer_email")
)

silver.persist(StorageLevel.MEMORY_AND_DISK)
print(f"строк в silver (после очистки): {silver.count()}")

rm("s3a://warehouse/elt/silver/sales")
(silver.write.mode("overwrite").format("orc")
    .option("compression", "zstd")
    .save("s3a://warehouse/elt/silver/sales"))
size, files = dir_info("s3a://warehouse/elt/silver/sales")
print(f"SILVER записан: {size/1e6:.1f} MB, files={files}, write={time.perf_counter()-t0:.1f}s")

# =========================
# GOLD — витрина для BI
# =========================
log_section("GOLD: агрегированные витрины")

t0 = time.perf_counter()

sales_by_store_cat = (silver
    .groupBy("region", "city", "store_name")
    .agg(
        F.count("*").alias("sales_count"),
        F.sum("quantity").alias("total_quantity"),
        F.round(F.sum("net_amount"), 2).alias("total_revenue"),
        F.round(F.avg("net_amount"), 2).alias("avg_order_value"),
        F.countDistinct("customer_email").alias("unique_customers"),
    )
    .orderBy(F.desc("total_revenue"))
)

rm("s3a://warehouse/elt/gold/sales_by_store")
sales_by_store_cat.write.mode("overwrite").format("parquet") \
    .option("compression", "zstd") \
    .save("s3a://warehouse/elt/gold/sales_by_store")

daily_revenue = (silver
    .groupBy("sale_date")
    .agg(
        F.count("*").alias("orders"),
        F.round(F.sum("net_amount"), 2).alias("revenue"),
        F.round(F.avg("discount_pct"), 3).alias("avg_discount"),
    )
    .orderBy("sale_date")
)

rm("s3a://warehouse/elt/gold/daily_revenue")
daily_revenue.write.mode("overwrite").format("parquet") \
    .option("compression", "zstd") \
    .save("s3a://warehouse/elt/gold/daily_revenue")

size1, files1 = dir_info("s3a://warehouse/elt/gold/sales_by_store")
size2, files2 = dir_info("s3a://warehouse/elt/gold/daily_revenue")
print(f"GOLD sales_by_store: {size1/1e6:.2f} MB, files={files1}")
print(f"GOLD daily_revenue:  {size2/1e6:.2f} MB, files={files2}")
print(f"GOLD записан за {time.perf_counter()-t0:.1f}s")

# =========================
# Проверка витрин
# =========================
log_section("ПРОВЕРКА")
print("Топ-5 магазинов по выручке:")
sales_by_store_cat.show(5, truncate=False)

print("Выручка по дням (первые 5):")
daily_revenue.show(5, truncate=False)

# =========================
# Итог
# =========================
print("\n" + "=" * 60)
print("  ELT ЗАВЕРШЁН")
print("=" * 60)
print("Bronze : s3a://warehouse/elt/bronze/{sales,stores}")
print("Silver : s3a://warehouse/elt/silver/sales   (ORC+ZSTD)")
print("Gold   : s3a://warehouse/elt/gold/{sales_by_store,daily_revenue}")

spark.stop()
print("\ndone.")