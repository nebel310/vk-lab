import json

def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}

def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t}

cells = [
md("# Сжатие в Lakehouse: ORC + агрегации\n\nВторой ноутбук. Заполняет столбцы: `формат`, `Размер, ГБ`, `Файлов`, `Время записи, с`, `Агрегация первая, сек`, `Агрегация вторая и следующие, сек`.\n\nЧто делаем:\n1. Локальный бенчмарк **Parquet vs ORC** (PyArrow) — размер и скорость записи.\n2. Iceberg-таблицы в Trino: **Parquet** и **ORC** с разными кодеками — размер, кол-во файлов, время вставки.\n3. **Агрегации** в Trino: первый запуск vs повторные (эффект кэша).\n4. Итоговая таблица под твою форму."),

md("## 1. Импорты и клиенты"),
code("""import io, os, time, json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.orc as orc
import pyarrow.fs as pafs
import requests
import boto3
from IPython.display import display

S3_ENDPOINT = "http://localhost:9000"
S3_KEY = S3_MINIO_USER = "minioadmin"
S3_SECRET = "minioadmin"
TRINO_URL = "http://localhost:8080"

fs = pafs.S3FileSystem(
    endpoint_override=S3_ENDPOINT,
    access_key=S3_KEY, secret_key=S3_SECRET,
    scheme="http", region="us-east-1",
)
s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT,
                  aws_access_key_id=S3_KEY, aws_secret_access_key=S3_SECRET,
                  region_name="us-east-1")
print("clients ready")"""),

md("## 2. Клиент Trino с поддержкой session-свойств"),
code("""def trino(sql, catalog="lakekeeper", schema="lakehouse", session=None, timeout=900):
    h = {"X-Trino-User": "trino", "X-Trino-Catalog": catalog, "X-Trino-Schema": schema}
    if session:
        h["X-Trino-Session"] = ",".join(f"{k}={v}" for k, v in session.items())
    r = requests.post(TRINO_URL + "/v1/statement", data=sql.encode(), headers=h, timeout=timeout)
    j = r.json()
    if j.get("error"):
        raise RuntimeError(j["error"].get("message", str(j)))
    cols = [c["name"] for c in j.get("columns", [])]
    data = j.get("data") or []
    uri = j.get("nextUri")
    while uri:
        j = requests.get(uri, headers={"X-Trino-User": "trino"}, timeout=timeout).json()
        if j.get("error"):
            raise RuntimeError(j["error"].get("message", str(j)))
        data += j.get("data") or []
        uri = j.get("nextUri")
    return cols, data

def trino_time(sql, **kw):
    t0 = time.perf_counter()
    trino(sql, **kw)
    return time.perf_counter() - t0

print("trino client ready")"""),

md("## 3. Датасет (~3 млн строк) локально в памяти"),
code("""rng = np.random.default_rng(42)
N = 3_000_000
regions = ["Москва", "Санкт-Петербург", "Казань", "Екатеринбург", "Новосибирск", "Сочи", "Краснодар"]
categories = ["Электроника", "Одежда", "Продукты", "Дом", "Спорт", "Книги", "Игрушки"]

df = pd.DataFrame({
    "order_id":   np.arange(N, dtype=np.int64),
    "store_id":   rng.integers(1, 200, N, dtype=np.int64),
    "region":     pd.Categorical(rng.choice(regions, N)),
    "category":   pd.Categorical(rng.choice(categories, N)),
    "price":      np.round(rng.lognormal(mean=4.0, sigma=1.0, size=N), 2),
    "qty":        rng.integers(1, 10, N, dtype=np.int32),
    "order_date": pd.date_range("2024-01-01", periods=N, freq="15s").strftime("%Y-%m-%d %H:%M:%S"),
    "is_promo":   rng.integers(0, 2, N).astype(bool),
})
tbl = pa.Table.from_pandas(df, preserve_index=False)
print("rows:", tbl.num_rows)"""),

md("## 4. Локальный бенчмарк Parquet vs ORC\n\n`GZIP` в ORC у PyArrow называется `ZLIB` (тот же алгоритм deflate). Учти это при заполнении столбца «Сжатие»."),
code("""os.makedirs("bench", exist_ok=True)

parquet_codecs = ["NONE", "SNAPPY", "LZ4", "ZSTD", "GZIP"]
orc_codecs     = ["NONE", "SNAPPY", "LZ4", "ZSTD", "ZLIB"]

rows = []

for c in parquet_codecs:
    p = f"bench/t_parquet_{c}.parquet"
    kw = {"compression": c}
    if c == "ZSTD": kw["compression_level"] = 3
    if c == "GZIP": kw["compression_level"] = 9
    t0 = time.perf_counter()
    pq.write_table(tbl, p, **kw)
    wt = time.perf_counter() - t0
    size = os.path.getsize(p)
    t0 = time.perf_counter()
    pq.read_table(p)
    rt = time.perf_counter() - t0
    rows.append(("Parquet", c, size, wt, rt))

for c in orc_codecs:
    p = f"bench/t_orc_{c}.orc"
    kw = {"compression": c}
    if c == "ZSTD": kw["compression_level"] = 3
    t0 = time.perf_counter()
    orc.write_table(tbl, p, **kw)
    wt = time.perf_counter() - t0
    size = os.path.getsize(p)
    t0 = time.perf_counter()
    orc.read_table(p)
    rt = time.perf_counter() - t0
    rows.append(("ORC", c, size, wt, rt))

local = pd.DataFrame(rows, columns=["format", "codec", "size_bytes", "write_s", "read_s"])
local["size_mb"] = (local["size_bytes"] / 1e6).round(2)
local["size_gb"] = (local["size_bytes"] / 1e9).round(4)
local = local[["format", "codec", "size_gb", "size_mb", "write_s", "read_s"]]
local"""),

md("## 5. Запись в MinIO (bucket `warehouse`) и подсчёт файлов"),
code("""minio_rows = []
for _, r in local.iterrows():
    fmt = "parquet" if r["format"] == "Parquet" else "orc"
    key = f"warehouse/bench/{fmt}_{r['codec']}.{fmt}"
    src = f"bench/t_{fmt}_{r['codec']}.{fmt}"
    s3.upload_file(src, "warehouse", key.split("warehouse/",1)[1])
    minio_rows.append((r["format"], r["codec"], key))

for fmt, c, key in minio_rows:
    info = fs.get_file_info(key)
    print(f"{fmt:8s} {c:7s} {info.size/1e6:8.2f} MB   {key}")"""),

md("## 6. Iceberg в Trino: Parquet и ORC, разные кодеки\n\nTrino поддерживает session-свойство `iceberg.compression_codec`. Ставим его **перед каждой вставкой** — он влияет на записываемые файлы."),
code("""trino('CREATE SCHEMA IF NOT EXISTS lakekeeper.lakehouse')

DDL_PARQUET = '''CREATE TABLE lakekeeper.lakehouse.bench_parquet (
  order_id bigint, store_id bigint, region varchar, category varchar,
  price double, qty integer, order_date varchar, is_promo boolean
) WITH (format = 'PARQUET')'''

DDL_ORC = '''CREATE TABLE lakekeeper.lakehouse.bench_orc (
  order_id bigint, store_id bigint, region varchar, category varchar,
  price double, qty integer, order_date varchar, is_promo boolean
) WITH (format = 'ORC')'''

INSERT_SQL = '''
INSERT INTO {tbl}
SELECT
  a.s * 1000 + b.s,
  (a.s * 1000 + b.s) % 200,
  CASE (a.s * 1000 + b.s) % 7 WHEN 0 THEN 'Москва' WHEN 1 THEN 'Санкт-Петербург'
       WHEN 2 THEN 'Казань' WHEN 3 THEN 'Екатеринбург' WHEN 4 THEN 'Новосибирск'
       WHEN 5 THEN 'Сочи' ELSE 'Краснодар' END,
  CASE (a.s * 1000 + b.s) % 7 WHEN 0 THEN 'Электроника' WHEN 1 THEN 'Одежда'
       WHEN 2 THEN 'Продукты' WHEN 3 THEN 'Дом' WHEN 4 THEN 'Спорт'
       WHEN 5 THEN 'Книги' ELSE 'Игрушки' END,
  10.0 * ((a.s * 1000 + b.s) % 10000),
  (a.s * 1000 + b.s) % 10,
  '2024-01-01',
  ((a.s * 1000 + b.s) % 2 = 0)
FROM UNNEST(sequence(1, 2000)) AS a(s), UNNEST(sequence(1, 1000)) AS b(s)
'''

print("DDL/INSERT готовы")"""),

md("## 7. Прогон по форматам и кодекам"),
code("""trino_rows = []

plan = [
    ("Parquet", "SNAPPY", DDL_PARQUET, "bench_parquet"),
    ("Parquet", "ZSTD",   DDL_PARQUET, "bench_parquet"),
    ("Parquet", "GZIP",   DDL_PARQUET, "bench_parquet"),
    ("Parquet", "LZ4",    DDL_PARQUET, "bench_parquet"),
    ("Parquet", "NONE",   DDL_PARQUET, "bench_parquet"),
    ("ORC",     "SNAPPY", DDL_ORC,     "bench_orc"),
    ("ORC",     "ZSTD",   DDL_ORC,     "bench_orc"),
    ("ORC",     "GZIP",   DDL_ORC,     "bench_orc"),
    ("ORC",     "LZ4",    DDL_ORC,     "bench_orc"),
    ("ORC",     "NONE",   DDL_ORC,     "bench_orc"),
]

for fmt, codec, ddl, tname in plan:
    print(f"=== {fmt} / {codec} ===")
    trino(f"DROP TABLE IF EXISTS lakekeeper.lakehouse.{tname}")
    trino(ddl)

    sess = {"iceberg.compression_codec": codec}
    t0 = time.perf_counter()
    trino(INSERT_SQL.format(tbl=f"lakekeeper.lakehouse.{tname}"), session=sess)
    write_s = time.perf_counter() - t0

    cnt_cols, cnt_data = trino(f"SELECT count(*) FROM lakekeeper.lakehouse.{tname}")
    rows_in_table = int(cnt_data[0][0])

    # считаем файлы и суммарный размер в MinIO
    prefix = f"{tname}-"
    files, total = 0, 0
    for o in s3.list_objects_v2(Bucket="warehouse", Prefix=prefix).get("Contents", []):
        if o["Key"].endswith((".parquet", ".orc")):
            files += 1
            total += o["Size"]

    # агрегация: первый и последующие запуски
    agg_sql = f"SELECT category, COUNT(*) n, SUM(price) s FROM lakekeeper.lakehouse.{tname} GROUP BY category"
    t_first = trino_time(agg_sql)
    t_second = trino_time(agg_sql)
    t_third = trino_time(agg_sql)

    trino_rows.append({
        "format": fmt, "codec": codec,
        "rows": rows_in_table,
        "size_bytes": total, "files": files,
        "write_s": round(write_s, 2),
        "agg_first_s": round(t_first, 3),
        "agg_next_s": round((t_second + t_third) / 2, 3),
    })
    print(f"  rows={rows_in_table}, size={total/1e6:.1f} MB, files={files}, "
          f"write={write_s:.2f}s, agg1={t_first:.3f}s, agg2+={((t_second+t_third)/2):.3f}s")

iceberg = pd.DataFrame(trino_rows)
iceberg"""),

md("## 8. Итоговая таблица под форму"),
code("""out = iceberg.copy()
out["Размер, ГБ"] = (out["size_bytes"] / 1e9).round(4)
out["Сжатие"] = out["codec"]
out["формат"] = out["format"]
out["Файлов"] = out["files"]
out["Время записи, с"] = out["write_s"]
out["Агрегация первая, сек"] = out["agg_first_s"]
out["Агрегация вторая и следующие, сек"] = out["agg_next_s"]
out["Trino"] = "OK"
out["spark"] = "—"

result = out[["Сжатие","формат","Размер, ГБ","Файлов","Время записи, с",
              "Агрегация первая, сек","Агрегация вторая и следующие, сек",
              "Trino","spark"]]
display(result)
print()
print("Скопируй это в свою Excel-таблицу.")
result.to_csv("result_lakehouse.csv", index=False)
print("также сохранено в result_lakehouse.csv")"""),

md("## 9. Локальный бенчмарк — вторая таблица (для наглядности)"),
code("""display(local)
local.to_csv("result_local_bench.csv", index=False)
print("сохранено в result_local_bench.csv")"""),
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}
with open("orc_i_agregacii.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("готово: orc_i_agregacii.ipynb")