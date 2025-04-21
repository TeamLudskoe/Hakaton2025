import polars as pl
from pathlib import Path
import multiprocessing
import time
import shutil
import re

# Конфигурация
BASE_PATH = Path(r"D:\HakatonNew\Dataset\Dataset\1000k\telecom1000k")
TEMP_DIR = Path("temp_processing")
OUTPUT_FILE = "result_dispersia_time15.csv"

# Временные интервалы для ночного времени (23:00-06:00)
NIGHT_START = 23
NIGHT_END = 6


def init_temp_dir():
    """Инициализация временной директории"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(exist_ok=True)


def clean_temp_dir():
    """Очистка временной директории"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)


def extract_time_from_filename(filename):
    """Извлечение времени из имени файла"""
    match = re.search(r'(\d{2})_(\d{2})_(\d{2})', filename.stem)
    if match:
        return int(match.group(1))  # Возвращаем только час для простоты
    return None


def is_night_time(hour):
    """Проверка, относится ли время к ночному"""
    if hour is None:
        return False
    return hour >= NIGHT_START or hour < NIGHT_END


def process_psx_file(file_path):
    """Обработка одного файла PSX с сохранением во временный файл"""
    try:
        temp_file = TEMP_DIR / f"processed_{file_path.stem}.parquet"
        hour = extract_time_from_filename(file_path)
        time_type = "night" if is_night_time(hour) else "day"

        if file_path.suffix == '.csv':
            df = pl.read_csv(file_path, separator=',', infer_schema_length=10000)
        elif file_path.suffix == '.txt':
            df = pl.read_csv(file_path, separator='|', infer_schema_length=10000)
        else:
            return None

        # Очистка данных - исключаем отрицательные значения
        df = df.filter(
            (pl.col("UpTx") >= 0) &
            (pl.col("DownTx") >= 0))

        # Добавляем информацию о времени суток
        df = df.with_columns(pl.lit(time_type).alias("TimeType"))

        df.select(["IdSubscriber", "UpTx", "DownTx", "TimeType"]).write_parquet(temp_file)
        return temp_file

    except Exception as e:
        print(f"⚠️ Ошибка обработки {file_path.name}: {str(e)[:100]}...")
        return None


def calculate_iqr_thresholds(df):
    """Вычисление пороговых значений с использованием IQR с учетом времени суток"""
    return (
        df.group_by(["Type", "TimeType"])
        .agg(
            pl.quantile("ratio", 0.25).alias("q1"),
            pl.quantile("ratio", 0.75).alias("q3")
        )
        .with_columns(
            (pl.col("q3") - pl.col("q1")).alias("iqr"),
            (pl.col("q3") + 2.5 * (pl.col("q3") - pl.col("q1"))).alias("upper_threshold")
        )
    )


def load_and_process_data():
    """Основная функция обработки данных"""
    init_temp_dir()

    try:
        # 1. Загрузка активных подписчиков
        print("🔍 Загрузка данных подписчиков...")
        subs = pl.read_csv(BASE_PATH / "subscribers.csv").filter(
            pl.col("Status") == "ON"
        ).select(["IdOnPSX", "IdClient"])

        # 2. Параллельная обработка файлов PSX
        psx_files = list(BASE_PATH.glob("psx_*.*"))
        if not psx_files:
            raise FileNotFoundError("Не найдены файлы psx_*")

        print(f"🔄 Обработка {len(psx_files)} файлов трафика...")

        with multiprocessing.Pool(processes=min(4, multiprocessing.cpu_count())) as pool:
            processed_files = pool.map(process_psx_file, psx_files)

        psx_dfs = [pl.read_parquet(f) for f in processed_files if f]
        if not psx_dfs:
            raise ValueError("Не удалось загрузить данные трафика")

        psx = pl.concat(psx_dfs)

        # 3. Агрегация трафика (дополнительная проверка на отрицательные значения)
        print("🧮 Расчет статистики трафика...")
        traffic = psx.group_by(["IdSubscriber", "TimeType"]).agg(
            pl.sum("UpTx").alias("sumUpTx"),
            pl.sum("DownTx").alias("sumDownTx")
        ).filter(pl.col("sumDownTx") > 0)

        # 4. Добавление информации о клиентах
        print("👥 Добавление данных клиентов...")
        clients_df = pl.read_parquet(BASE_PATH / "client.parquet").select(["Id", "IdPlan"])
        physical_df = pl.read_parquet(BASE_PATH / "physical.parquet").select("Id")

        # 5. Определение типа клиента и соединение данных
        print("🔗 Соединение данных и определение типа клиента...")
        client_types = (
            subs.select("IdClient")
            .join(
                physical_df.rename({"Id": "IdClient"}).with_columns(pl.lit("P").alias("Type")),
                on="IdClient",
                how="left"
            )
            .with_columns(pl.col("Type").fill_null("J"))
            .unique()
        )

        joined = (
            subs.join(traffic, left_on="IdOnPSX", right_on="IdSubscriber", how="inner")
            .join(client_types, on="IdClient", how="left")
            .join(clients_df, left_on="IdClient", right_on="Id", how="left")
            .with_columns(
                (pl.col("sumUpTx") / pl.col("sumDownTx")).alias("ratio")
            )
            .filter(pl.col("sumDownTx") > 0)
        )

        # 6. Расчет IQR порогов с учетом времени суток
        print("📊 Расчет IQR порогов для аномалий...")
        iqr_stats = calculate_iqr_thresholds(joined)

        # 7. Фильтрация аномалий
        print("🔎 Фильтрация аномальных записей...")
        anomalies = (
            joined.join(iqr_stats, on=["Type", "TimeType"])
            .filter(pl.col("ratio") > pl.col("upper_threshold"))
        )

        # 8. Формирование результата с уникальными значениями
        result = (
            anomalies.select(
                pl.col("IdOnPSX").alias("Id"),
                pl.col("IdClient").alias("UID"),
                pl.col("Type"),
                pl.col("IdPlan").alias("IdPlan"),
                pl.lit(True).alias("TurnOn"),
                pl.lit(True).alias("Hacked"),
                pl.lit(0).alias("Traffic")
            )
            .unique(subset=["Id"])  # Оставляем только уникальные Id
        )
        # 9. Сохранение результатов
        print(f"💾 Сохранение {result.height} аномалий в {OUTPUT_FILE}...")
        result.write_csv(OUTPUT_FILE)
        print("✅ Обработка завершена успешно!")

        # Дополнительная статистика
        #print("\n📈 Статистика обнаруженных аномалий:")
        #print(result.group_by(["Type", "Period"]).agg(pl.len()))

        return True

    except Exception as e:
        print(f"❌ Критическая ошибка: {type(e).__name__}: {str(e)}")
        return False
    finally:
        clean_temp_dir()


if __name__ == "__main__":
    print("🚀 Запуск обработки данных...")
    start_time = time.time()

    success = load_and_process_data()

    elapsed = time.time() - start_time
    print(f"⏱ Общее время выполнения: {elapsed:.2f} секунд")
    exit(0 if success else 1)