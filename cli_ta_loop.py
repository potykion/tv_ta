"""
- Загружает json с ТА и вставляет в бд
- Проставляет изменения тикеров, считает точность предсказаний, делает новые предсказания
- Строит график метрик предсказаний
"""

import json
import sqlite3
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from tv_ta.api import AnalysisRepo, PredictionRepo, Analysis
from tv_ta.config import BASE_DIR


class Deps:
    db = "tv_ta.db"
    table = "ta_indicators_1h"
    prediction_scores_table = "ta_prediction_scores_1h"

    # json_files_dir = Path(r"C:\Users\admin\Downloads\Telegram Desktop")
    json_files_dir = Path(__file__).parent.absolute()

    @cached_property
    def sqlite_conn(self):
        return sqlite3.connect(BASE_DIR / self.db, check_same_thread=False)

    @cached_property
    def sqlite_cursor(self):
        return self.sqlite_conn.cursor()

    @property
    def analysis_repo(self):
        return AnalysisRepo(self.sqlite_cursor, table=self.table)

    @property
    def prediction_repo(self):
        return PredictionRepo()


def main():
    deps = Deps()

    json_files: list[str] = (
        """ta_2024-07-12_15-02-15.json""".strip().split()
    )
    json_files = [file.strip() for file in json_files]

    for new_samples in read_samples_from_json(deps.json_files_dir, json_files, deps.analysis_repo):
        print("load_samples_from_json...")
        deps.analysis_repo.insert_samples(samples=new_samples)
        print()

        print("set_change_next...")
        set_change_next(deps.analysis_repo)
        print()

        print("predict...")
        predict(deps.analysis_repo, deps.prediction_repo)
        print()

        print("---")

    print("plotting score...")
    read_scores_and_plot(deps.db, deps.prediction_scores_table)
    print()

    print("Done!")


def read_samples_from_json(dir_, json_files, anal_repo):
    assert json_files, "No json_files set!"

    for file in json_files:
        path = dir_ / file
        with open(path, "r", encoding="utf-8") as f:
            raw_indicators = json.load(f)

        new_sample = anal_repo.get_last_sample() + 1

        new_samples = [Analysis(**ind).model_copy(update=dict(sample=new_sample)) for ind in raw_indicators]

        yield new_samples


def set_change_next(repo: AnalysisRepo):
    last_sample = repo.get_last_sample()
    if last_sample < 2:
        return

    with repo.q.commit_after():
        repo.update_change_next()

    last_sample_w_pred = repo.get_last_sample_w_prediction()
    if last_sample_w_pred < 3:
        return
    print("prev prediction results:")
    results = repo.list_prediction_results(last_sample_w_pred)
    print(tabulate(results))

    accuracy, rmse, r2 = repo.get_prediction_scores(last_sample_w_pred)
    print(f"score: accuracy📈 = {accuracy}; RMSE📉 = {rmse}; R2📈0️⃣ = {r2}")
    repo.insert_prediction_scores(accuracy, rmse, r2)
    max_accuracy = repo.get_max_accuracy()
    print(f"max accuracy: {max_accuracy}")


def predict(repo: AnalysisRepo, predict_repo: PredictionRepo):
    last_sample = repo.get_last_sample()
    if last_sample < 2:
        return

    analysis_to_train = repo.list_train_samples()
    analysis_to_predict = repo.list_predict_samples()

    X_train = predict_repo.analysis_to_X_df(analysis_to_train)
    y_train = predict_repo.analysis_to_y_df(analysis_to_train)
    X_predict = predict_repo.analysis_to_X_df(analysis_to_predict)

    predictions = predict_repo.predict(X_train, y_train, X_predict)

    with repo.q.commit_after():
        for anal, pred in zip(analysis_to_predict, predictions):
            repo.update_prediction(anal.id, pred)

    last_sample = repo.get_last_sample()
    print("new predictions:")
    results = repo.list_predictions(last_sample)
    print(tabulate(results))


def read_scores_and_plot(db, table):
    df = read_table_as_df(db, table)
    plot(df)


def read_table_as_df(db, table):
    conn = sqlite3.connect(BASE_DIR / db, check_same_thread=False)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df


def plot(df):
    plt.figure(figsize=(10, 6))

    plt.axhline(y=0, color="gray", linestyle="-")
    plt.axhline(y=0.5, color="lightgray", linestyle="--")

    plt.plot(df["dt"], df["accuracy"], label="Accuracy ↑")
    plt.plot(df["dt"], df["rmse"], label="RMSE ↓")
    plt.plot(df["dt"], df["r2"], label="R² ↑")



    plt.xlabel("Date-Time")
    plt.ylabel("Value")
    plt.title("Line Chart of Accuracy, RMSE, and R^2 over Time")
    plt.legend()

    plt.xticks([])

    # Annotating the last point
    last_point = df.iloc[-1]  # Get the last row of the DataFrame
    last_x_value = last_point["dt"]  # Assuming 'dt' is the column name for dates/times
    last_y_values = [
        last_point["accuracy"],
        last_point["rmse"],
        last_point["r2"],
    ]  # Values for accuracy, rmse, and r2
    labels = ["Accuracy", "RMSE", "R²"]  # Labels corresponding to the lines

    for i, (y_value, label) in enumerate(zip(last_y_values, labels)):
        plt.annotate(
            f"{label}: {y_value}",
            xy=(last_x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5),
        )

    plt.savefig("line_chart.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
