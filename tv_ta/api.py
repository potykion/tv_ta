import datetime
import decimal
import json
import sqlite3

import tradingview_ta
from pydantic import BaseModel
from tradingview_ta import get_multiple_analysis, Interval

from tv_ta.q import Q

from tv_ta.iter_utils import groupby_dict

try:
    import numpy as np
    import pandas as pd
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
except ImportError:
    ...


class Analysis(BaseModel):
    id: int | None = None
    ticker: str
    dt: datetime.datetime
    interval: str = "1d"
    indicators: dict
    change: decimal.Decimal
    change_next: decimal.Decimal | None = None
    change_predict: decimal.Decimal | None = None
    sample: int
    indicators_prev_day: dict | None = None


def analysis_from_row(row: sqlite3.Row) -> Analysis:
    return Analysis(
        id=row["id"],
        ticker=row["ticker"],
        dt=row["dt"],
        interval=row["interval"],
        indicators=json.loads(row["indicators"]),
        indicators_prev_day=json.loads(row["indicators_prev_day"] or "{}"),
        change=row["change"],
        change_next=row["change_next"],
        change_predict=row["change_predict"],
        sample=row["sample"],
    )


class TAAnalysisRepo:
    exchange = "MOEX"

    def __init__(self, interval=Interval.INTERVAL_1_DAY):
        self.interval = interval

    def load_samples(self, tickers, dt, sample):
        tickers_w_exchange = self.make_ticker_w_exchange(tickers)
        ta_analysis = get_multiple_analysis(
            screener="russia",
            interval=self.interval,
            symbols=tickers_w_exchange,
        )

        analysis_samples = []
        for index, ticker in enumerate(tickers):
            ta = ta_analysis[tickers_w_exchange[index]]
            if not ta:
                continue

            indicators = ta.indicators
            analysis = Analysis(
                ticker=ticker,
                dt=dt,
                interval="1h",
                indicators=indicators,
                change=indicators["change"],
                sample=sample,
            )
            analysis_samples.append(analysis)

        return analysis_samples

    def make_ticker_w_exchange(self, tickers):
        tickers_w_exchange = []
        for ticker in tickers:
            if ticker.startswith(self.exchange + ":"):
                tickers_w_exchange.append(ticker)
            else:
                tickers_w_exchange.append(f"{self.exchange}:{ticker}")
        return tickers_w_exchange


class AnalysisRepo:
    def __init__(
            self,
            sqlite_cursor,
            *,
            table=None,
            prediction_scores_table=None,
    ):
        self.q = Q(sqlite_cursor, select_as=analysis_from_row)
        self.table = table or "ta_indicators_1d"
        self.prediction_scores_table = prediction_scores_table or "ta_prediction_scores_1h"

    def insert_samples(self, samples):
        with self.q.commit_after():
            for analysis in samples:
                params = (
                    analysis.ticker,
                    analysis.dt,
                    analysis.sample,
                    analysis.interval,
                    json.dumps(analysis.indicators),
                    str(analysis.change),
                )
                self.q.execute(
                    f'insert into {self.table} (ticker, dt, sample, "interval", indicators, change) values (?, ?, ?, ?, ?, ?)',
                    params,
                )

    def get_last_sample(self):
        return self.q.select_val(f"select max(sample) from {self.table}") or 0

    def get_last_sample_w_prediction(self):
        return (
                self.q.select_val(f"select max(sample) from {self.table} where change_predict_2 is not null") or 0
        )

    def update_change_next(self):
        analysis_by_sample = groupby_dict(
            self.q.select_all(
                f"select * from {self.table} where change_next is null order by sample, ticker"
            ),
            key_func=lambda anal: anal.sample,
        )
        for sample, sample_analysis in analysis_by_sample.items():
            next_sample_analysis = self.q.select_all(
                f"select * from {self.table} where sample = ? order by ticker", (sample + 1,)
            )
            for anal, next_anal in zip(sample_analysis, next_sample_analysis):
                anal.change_next = next_anal.change
                self.q.execute(
                    f"UPDATE {self.table} SET change_next = ? where id = ?",
                    (str(anal.change_next), anal.id),
                )

    def list_prediction_results(self, sample):
        return self.q.select_all(
            f"select ticker, change_next, change_predict_2 from {self.table} where sample = ? order by change_predict_2 desc limit 10",
            (sample,),
            as_=dict,
        )

    def list_predictions(self, sample):
        return self.q.select_all(
            f"select ticker, change_predict_2 from {self.table} where sample = ? order by change_predict_2 desc limit 10",
            (sample,),
            as_=dict,
        )

    def get_prediction_scores(self, sample):
        values = self.q.select_all(
            f"select change_next, change_predict_2 from {self.table} where sample = ?", (sample,), as_=dict
        )
        actual_values = np.array([val["change_next"] for val in values])
        predictions = np.array([val["change_predict_2"] for val in values])

        actual_classes = np.array([int((val["change_next"] or 0) > 0) for val in values])
        prediction_classes = np.array([int(val["change_predict_2"] > 0) for val in values])

        accuracy = accuracy_score(actual_classes, prediction_classes)
        # precision = precision_score(actual_classes, prediction_classes, average='binary')
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        r2 = r2_score(actual_values, predictions)

        return accuracy, rmse, r2

    def list_train_samples(self):
        return self.q.select_all(f"select * from main.{self.table} where change_next is not null")

    def list_predict_samples(self):
        return self.q.select_all(
            f"select * from {self.table} "
            f"where change_next is null and "
            f"sample = (select max(sample) from {self.table})"
        )

    def update_prediction(self, id, prediction):
        self.q.execute(f"update {self.table} set change_predict_2 =? where id =?", (prediction, id))

    def insert_prediction_scores(self, accuracy, rmse, r2):
        self.q.execute(
            f"insert into {self.prediction_scores_table} (dt, accuracy, rmse, r2) values (?, ?, ?, ?)",
            (datetime.datetime.now(), accuracy, rmse, r2),
            commit=True,
        )

    def get_max_accuracy(self):
        return self.q.select_val(f"select max(accuracy) from {self.prediction_scores_table}")


class PredictionRepo:
    model_params = {
        "colsample_bytree": 0.5891068053804113,
        "gamma": 0.161134814723787,
        "learning_rate": 0.018598396703702173,
        "max_depth": 2.0,
        "min_child_weight": 6.0,
        "n_estimators": 500.0,
        "reg_alpha": 9.526447733547432,
        "reg_lambda": 4.144959618046851,
        "subsample": 0.1518126264708962,
    }

    def predict(self, X_train, y_train, X_predict, *, params=None, to_list=True):
        params = params or self.model_params
        params["n_estimators"] = int(params["n_estimators"])
        params["max_depth"] = int(params["max_depth"])

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_predict)

        if to_list:
            predictions = predictions.tolist()

        return predictions

    def analysis_to_X_df(self, analysis: list[Analysis]) -> "pd.DataFrame":
        indicators = tradingview_ta.TA_Handler.indicators  # = 90 features

        def str_indicators(i):
            """xgboost compatible cols"""
            return ["".join(ch for ch in ind if ch not in "[]<") + str(i) for ind in indicators]

        columns = [
            *str_indicators(1),
        ]

        X = pd.DataFrame(
            [
                [
                    *(anal.indicators[ind] for ind in indicators),
                ]
                for anal in analysis
            ],
            columns=columns,
        )
        X = X.fillna(0)
        return X

    def analysis_to_y_df(self, analysis: list[Analysis]) -> "pd.DataFrame":
        y = pd.DataFrame([anal.change_next for anal in analysis], columns=["change_next"])
        y = y["change_next"].values.ravel()
        return y
