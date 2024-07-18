import sqlite3
from functools import cached_property
from pathlib import Path

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tv_ta.config import BASE_DIR
from tv_ta.api import PredictionRepo, AnalysisRepo


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


def main(deps: Deps):
    analysis_to_train = deps.analysis_repo.list_train_samples()
    X = deps.prediction_repo.analysis_to_X_df(analysis_to_train)
    y = deps.prediction_repo.analysis_to_y_df(analysis_to_train)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    def objective(params):
        preds = deps.prediction_repo.predict(X_train, y_train, X_test, params=params, to_list=False)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return {"loss": rmse, "status": STATUS_OK}

    space = {
        "max_depth": hp.quniform("max_depth", 1, 15, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.5)),
        "n_estimators": hp.quniform("n_estimators", 50, 1000, 50),
        "subsample": hp.uniform("subsample", 0.1, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.1, 1.0),
        "reg_alpha": hp.uniform("reg_alpha", 0, 10),
        "reg_lambda": hp.uniform("reg_lambda", 0, 10),
        "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
        "gamma": hp.loguniform("gamma", np.log(0.1), np.log(1)),
    }

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        show_progressbar=True,
    )

    print("Best Params:", best_params)


if __name__ == "__main__":
    main(Deps())
