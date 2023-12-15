#!/usr/bin/env python3

import pandas as pd
from pandas import DataFrame, read_sql_table
from google.oauth2 import service_account
import pandas_gbq as gbq
import sqlalchemy as sa
import os.path


def get_gbq_credentials() -> service_account.Credentials:
    # path = '~/.config/gcloud/hs-deep-lab-donoho-ad747d94d2ec.json'  # Pandas-GBQ
    path = '~/.config/gcloud/hs-deep-lab-donoho-3d5cf4ffa2f7.json'  # Pandas-GBQ-DataSource
    expanded_path = os.path.expanduser(path)
    credentials = service_account.Credentials.from_service_account_file(expanded_path)
    return credentials


def write_results_to_gbq(db_url: str, table_name: str):
    engine = sa.create_engine(db_url, echo=True)

    with engine.connect() as db:
        df = read_sql_table(table_name, db)
        df.drop(['index'], axis=1, inplace=True)
        # df = read_sql_table(table_name, db, index_col='index')

    credentials = get_gbq_credentials()

    df.to_gbq(f"EMS.{table_name}-test", if_exists='append', chunksize=500, progress_bar=False, credentials=credentials)

    engine.dispose()


def read_gbq_table(table_name: str) -> DataFrame:
    credentials = get_gbq_credentials()
    sql = f'SELECT * FROM `EMS.{table_name}`'
    df = gbq.read_gbq(sql, progress_bar_type=None, credentials=credentials)
    return df


if __name__ == "__main__":
    write_results_to_gbq('sqlite:///data/MatrixCompletion.db3', 'mc-0006')
#     df = read_gbq_table('mc-0005')
#     print(df)
    pass