#!/usr/bin/env python3

"""
    QuadriCovariance Experiment
"""

import os
import sqlalchemy as sa
from pandas import read_sql_table, DataFrame
from EMS.manager import active_remote_engine, do_on_cluster, unroll_experiment


# The Cloud SQL Python Connector can be used along with SQLAlchemy using the
# 'creator' argument to 'create_engine'
# def init_connection_engine() -> sa.engine.Engine:
#     connector = Connector()
#     def getconn() -> dbapi.Connection:
#         conn: dbapi.Connection = connector.connect(
#             os.environ["POSTGRES_CONNECTION_NAME"],
#             "pg8000",
#             user=os.environ["POSTGRES_USER"],
#             password=os.environ["POSTGRES_PASS"],
#             db=os.environ["POSTGRES_DB"],
#         )
#         return conn
#
#     engine = sa.create_engine(
#         "postgresql+pg8000://",
#         creator=getconn, echo=False
#     )
#     engine.dialect.description_encoding = None
#     return engine


def copy_results_to_sql(src_url: str, table_name: str):
    src_engine = sa.create_engine(src_url, echo=False)
    with src_engine.connect() as db:
        df = read_sql_table(table_name, db, index_col='index')

    # df.sort_values(by=['p', 'gam', 'mc'], inplace=True)
    # df.reset_index(drop=True, inplace=True)

    # dst_engine = init_connection_engine()
    dst_engine, metadata = active_remote_engine()
    with dst_engine.connect() as db:
        df.to_sql(table_name, db, if_exists='replace', method='multi', chunksize=500)

    src_engine.dispose()
    dst_engine.dispose()


def read_remote_results(table_name: str) -> DataFrame:
    remote_engine, metadata = active_remote_engine()
    with remote_engine.connect() as db:
        df = read_sql_table(table_name, db, index_col='index')

    remote_engine.dispose()
    return df


if __name__ == "__main__":
    # copy_results_to_sql('sqlite:///data/MatrixCompletion.db3',
    #                     'mc:0001')
    # init_connection_engine()
    df = read_remote_results('en:0030')
    print(df)
    pass