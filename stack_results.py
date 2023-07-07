#!/usr/bin/env python3

"""
    MatrixCompletion Experiment
"""

from numpy import float64
from pandas import DataFrame, read_sql_table, concat
import sqlalchemy as sa


def stack_results(table_name: str):
    engine = sa.create_engine('sqlite:///data/MatrixCompletion.db3', echo=False)

    with engine.connect() as db:
        pdf = read_sql_table(table_name, db, index_col='index')

    engine.dispose()

    pdf.sort_values(by=['p', 'c4', 'mc'], inplace=True)
    pdf.reset_index(drop=True, inplace=True)
    pdf.drop(['m', 'c4', 'd_type', 'q_type'], axis=1, inplace=True)

    mean_table = pdf.groupby(by=['gam', 'p']).mean()
    print(mean_table)
    mean_table.to_csv('gam_p_mean_' + table_name + '.csv')
    mean_table = pdf.groupby(by=['p', 'gam']).mean()
    print(mean_table)
    mean_table.to_csv('p_gam_mean_' + table_name + '.csv')


def stack_q_results(table_name: str):
    engine = sa.create_engine('sqlite:///data/MatrixCompletion.db3', echo=False)

    with engine.connect() as db:
        pdf = read_sql_table(table_name, db, index_col='index')

    engine.dispose()

    pdf.reset_index(drop=True, inplace=True)
    pdf.drop(['p', 'm', 'c4', 'd_type'], axis=1, inplace=True)

    mean_table = pdf.groupby(by=['gam', 'q_type']).mean()
    print(mean_table)
    mean_table.to_csv('gam_q_mean_' + table_name + '.csv')


def write_results_to_csv(table_name: str):
    engine = sa.create_engine('sqlite:///data/MatrixCompletion.db3', echo=True)
    ldb = engine.connect()  # ldb == Local Database. 'db' will be the remote persistent db.

    pdf = read_sql_table(table_name, ldb, index_col='index')
    # pdf.sort_values(by=['c4'], inplace=True)
    # pdf.sort_values(by=['c4', 'mc'], inplace=True)
    # pdf.reset_index(drop=True, inplace=True)

    # pdfg = pdf.groupby(by='c4').mean()
    # print(pdfg)
    pdf.to_csv(f'{table_name}.csv')

    pass

    ldb.close()
    engine.dispose()


def sort_rewrite_results_to_sql(table_name: str, new_table_name: str):
    engine = sa.create_engine('sqlite:///data/MatrixCompletion.db3', echo=True)
    ldb = engine.connect()  # ldb == Local Database. 'db' will be the remote persistent db.

    pdf = read_sql_table(table_name, ldb, index_col='index')
    # pdf.sort_values(by=['c4', 'mc'], inplace=True)
    pdf.reset_index(drop=True, inplace=True)

    pdf.to_sql(new_table_name, ldb, if_exists='replace', method='multi')

    ldb.close()
    engine.dispose()


def reset_index(table_name: str):
    engine = sa.create_engine('sqlite:///data/MatrixCompletion.db3', echo=False)

    with engine.connect() as ldb:
        pdf = read_sql_table(table_name, ldb, index_col='index')
        pdf.reset_index(drop=True, inplace=True)
        pdf.to_sql(table_name, ldb, if_exists='replace', method='multi')

    engine.dispose()


def drop_table(table_name):
    engine = sa.create_engine('sqlite:///data/MatrixCompletion.db3', echo=True)
    base = sa.declarative_base()
    metadata = sa.MetaData(engine, reflect=True)
    table = metadata.tables.get(table_name)
    if table is not None:
        logging.info(f'Deleting {table_name} table')
        base.metadata.drop_all(engine, [table], checkfirst=True)


if __name__ == "__main__":
    write_results_to_csv('mc-0002')
    # sort_rewrite_results_to_sql('en:0011', 'en:0012')
#     reset_index('en:0022')
#     stack_q_results('en:0023')
#     stack_results('en:0023')
    # drop_table('en:0021')
