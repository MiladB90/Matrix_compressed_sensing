#!/usr/bin/env python3
import cvxpy as cp
import numpy as np
import pandas as pd
from numpy.random import Generator
from numpy import ndarray
from cvxpy.atoms import normNuc, multiply, norm
import cvxpy
from pandas import DataFrame
from scipy import stats as st
from sklearn.linear_model import LinearRegression
from dask_jobqueue import SLURMCluster

from EMS.manager import active_remote_engine, do_on_cluster, unroll_experiment, get_gbq_credentials
from dask.distributed import Client, LocalCluster
import coiled
import logging

logging.basicConfig(level=logging.INFO)


def seed(m: int, n: int, snr: float, p: float, mc: int) -> int:
    return round(1 + m * 1000 + n * 1000 + round(snr * 1000) + round(p * 1000) + mc * 100000)


def _df(c: list, l: list) -> DataFrame:
    d = dict(zip(c, l))
    return DataFrame(data=d, index=[0])

def df_experiment_svv(m: int, n: int, snr: float, p: float, mc: int, max_matrix_dim: int,
                      proj_dim: int, proj_entry_std: float,
                      cos_l: float, cos_r: float, svv: np.array,
                      slope: float, intercept: float, r_squared: float,
                      noise_frob_squared: float, entr_noise_std: float) -> DataFrame:
    c = ['m', 'n', 'snr', 'p', 'mc', 'max_matrix_dim', 'proj_dim', 'proj_entry_std',
         'cosL', 'cosR', 'nsspecfit_slope','nsspecfit_intercept', 'nsspecfit_r2',
         'noise_frob_squared', 'entr_noise_std']
    d = [m, n, snr, p, mc, max_matrix_dim, proj_dim, proj_entry_std,
         cos_l, cos_r, slope, intercept, r_squared,
         noise_frob_squared, entr_noise_std]
    for i, sv in enumerate(svv):
        c.append(f'sv{i}')
        d.append(sv)
    return _df(c, d)   

def make_data(m: int, n: int, p: float, rng: Generator) -> tuple:
    u = rng.normal(size=m)
    v = rng.normal(size=n)
    # normalizing to have unit vectors
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    
    M = np.outer(u, v)
    entr_noise_std = 1 / np.sqrt(n) 
    noise = rng.normal(0, entr_noise_std, (m, n))

    # random projection matrix: m * n --> p * m * n
    proj_dim = int(p * m * n)
    proj_entry_std = 1 / np.sqrt(m * n)
    proj_mat = rng.normal(0, proj_entry_std, (proj_dim, m * n))

    return u, v, M, proj_dim, proj_entry_std, proj_mat, noise, entr_noise_std

# optimization problem solver
def nuc_norm_cs_solver(m: int, n: int, proj_mat: ndarray, Y: ndarray) -> ndarray:
    # it solves argmin |X|_0 s.t. proj_mat * vec(X) = Y
    X = cp.Variable((m, n))
    objective = cp.Minimize(normNuc(X))
    Z = proj_mat @ cvxpy.vec(X, order='C') - Y
    constraints = [Z == 0]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    return X.value


# measurements
def vec_cos(v: np.array, vhat: np.array):
    return np.abs(np.inner(v, vhat))


def take_measurements_svv(Mhat, u, v, noise):
    uhatm, svv, vhatmh = np.linalg.svd(Mhat, full_matrices=False)
    cosL = vec_cos(u, uhatm[:, 0])
    cosR = vec_cos(v, vhatmh[0, :])

    # make noise_spectrum
    m, n = Mhat.shape
    noise_spectrum = np.linalg.svd(noise, compute_uv=False)

    # extract non-zero svs:
    r1 = sum(svv > 0.001)
    r2 = sum(noise_spectrum > 0.001)
    r = min(r1, r2)

    regr = LinearRegression()
    X = noise_spectrum[1:r].reshape(-1, 1)
    Y = svv[1:r].reshape(-1,1)

    regr.fit(X, Y)
    slope = regr.coef_[0, 0]
    intercept = regr.intercept_[0]
    r_squared = regr.score(X, Y)

    return cosL, cosR, svv, slope, intercept, r_squared

def do_matrix_compressed_sensing(*, m: int, n: int, snr: float, p: int, mc: int, max_matrix_dim: int) -> DataFrame:
    rng = np.random.default_rng(seed=seed(m, n, snr, p, mc))

    u, v, M, proj_dim, proj_entry_std, proj_mat, noise, entr_noise_std = make_data(m, n, p, rng)
    noisy_signal = snr * M + noise
    Y = proj_mat @ noisy_signal.flatten(order='C')
    Mhat = nuc_norm_cs_solver(m=m, n=n, proj_mat=proj_mat, Y=Y)

    cos_l, cos_r, svv, slope, intercept, r_squared = take_measurements_svv(Mhat, u, v, noise)

    # add noise energy
    noise_frob_squared = np.linalg.norm(noise, ord='fro') ** 2

    # fixed the length of svv for all runs
    fullsvv = np.full([max_matrix_dim], np.nan)
    fullsvv[:len(svv)] = svv

    return df_experiment_svv(m, n, snr, p, mc, max_matrix_dim, proj_dim, proj_entry_std,
                             cos_l, cos_r, fullsvv, slope, intercept, r_squared,
                             noise_frob_squared, entr_noise_std)

def test_experiment() -> dict:
    # 3800 rows
    exp = dict(table_name='milad_cs_0001',
               base_index=0,
               db_url='sqlite:///data/MatrixCompletion.db3',
               multi_res=[{
                   'm': [100],
                   'n': [100, 200, 300, 400],
                   'snr': [round(p, 3) for p in np.linspace(1, 10, 10)],
                   'p': [round(p, 3) for p in np.linspace(.1, 1, 19)],
                   'mc': [round(p) for p in np.linspace(1, 20, 20)]
               }]
              )
    # add max_matrix_dim for having unified output size
    mr = exp['multi_res']
    max_matrix_dim = 0
    for params in mr:
        paramlist =[max_matrix_dim]
        paramlist.extend(params['m'])
        paramlist.extend(params['n'])
        max_matrix_dim = max(paramlist)
    for params in mr:
        params['max_matrix_dim'] = [int(max_matrix_dim)]
    return exp

def do_coiled_experiment():
    exp = test_experiment()
    # logging.info(f'{json.dumps(dask.config.config, indent=4)}')
    software_environment = 'adonoho/matrix_completion'
    # logging.info('Deleting environment.')
    # coiled.delete_software_environment(software_environment)
    logging.info('Creating environment.')
    coiled.create_software_environment(
        name=software_environment,
        conda="environment-coiled.yml",
        pip=[
            "git+https://GIT_TOKEN@github.com/adonoho/EMS.git"
        ]
    )
    with coiled.Cluster(software=software_environment, n_workers=80) as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, do_matrix_compressed_sensing, client, credentials=get_gbq_credentials())


def do_local_experiment():
    exp = test_experiment()
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, do_matrix_compressed_sensing, client, credentials=get_gbq_credentials())

def do_sherlock_experiment():
    exp = test_experiment()
    nodes = 1000
    with SLURMCluster(queue='normal,owners,donoho,hns,stat,bigmem',
                      cores=1, memory='256GiB', processes=1, walltime='48:00:00') as cluster:
        cluster.scale(jobs=nodes)
        logging.info(cluster.job_script())
        with Client(cluster) as client:
            do_on_cluster(exp, do_matrix_compressed_sensing, client, credentials=get_gbq_credentials())
        cluster.scale(0)


def do_test():
    # print(get_gbq_credentials())
    # exp = test_experiment()
    # import json
    # j_exp = json.dumps(exp, indent=4)
    # print(j_exp)
    # params = unroll_experiment(exp)
    # for p in params:
    #     df = do_matrix_compressed_sensing(**p)
    #     print(df)
    pass
    # df = do_matrix_compressed_sensing(m=100, n=100, snr=10., p=2./3., mc=20, max_matrix_dim=100)
    df = do_matrix_compressed_sensing(m=12, n=20, snr=4., p=0.75, mc=20, max_matrix_dim=20)
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(df)

if __name__ == "__main__":
    # do_local_experiment()
    do_sherlock_experiment()
    # do_coiled_experiment()
    # do_test()
