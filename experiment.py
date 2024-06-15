#!/usr/bin/env python3
import cvxpy as cp
import numpy as np
from numpy.random import Generator
from cvxpy.atoms import normNuc, multiply, norm
from pandas import DataFrame
from scipy import stats as st
from sklearn.linear_model import LinearRegression
# from dask_jobqueue import SLURMCluster

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
                      cos_l: float, cos_r: float, svv: np.array,
                      slope: float, intercept: float, r_squared: float,
                      noise_frob_squared: float, entr_noise_std: float) -> DataFrame:
    c = ['m', 'n', 'snr', 'p', 'mc', 'max_matrix_dim', 'cosL', 'cosR', 'nsspecfit_slope',
         'nsspecfit_intercept', 'nsspecfit_r2', 'noise_frob_squared', 'entr_noise_std']
    d = [m, n, snr, p, mc, max_matrix_dim, cos_l, cos_r, slope, intercept, r_squared, noise_frob_squared, entr_noise_std]
    for i, sv in enumerate(svv):
        c.append(f'sv{i}')
        d.append(sv)
    return _df(c, d)   


def make_data(m: int, n: int, p: float, rng: Generator) -> tuple:
    u = rng.normal(size=m)
    v = rng.normal(size=n)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    M = np.outer(u, v)
    entr_noise_std = 1 / np.sqrt(n) 
    noise = rng.normal(0, entr_noise_std, (m, n))
    observes = st.bernoulli.rvs(p, size=(m, n), random_state=rng)

    return u, v, M, noise, observes, entr_noise_std   


# problem setup
def nuc_norm_problem(Y, observed) -> tuple:
    X = cp.Variable(Y.shape)
    objective = cp.Minimize(normNuc(X))
    Z = multiply(X - Y, observed)
    constraints = [Z == 0]

    prob = cp.Problem(objective, constraints)

    prob.solve()

    return X, prob


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
    X = noise_spectrum[1:r].reshape(-1,1)
    Y = svv[1:r].reshape(-1,1)

    regr.fit(X, Y)
    slope = regr.coef_[0, 0]
    intercept = regr.intercept_[0]
    r_squared = regr.score(X, Y)

    return cosL, cosR, svv, slope, intercept, r_squared

def do_matrix_completion(*, m: int, n: int, snr: float, p: float, mc: int, max_matrix_dim: int) -> DataFrame:
    rng = np.random.default_rng(seed=seed(m, n, snr, p, mc))

    u, v, M, noise, obs, entr_noise_std = make_data(m, n, p, rng)
    Y = snr * M + noise
    X, _ = nuc_norm_problem(Y=Y, observed=obs)
    Mhat = X.value

    cos_l, cos_r, svv, slope, intercept, r_squared = take_measurements_svv(Mhat, u, v, noise)

    # add noise energy
    noise_frob_squared = np.linalg.norm(noise, ord='fro') ** 2

    # fixed the length of svv for all runs
    fullsvv = np.full([max_matrix_dim], np.nan)
    fullsvv[:len(svv)] = svv

    return df_experiment_svv(m, n, snr, p, mc, max_matrix_dim, cos_l, cos_r, fullsvv, slope, intercept, r_squared,
                             noise_frob_squared, entr_noise_std)


def test_experiment() -> dict:
    # exp = dict(table_name='test',
    #            base_index=0,
    #            db_url='sqlite:///data/MatrixCompletion.db3',
    #            multi_res=[{
    #                'n': [10],
    #                'snr': [1.0],
    #                'p': [0.0],
    #                'mc': [0]
    #            }])
    # exp = dict(table_name='test',
    #            base_index=0,
    #            db_url='sqlite:///data/MatrixCompletion.db3',
    #            multi_res=[{
    #                'n': [round(p) for p in np.linspace(10, 100, 10)],
    #                'snr': [round(p, 0) for p in np.linspace(1, 10, 10)],
    #                'p': [round(p, 1) for p in np.linspace(0, 1, 11)],
    #                'mc': list(range(5))
    #            }])
    # exp = dict(table_name='mc:0001',
    #            base_index=0,
    #            db_url='sqlite:///data/MatrixCompletion.db3',
    #            multi_res=[{
    #                'n': [round(p) for p in np.linspace(10, 1000, 41)],
    #                'snr': [round(p, 3) for p in np.linspace(1, 20, 39)],
    #                'p': [round(p, 3) for p in np.linspace(0., 1., 41)],
    #                'mc': list(range(20))
    #            }])
    # exp = dict(table_name='mc-0002',
    #            base_index=0,
    #            db_url='sqlite:///data/MatrixCompletion.db3',
    #            multi_res=[{
    #                'n': [round(p) for p in np.linspace(10, 500, 21)],
    #                'snr': [round(p, 3) for p in np.linspace(1, 20, 39)],
    #                'p': [round(p, 3) for p in np.linspace(0., 1., 41)],
    #                'mc': list(range(20))
    #            }])
    # exp = dict(table_name='mc-0003',
    #            base_index=0,
    #            db_url='sqlite:///data/MatrixCompletion.db3',
    #            multi_res=[{
    #                # 'n': [round(p) for p in np.linspace(10, 500, 21)],
    #                'n': [500],
    #                'snr': [round(p, 3) for p in np.linspace(1, 20, 20)],
    #                'p': [round(p, 3) for p in np.linspace(0.05, 1., 20)],
    #                'mc': [20]
    #            }])
    # exp = dict(table_name='mc-0003',
    #            base_index=400,
    #            db_url='sqlite:///data/MatrixCompletion.db3',
    #            multi_res=[{
    #                'n': [500],
    #                'snr': [round(p, 3) for p in np.linspace(1, 20, 20)],
    #                'p': [1.],
    #                'mc': [20]
    #            },{
    #                'n': [500],
    #                'snr': [round(p, 3) for p in np.linspace(1, 20, 20)],
    #                'p': [2./3.],
    #                'mc': [20]
    #            }])
    # mr = exp['multi_res']
    # for snr in np.linspace(3., 6., 31):
    #     for x in np.linspace(1.5, 4.0, 26):
    #         p = (x / snr) ** 2
    #         if p <= 1.0:
    #             d = {
    #                 'n': [500],
    #                 'snr': [snr],
    #                 'p': [p],
    #                 'mc': [20]
    #             }
    #             mr.append(d)
    # exp = dict(table_name='mc-0004',
    #            base_index=0,
    #            db_url='sqlite:///data/MatrixCompletion.db3',
    #            multi_res=[{
    #                'm': [500],
    #                'n': [500],
    #                'snr': [round(p, 3) for p in np.linspace(1, 20, 20)],
    #                'p': [round(p, 3) for p in np.linspace(0.05, 1., 20)],
    #                'mc': [20]
    #            }])
    # exp = dict(table_name='mc-0004',
    #            base_index=400,
    #            db_url='sqlite:///data/MatrixCompletion.db3',
    #            multi_res=[{
    #                'm': [100, 200, 300, 400, 500],
    #                'n': [500],
    #                'snr': [round(p, 3) for p in np.linspace(1, 10, 10)],
    #                'p': [0.5, 0.75, 1.0],
    #                'mc': [20]
    #            }])
    # exp = dict(table_name='mc-0004',
    #            base_index=520,
    #            db_url='sqlite:///data/MatrixCompletion.db3',
    #            multi_res=[{
    #                'm': [100, 200, 300, 400, 500],
    #                'n': [500],
    #                'snr': [round(p, 3) for p in np.linspace(1, 10, 10)],
    #                'p': [0.5, 0.75, 1.0],
    #                'mc': list(range(21, 60))
    #            }])
    # exp = dict(table_name='mc-0005',
    #            base_index=0,
    #            db_url='sqlite:///data/MatrixCompletion.db3',
    #            multi_res=[{
    #                'm': [500],
    #                'n': [500],
    #                'snr': [round(p, 3) for p in np.linspace(1, 10, 10)],
    #                'p': [0.5, 0.75, 1.0],
    #                'mc': [20]
    #            }])
    # exp = dict(table_name='mc-0007',
    #            base_index=0,
    #            db_url='sqlite:///data/EMS.db3',
    #            multi_res=[])
    # mr = exp['multi_res']
    # for snr in np.linspace(1.5, 2.25, 4):
    #     p = 1./(2. * (snr - 1.) ** 2 + 1.)
    #     d = {
    #         'm': [500],
    #         'n': [500],
    #         'snr': [snr],
    #         'p': list(np.linspace(p**2, min(p**3, 1.), 10)),
    #         'mc': [20]
    #     }
    #     mr.append(d)
    # exp = dict(table_name='mc-0007',
    #            description="20230824, Add more seed indicies to the existing [20-40] in table mc-0007, "
    #             "i.e. do 100 more range(41,141).",
    #            params=[])
    # mr = exp['params']
    # for snr in np.linspace(1.5, 2.25, 4):
    #     p = 1./(2. * (snr - 1.) ** 2 + 1.)
    #     d = {
    #         'm': [500],
    #         'n': [500],
    #         'snr': [snr],
    #         'p': list(np.linspace(p**2, min(p**3, 1.), 10)),
    #         'mc': list(range(41, 141))
    #     }
    #     mr.append(d)
    # exp = dict(table_name='mc-0008',
    #            base_index=0,
    #            db_url='sqlite:///data/EMS.db3',
    #            multi_res=[])
    # mr = exp['multi_res']
    # for snr in np.linspace(1.5, 2.25, 4):
    #     p = 1./(snr * (2. * (snr - 1.) ** 2 + 1.))
    #     d = {
    #         'm': [500],
    #         'n': [500],
    #         'snr': [snr],
    #         'p': list(np.linspace(p / 2., min(p * 3, 1.), 10)),
    #         'mc': [20]
    #     }
    #     mr.append(d)
    # exp = dict(table_name='mc-0009',
    #            base_index=0,
    #            db_url='sqlite:///data/EMS.db3',
    #            multi_res=[])
    # mr = exp['multi_res']
    # for m in np.linspace(10, 100, 19):
    #     d = {
    #         'm': [round(m)],
    #         'n': [round(m)],
    #         'snr': [1, 5, 10, 20, 1000],
    #         'p': [round(p, 1) for p in np.linspace(.1, 1, 10)],
    #         'mc': list(range(21,31))
    #     }
    #     mr.append(d)
    # for m in np.linspace(100, 500, 21):
    #     d = {
    #         'm': [round(m)],
    #         'n': [round(m)],
    #         'snr': [1, 5, 10, 20, 1000],
    #         'p': [round(p, 1) for p in np.linspace(.1, 1, 10)],
    #         'mc': list(range(21,31))
    #     }
    #     mr.append(d)
    # exp = dict(table_name='milad_mc_0013',
    #            base_index=0,
    #            db_url='sqlite:///data/MatrixCompletion.db3',
    #            multi_res=[{
    #                'm': [100, 200, 300, 400, 500],
    #                'n': [500],
    #                'snr': [round(p, 3) for p in np.linspace(1, 20, 20)],
    #                'p': [round(p, 3) for p in np.linspace(.1, 1, 19)],
    #                'mc': [round(p) for p in np.linspace(1, 20, 20)]
    #            }])
    exp = dict(table_name='milad_mc_0015',
               base_index=0,
               db_url='sqlite:///data/MatrixCompletion.db3',
               multi_res=[{
                   'm': [100, 200, 300, 400, 500],
                   'n': [500],
                   'snr': [round(p, 3) for p in np.linspace(1, 20, 20)],
                   'p': [round(p, 3) for p in np.linspace(.1, 1, 19)],
                   'mc': [round(p) for p in np.linspace(21, 100, 80)]
               }]
              )
    # add max_matrix_dim for having unified output size
    # 152k rows
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
            do_on_cluster(exp, do_matrix_completion, client, credentials=get_gbq_credentials())


def do_local_experiment():
    exp = test_experiment()
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, do_matrix_completion, client, credentials=get_gbq_credentials())

# def do_sherlock_experiment():
#     exp = test_experiment()
#     nodes = 200
#     with SLURMCluster(queue='normal,owners,donoho,hns,stat',
#                       cores=1, memory='4GiB', processes=1,
#                       walltime=’24:00:00') as cluster:
#         cluster.scale(jobs=nodes)
#         logging.info(cluster.job_script())
#         with Client(cluster) as client:
#             do_on_cluster(exp, do_matrix_completion, client, credentials=get_gbq_credentials())
#         cluster.scale(0)

def do_test():
    print(get_gbq_credentials())
    # exp = test_experiment()
    # import json
    # j_exp = json.dumps(exp, indent=4)
    # print(j_exp)
    # params = unroll_experiment(exp)
    # for p in params:
    #     df = do_matrix_completion(**p)
    #     print(df)
    pass
    # df = do_matrix_completion(m=100, n=100, snr=10., p=2./3., mc=20)
    # df = do_matrix_completion(m=12, n=8, snr=20., p=2./3., mc=20)
    # print(df)


if __name__ == "__main__":
    do_local_experiment()
    # do_sherlock_experiment()
    # do_coiled_experiment()
    # do_test()
