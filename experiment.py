#!/usr/bin/env python3

import cvxpy as cp
import numpy as np
from numpy.random import Generator
from cvxpy.atoms import normNuc, multiply, norm
from pandas import DataFrame
from scipy import stats as st
from sklearn.utils.extmath import randomized_svd

from EMS.manager import active_remote_engine, do_on_cluster, unroll_experiment
from dask.distributed import Client, LocalCluster
import logging

logging.basicConfig(level=logging.INFO)

def seed(n: int, snr: float, p: float, mc: int) -> int:
    return round(1 + n * 1000 + round(snr * 1000) + round(p * 1000) + mc * 100000)


def _df(c: list, l: list) -> DataFrame:
    d = dict(zip(c, l))
    return DataFrame(data=d, index=[0])


def df_experiment(n: int, snr: float, p: float, mc: int, t: float, cos_l: float, cos_r: float) -> DataFrame:
    c = ['n', 'snr', 'p', 'mc', 't', 'cosL', 'cosR']
    d = [n, snr, p, mc, t, cos_l, cos_r]
    return _df(c, d)


def suggested_t(observed, n):
    return np.sqrt(np.sum(observed) / n)


def make_data(n: int, p: float) -> tuple:
    u, v = np.random.normal(size=(2, n))
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    M = np.outer(u, v)

    noise = np.random.normal(0, 1 / np.sqrt(n), (n, n))

    observes = st.bernoulli.rvs(p, size=(n, n))

    return u, v, M, noise, observes


def make_observe_index(folderadd, n, plist):
    observes = [0] * len(plist)

    ind = -1
    for p in plist:
        ind += 1
        observes[ind] = st.bernoulli.rvs(p, size=(n, n))

    header = 'plist\n' + str(plist)
    add = folderadd + 'obs_indx.txt'
    write_list(add, header, observes, fmt='%d')

    return observes


def make_primary_data(n, root=''):
    u, v = np.random.normal(size=(2, n))
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    M = np.outer(u, v)

    noise = np.random.normal(0, 1 / np.sqrt(n), (n, n))

    plist = np.linspace(0, 1, 11)
    snrlist = np.linspace(1, 10, 10)

    folderadd = root + 'n=' + str(n)
    if path.exists(folderadd):
        folderadd += '_' + get_time()
    folderadd += '/'
    mkdir(folderadd)

    header = 'n,u,v,M,noise,plist,snrlist'
    add = folderadd + 'setup.txt'
    write_list(add=add, header=header, towritelist=[n, u, v, M, noise, plist, snrlist])

    observes = make_observe_index(folderadd=folderadd, n=n, plist=plist)

    return u, v, M, noise, plist, snrlist, observes, folderadd


# problem setup
def nuc_norm_problem(Y, observed, t) -> tuple:
    X = cp.Variable(Y.shape)
    objective = cp.Minimize(normNuc(X))
    Z = multiply(X - Y, observed)
    constraints = [Z == 0] if t == 0. else [norm(Z, "fro") <= t]

    prob = cp.Problem(objective, constraints)

    prob.solve()

    return X, prob


# measurements
def veccos(v, vhat):
    return np.abs(np.inner(v, vhat))


def take_measurements(Mhat, u, v):
    uhat, snrhat, vhat = randomized_svd(Mhat, n_components=1)
    uhat = uhat.T

    cosL = veccos(u, uhat)
    cosR = veccos(v, vhat)

    return cosL, cosR


def do_matrix_completion(*, n: int, snr: float, p: float, mc: int, tmethod='0') -> DataFrame:
    rng = np.random.default_rng(seed=seed(n, snr, p, mc))

    u, v, M, noise, obs = make_data(n, p)
    t = 0. if tmethod == '0' else suggested_t(observed=obs, n=n)
    Y = snr * M + noise
    X, _ = nuc_norm_problem(Y=Y, observed=obs, t=t)
    Mhat = X.value

    cos_l, cos_r = take_measurements(Mhat, u, v)

    return df_experiment(n, snr, p, mc, t, cos_l, cos_r)


def nucnormproblem(Y, observed, t):
    X = cp.Variable(Y.shape)
    objective = cp.Minimize(normNuc(X))
    Z = multiply(X - Y, observed)
    if t == 0:
        # print('t=0 is running')
        constraints = [Z == 0]
    elif t > 0:
        constraints = [norm(Z, "fro") <= t]

    prob = cp.Problem(objective, constraints)

    start = time()
    prob.solve()
    runtime = time() - start

    # print("runtime: {:.4f}".format(runtime))

    return X, prob


def get_measurements(n, root='', tmethod='0'):
    start = time()
    print('n=', n, ' started at ' + get_time())
    u, v, M, noise, plist, snrlist, observes, folderadd = make_primary_data(n, root=root)

    add = folderadd + 'measurements.txt'
    header = 'n, snr, p, t, cosL, cosR'
    with open(add, "w+") as f:
        f.write(header + '\n\n')

    for snr in snrlist:
        ind = -1
        for p in plist:
            ind += 1
            obs = observes[ind]
            if tmethod == '0':
                t = 0
            elif tmethod == 'var':
                t = suggested_t(observed=obs, n=n)

            Y = snr * M + noise

            X, _ = nucnormproblem(Y=Y, observed=obs, t=t)
            Mhat = X.value

            cosL, cosR = take_measurements(Mhat, u, v)

            towrite = np.array([[n, snr, p, t, cosL, cosR]], dtype=object)
            with open(add, "ab") as f:
                np.savetxt(f, towrite, fmt="%.6f", delimiter=',')

    duration = time() - start
    minutes, seconds = divmod(duration, 60)
    print('run time {:d}:{:d}'.format(int(minutes), int(seconds)))


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
    exp = dict(table_name='mc-0002',
               base_index=9853,
               db_url='sqlite:///data/MatrixCompletion.db3',
               multi_res=[{
                   'n': [round(p) for p in np.linspace(10, 500, 21)],
                   'snr': [round(p, 3) for p in np.linspace(1, 20, 39)],
                   'p': [round(p, 3) for p in np.linspace(0., 1., 41)],
                   'mc': [20]
               }])
    return exp


def do_local_experiment():
    exp = test_experiment()
    with LocalCluster(dashboard_address='localhost:8787') as cluster:
        with Client(cluster) as client:
            do_on_cluster(exp, do_matrix_completion, client)


def do_test():
    exp = test_experiment()
    print(exp)
    params = unroll_experiment(exp)
    pass
    # for p in params:
    #     df = do_matrix_completion(**p)
    #     print(df)
    # df = do_matrix_completion(n=10, snr=1., p=0., mc=0)
    # print(df)


if __name__ == "__main__":
    do_local_experiment()
    # do_test()
