#!/usr/bin/env python3

from EMS.manager import active_remote_engine, do_on_cluster
from dask.distributed import Client, LocalCluster
from numpy.random import Generator, PCG64
from pandas import DataFrame
import numpy as np
from scipy import stats as st
import cvxpy as cp
from cvxpy.atoms import normNuc, multiply, norm
from sklearn.utils.extmath import randomized_svd


from cvxpy.expressions import constants
from cvxpy.expressions.cvxtypes import constant
# make data

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

def suggested_t(observed, n):
  return np.sqrt(np.sum(observed) / n)

def make_primary_data(n, root=''):

  u, v = np.random.normal(size=(2, n))
  u /= np.linalg.norm(u)
  v /= np.linalg.norm(v)

  M = np.outer(u, v)

  noise = np.random.normal(0, 1/np.sqrt(n), (n, n))

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
def nucnormproblem(Y, observed, t):
  X = cp.Variable(Y.shape)
  objective = cp.Minimize(normNuc(X))
  Z = multiply(X - Y, observed)
  if t==0:
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


# measurements
def veccos(v, vhat):
  return np.abs(np.inner(v, vhat))


def take_measurements(Mhat, u, v):

  uhat, snrhat, vhat = randomized_svd(Mhat, n_components=1)
  uhat = uhat.T

  cosL = veccos(u, uhat)
  cosR = veccos(v, vhat)

  return cosL, cosR


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
