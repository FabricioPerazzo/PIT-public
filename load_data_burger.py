import numpy as np
import jax.numpy as jnp
from pyDOE import lhs
from jax import random

class PDEDataSet:
    def __init__(self, args):
        self.args = args
        self.load_pde_data()
        print("loaded pde data")

        # generate train/test data
        X_initial = []
        u_initial = []
        X_labeled = []
        u_labeled = []
        X_unlabeled = []

        # randomly select N_i data points from from IC (t=0)
        # fix the random seed to always draw the same intial samples
        data_ic_rng = random.PRNGKey(0)
        idx_i = np.array(random.choice(data_ic_rng, len(self.x), (self.N_i,), replace=False))
        # randomly select N_b data points from from BC (x=0, x=L)
        idx_b = np.array(random.choice(random.PRNGKey(1), len(self.t)*2, (self.N_b,), replace=False))

        for i in range(args.num_pdes):
            grid_data = self.Exact[i]
            X_i, u_i, X_l, u_l, X_s = self.pre_process_shock_data(
                self.meshgrid, grid_data, idx_i, idx_b)

            # get labeled data inside the domain
            idx_t = np.random.choice(len(self.t), self.N_in_labeled, replace=True)
            idx_x = np.random.choice(len(self.x), self.N_in_labeled, replace=True)
            t_rnd = self.t[idx_t]
            x_rnd = self.x[idx_x]
            X_labeled_inside = np.vstack((x_rnd, t_rnd)).T
            u_labeled_inside = grid_data[idx_t, idx_x][:, None]

            # concatenate inside samples to initial and boundary samples
            X_l = np.concatenate((X_i, X_l, X_labeled_inside), axis=0)
            u_l = np.concatenate((u_i, u_l, u_labeled_inside), axis=0)

            X_initial.append(X_i)
            u_initial.append(u_i)
            X_labeled.append(X_l)
            u_labeled.append(u_l)
            X_unlabeled.append(X_s)

        # convert list to arrays
        self.X_initial = np.array(X_initial)
        self.u_initial = np.array(u_initial)
        self.X_labeled = np.array(X_labeled)
        self.u_labeled = np.array(u_labeled)
        self.X_unlabeled = np.array(X_unlabeled)
        print("generated train data")


    def load_pde_data(self):
        args = self.args
        # load data
        N_i = 200           # number of data points at t=0          #200
        N_b = 200           # number of data points on the boundary #200
        N_in = args.N_in    # number of data points in the interior
        N_b_unlabeled = 0   # number of random samples from the boundary #0
        # to include in unlabeled set
        N_in_labeled = 0    # number of labeled data points in the interior

        # load data
        data_file = 'burgers_full_mat_fs_simpler.npy'

        pde_full_mat = jnp.load(data_file, allow_pickle=True).item()
        t = pde_full_mat['t']
        x = pde_full_mat['x']
        args.nu = pde_full_mat['nu']
        Exact = pde_full_mat['usol']

        X, T = np.meshgrid(x, t)
        full_domain = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        meshgrid = np.array((X, T))

        # Domain bounds
        lb = full_domain.min(0)
        ub = full_domain.max(0)

        # normalize data
        if args.pde_name == 'heat':
            scaling_factor = np.std(Exact, axis=0)[None]
            Exact = Exact / scaling_factor
            print('scaled data')

        self.full_domain = full_domain
        self.lb = lb
        self.ub = ub
        self.N_b_unlabeled = N_b_unlabeled
        self.N_i = N_i
        self.N_b = N_b
        self.N_in_labeled = N_in_labeled
        self.x = x
        self.t = t
        self.Exact = Exact
        self.meshgrid = meshgrid



    def random_sample_unlabeled(self, X_unlabeled):
        # sample random points inside the domain
        N_in = self.args.N_in
        batch_size = X_unlabeled.shape[0]
        X_samples = self.lb + (self.ub-self.lb)*lhs(2, N_in*batch_size)
        X_samples = X_samples.reshape((batch_size, N_in, 2))

        # sample N_b_unlabeled from the unlabeled data
        idx = np.random.choice(len(self.x)+len(self.t)*2, self.N_b_unlabeled, replace=False)
        X_unlabeled = X_unlabeled[:, idx]

        X_unlabeled = np.concatenate((X_samples, X_unlabeled), axis=1)
        # rng = np.random.default_rng()
        # rng.shuffle(X_unlabeled, axis=1)
        return X_unlabeled


    def pre_process_shock_data(self, meshgrid, Exact, idx_initial, idx_boundary):
        X, T = meshgrid[0], meshgrid[1]

        xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
        uu1 = Exact[0:1, :].T
        xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
        uu2 = Exact[:, 0:1]
        xx3 = np.hstack((X[:, -1:], T[:, -1:]))
        uu3 = Exact[:, -1:]

        X_initial = xx1
        u_initial = uu1

        X_boundary = np.vstack([xx2, xx3])
        u_boundary = np.vstack([uu2, uu3])

        # X_samples = lb + (ub-lb)*lhs(2, N_f)
        # X_samples = np.vstack((X_samples, X_boundary))
        X_samples = np.vstack([xx1, xx2, xx3])
        X_samples = self.random_sample_unlabeled(X_samples[None]).squeeze()

        # randomly sample from the initial and boundary data
        X_boundary = X_boundary[idx_boundary, :]
        u_boundary = u_boundary[idx_boundary, :]

        X_initial = X_initial[idx_initial, :]
        u_initial = u_initial[idx_initial, :]

        return X_initial, u_initial, X_boundary, u_boundary, X_samples
