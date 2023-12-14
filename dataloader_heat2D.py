import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import pickle

class PDEDataset():

    def __init__(self, args):
        self.num_pdes = args.num_pdes
        self.N_pinn = args.N_pinn
        self.N_b = args.N_b
        self.N_i = args.N_i
        self.resol = args.resol

    def load(self):
        dataset = np.load('heat2D_r64_t50.npy', allow_pickle=True).item()
        u0 = jnp.array(dataset['u_init'])[..., None] # (1000 x 64 x 64 x 1)
        u = jnp.array(dataset['u']) # (1000 x 64 x 64 x 50 x 1)
        total_pde_num = u.shape[0]
        
        self.Exact = u[:self.num_pdes]

        x, y = jnp.linspace(0, 1, self.resol), jnp.linspace(0, 1, self.resol)

        X_initial = jnp.meshgrid(x, y)
        X_initial = jnp.stack(X_initial, axis=-1)
        X_initial = X_initial.reshape(-1, 2)[None, ...].repeat(total_pde_num, axis=0)
        X_initial = jnp.concatenate([X_initial, jnp.zeros((total_pde_num, self.N_i, 1))], axis=-1)

        self.X_initial = X_initial[:self.num_pdes]

        u_initial = u0.reshape(total_pde_num, self.resol**2, 1)
        self.u_initial = u_initial[:self.num_pdes]

        x_b = jnp.array(np.random.uniform(0, 1, (total_pde_num, self.N_b, 1)))
        y_b = jnp.array(np.random.uniform(0, 1, (total_pde_num, self.N_b, 1)))
        t_b = jnp.array(np.random.uniform(0, 1, (total_pde_num, self.N_b, 1)))

        x_0 = (x[None, 0].repeat(x_b.shape[1], axis=0))[None, ...].repeat(total_pde_num, axis=0)[..., None]
        x_L = (x[None, -1].repeat(x_b.shape[1], axis=0))[None, ...].repeat(total_pde_num, axis=0)[..., None]

        X_b_0 = jnp.concatenate([x_0, y_b, t_b], axis=-1)
        X_b_L = jnp.concatenate([x_L, y_b, t_b], axis=-1)

        Y_b_0 = jnp.concatenate([x_b, x_0, t_b], axis=-1)
        Y_b_L = jnp.concatenate([x_b, x_L, t_b], axis=-1)

        X_b = jnp.concatenate([X_b_0, X_b_L], axis=1)
        Y_b = jnp.concatenate([Y_b_0, Y_b_L], axis=1)

        X_boundary = jnp.concatenate([X_b, Y_b], axis=1)
        self.X_boundary = X_boundary[:self.num_pdes]

        x_unlabeled = jnp.array(np.random.uniform(0, 1, (total_pde_num, self.N_pinn, 1)))
        y_unlabeled = jnp.array(np.random.uniform(0, 1, (total_pde_num, self.N_pinn, 1)))
        t_unlabeled = jnp.array(np.random.uniform(0, 1, (total_pde_num, self.N_pinn, 1)))

        X_unlabeled = jnp.concatenate([x_unlabeled, y_unlabeled, t_unlabeled], axis=-1)
        self.X_unlabeled = X_unlabeled[:self.num_pdes]

        t = jnp.linspace(0.02, 1, u.shape[3])

        self.x, self.y, self.t = x, y, t

        fd = jnp.meshgrid(x, y, t)
        fd = jnp.stack(fd, axis=-1).reshape(-1, 3)
        self.fd = fd

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        self.num_batches = len(dataset) // batch_size

    def __iter__(self):
        if self.shuffle:
            perm = random.permutation(random.PRNGKey(0), self.num_samples)
        else:
            perm = jnp.arange(self.num_samples)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = perm[i:i + self.batch_size]
            batch_X_init = jnp.array([self.dataset.X_init[idx] for idx in batch_indices])
            batch_u_init = jnp.array([self.dataset.u_init[idx] for idx in batch_indices])
            batch_X_boundary = jnp.array([self.dataset.X_boundary[idx] for idx in batch_indices])
            batch_X_pinn = jnp.array([self.dataset.X_pinn[idx] for idx in batch_indices])
            yield batch_X_init, batch_u_init, batch_X_boundary, batch_X_pinn


class Dataset:
    def __init__(self, X_init, u_init, X_boundary, X_pinn):
        self.X_init = X_init
        self.u_init = u_init
        self.X_boundary = X_boundary
        self.X_pinn = X_pinn

    def __len__(self):
        return self.u_init.shape[0]
    
    def __getitem__(self, index):
        return self.X_init[index], self.u_init[index], self.X_boundary[index], self.X_pinn[index]