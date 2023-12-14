import jax.numpy as jnp
from jax import random

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
            batch_X_labeled = jnp.array([self.dataset.X_labeled[idx] for idx in batch_indices])
            batch_u_labeled = jnp.array([self.dataset.u_labeled[idx] for idx in batch_indices])
            batch_X_pinn = jnp.array([self.dataset.X_pinn[idx] for idx in batch_indices])
            yield batch_X_init, batch_u_init, batch_X_labeled, batch_u_labeled, batch_X_pinn


class Dataset:
    def __init__(self, X_init, u_init, X_labeled, u_labeled, X_pinn):
        self.X_init = X_init
        self.u_init = u_init
        self.X_labeled = X_labeled
        self.u_labeled = u_labeled
        self.X_pinn = X_pinn

    def __len__(self):
        return self.u_init.shape[0]
    
    def __getitem__(self, index):
        return self.X_init[index], self.u_init[index], self.X_labeled[index], self.u_labeled[index], self.X_pinn[index]