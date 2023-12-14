from jax import grad, jit, vmap, hessian, jvp
import jax.numpy as jnp
from jax import random
from jax import nn
import jax

import haiku as hk
import optax

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

import importlib
import dataloader_heat2D
importlib.reload(dataloader_heat2D)
from dataloader_heat2D import PDEDataset, Dataset, DataLoader

import pickle
import sys

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

seq = hk.PRNGSequence(42)

save_name = sys.argv[1]
pinn_weight = int(sys.argv[2]) # 150
ic_weight = int(sys.argv[3]) # 80
bc_weight = int(sys.argv[4]) # 60

class Args():
    def __init__(self):
        # for data generation
        self.num_pdes = 1000
        self.num_train_pdes = 800 
        self.N_pinn = 1 # Sampling them later
        self.N_b = 50
        self.N_i = 64**2 # All of them, sample N_i later
        self.resol = 64
        self.pde_name = 'heat2D'
        self.nu = 0.05

        # for transformer
        self.pos_enc_dim = 12
        self.num_attn_blocks = 4
        self.num_transformer_blocks = 1
        self.key_size = 100
        self.ff_hidden_dim = 100
        self.embedding_dim = 100
        self.decoder_hidden_dim = 100

        # for loss
        self.init_cond_weight = ic_weight
        self.bound_cond_weight = bc_weight
        self.pinn_weight = pinn_weight

        # for training
        self.lr = 1e-4
        self.batch_size = 50
        self.num_epochs = 5000

args = Args()

dataset = PDEDataset(args)
dataset.load()

# Exact solution
Exact = dataset.Exact

# Initial conditions 
X_initial = dataset.X_initial 
u_initial = dataset.u_initial

# Boundary conditions
X_boundary = dataset.X_boundary

# Random locations to enforce PINN loss 
X_unlabeled = dataset.X_unlabeled 

x, y, t = dataset.x, dataset.y, dataset.t
fd = dataset.fd

data_loader = DataLoader(Dataset(X_initial[:args.num_train_pdes], u_initial[:args.num_train_pdes], 
                                 X_boundary[:args.num_train_pdes], X_unlabeled[:args.num_train_pdes]), 
                                 args.batch_size)

# For sampling
args.N_pinn = 400
args.N_i = 400

def pos_enc(positions, d_model):
   
    positions = jnp.asarray(positions)
    *leading_dims, n_positions = positions.shape

    # Create an array of dimensions
    dimensions = jnp.expand_dims(jnp.arange(d_model), axis=0)  
    positions = jnp.expand_dims(positions, axis=-1)

    # Apply the formula to compute the positional encoding
    angle_rates = jnp.power(10000, (-2 * (dimensions // 2)) / jnp.float32(d_model))
    
    positional_encoding = jnp.zeros((*leading_dims, n_positions, d_model))

    # apply sin to even indices in the array; 2i
    positional_encoding = positional_encoding.at[..., 0::2].set(jnp.sin(positions * angle_rates[:, 0::2]))

    # apply cos to odd indices in the array; 2i+1
    positional_encoding = positional_encoding.at[..., 1::2].set(jnp.cos(positions * angle_rates[:, 1::2]))

    return positional_encoding

def _layer_norm(x: jax.Array) -> jax.Array:
  """Applies a unique LayerNorm to `x` with default settings."""
  ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
  return ln(x)

class CrossAttention(hk.Module):

    def __init__(self, 
                 output_size: int,
                 key_size: int,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.output_size = output_size
        self.key_size = key_size
        self.W_q = hk.nets.MLP([self.key_size, self.key_size, self.key_size], activation=jnp.sin)
        self.W_k = hk.nets.MLP([self.key_size, self.key_size, self.key_size], activation=jnp.sin)
        self.W_vq = hk.nets.MLP([self.key_size, self.key_size, self.key_size], activation=jnp.sin)
        self.W_vk = hk.nets.MLP([self.key_size, self.key_size, self.key_size], activation=jnp.sin)
        self.W_final_proj = hk.nets.MLP([self.key_size, self.key_size, self.key_size], activation=jnp.sin)

    def __call__(self, 
                 X_query: jax.Array,
                 X_init: jax.Array,
                 ) -> jax.Array:
        
        Q = self.W_q(X_query)
        K = self.W_k(X_init)
        
        A = jnp.matmul(Q, K.T) / jnp.sqrt(self.key_size)
        A = nn.softmax(A, axis=-1)

        V_query = self.W_vq(X_query)
        V_init = self.W_vk(X_init)

        h_attn = jnp.matmul(A, V_init) + V_query
        h_final_proj = self.W_final_proj(h_attn)

        return h_final_proj

class TransformerBlock(hk.Module):

    def __init__(self, num_attn_blocks: int, 
                key_size: int,
                widening_factor: Optional[int] = 4,
                num_self_attn_blocks: Optional[int] = 4, 
                name: Optional[str] = None):
        super().__init__(name=name)
        self.key_size = key_size
        self.num_attn_blocks = num_attn_blocks
        self.widening_factor = widening_factor
        self.num_self_attn_blocks = num_self_attn_blocks

    def __call__(self, 
                 X_query: jax.Array,
                 X_init: jax.Array,
                 ) -> jax.Array:
        
        h_X = X_query
        *leading_dims, embedding_dim = h_X.shape

        for _ in range(self.num_attn_blocks):

            cross_attn_block = CrossAttention(embedding_dim, self.key_size)
            dense_block = hk.Sequential([
                hk.Linear(self.widening_factor * embedding_dim),
                nn.elu,
                hk.Linear(embedding_dim),
            ])

            h_X_norm = _layer_norm(h_X)
            h_X_attn = cross_attn_block(h_X_norm, X_init)

            h_X = h_X_norm + h_X_attn
            h_X_dense = dense_block(h_X)
            h_X = h_X + h_X_dense

        return h_X

def u_net(X_query, X_init, u_init):
    
    # Step 1: Positional encoding
    X_query = pos_enc(X_query, args.pos_enc_dim)
    *leading_dims, num_indep_vars, embedding_dim = X_query.shape
    X_query = X_query.reshape((*leading_dims, num_indep_vars*embedding_dim))

    X_init = pos_enc(X_init, args.pos_enc_dim)
    *leading_dims, num_indep_vars, embedding_dim = X_init.shape
    X_init = X_init.reshape((*leading_dims, num_indep_vars*embedding_dim))
    
    X_init = jnp.concatenate([X_init, u_init], axis=-1)

    # Step 2: Get embeddings
    embedding_net_layers = [args.ff_hidden_dim, args.ff_hidden_dim, args.ff_hidden_dim, args.embedding_dim]
    X_query_embedding_net = hk.nets.MLP(embedding_net_layers, activation=jnp.sin)
    X_init_embedding_net = hk.nets.MLP(embedding_net_layers, activation=jnp.sin)
    
    X_query = X_query_embedding_net(X_query)
    X_init = X_init_embedding_net(X_init)
    
    # Step 3: Transformer blocks  
    for _ in range(args.num_transformer_blocks):
        transformer_block = TransformerBlock(args.num_attn_blocks, args.key_size)
        X_query = transformer_block(X_query, X_init)
        
    # Step 4: Pointwise feedforward network
    decoder_layers = [args.decoder_hidden_dim, args.decoder_hidden_dim, 1]
    decoder = hk.nets.MLP(decoder_layers, activation=jnp.sin)
    u_query = decoder(X_query)
    
    return u_query

u_net = hk.without_apply_rng(hk.transform(u_net))

params = u_net.init(next(seq), jnp.ones((1,3)), jnp.ones((1,3)), jnp.ones((1,1)))

optimizer = optax.adam(args.lr)
opt_state = optimizer.init(params)

def loss(params, X_init, u_init, X_boundary, X_pinn):
    u = lambda X: u_net.apply(params, X, X_init, u_init)

    # Initial condition loss
    preds = u(X_init)
    intial_cond_loss = jnp.mean(jnp.square(preds - u_init))
    
    # Boundary condition loss
    x_0, x_L = X_boundary[:args.N_b, :], X_boundary[args.N_b:2*args.N_b, :]
    y_0, y_L = X_boundary[2*args.N_b:3*args.N_b, :], X_boundary[3*args.N_b:, :]

    x_b_loss = jnp.mean(jnp.square(u(x_0) - u(x_L)))
    y_b_loss = jnp.mean(jnp.square(u(y_0) - u(y_L)))
    bound_cond_loss = (x_b_loss + y_b_loss) / 2

    # PDE loss
    u = lambda X: u_net.apply(params, X, X_init, u_init).squeeze(-1)
    u_t = vmap(grad(u))(X_pinn)[..., 2]

    # hessian-vector product
    def hvp_fwdrev(f, primals, tangents):
        g = lambda primals: jvp(f, primals, tangents)[1]
        return grad(g)(primals)

    u_xx = vmap(hvp_fwdrev, in_axes=(None, 0, None))(u, (X_pinn,), (jnp.array([1.,0.,0.]),))[0][..., 0]
    u_yy = vmap(hvp_fwdrev, in_axes=(None, 0, None))(u, (X_pinn,), (jnp.array([0.,1.,0.]),))[0][..., 1]
    
    pde = u_t - args.nu * (u_xx + u_yy) # heat equation
    
    pde_loss = jnp.mean(jnp.square(pde))

    return args.init_cond_weight * intial_cond_loss + args.bound_cond_weight * bound_cond_loss  + args.pinn_weight * pde_loss

def batch_loss(params, X_init, u_init, X_boundary, X_pinn):
    return jnp.mean(vmap(loss, in_axes=(None, 0, 0, 0, 0))(params, X_init, u_init, X_boundary, X_pinn))

@jit
def update(params, opt_state, X_init, u_init, X_boundary, X_pinn):
    loss_value = batch_loss(params, X_init, u_init, X_boundary, X_pinn)
    grads = grad(batch_loss)(params, X_init, u_init, X_boundary, X_pinn)

    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, loss_value

def sample_X_pinn(N_pinn, batch_size):
    x_unlabeled = jnp.array(np.random.uniform(0, 1, (batch_size, N_pinn, 1)))
    y_unlabeled = jnp.array(np.random.uniform(0, 1, (batch_size, N_pinn, 1)))
    t_unlabeled = jnp.array([np.random.choice(t, N_pinn) for _ in range(batch_size)])[..., None]
    return jnp.concatenate([x_unlabeled, y_unlabeled, t_unlabeled], axis=-1)

def sample_X_init_and_u_init(N_i, batch_size, u0):
    x, y = jnp.linspace(0, 1, args.resol), jnp.linspace(0, 1, args.resol)
    X_initial = jnp.meshgrid(x, y)
    X_initial = jnp.stack(X_initial, axis=-1)
    X_initial = X_initial.reshape(-1, 2)[None, ...].repeat(batch_size, axis=0)
    idx = random.choice(random.PRNGKey(0), X_initial.shape[1], (N_i,), replace=False)
    X_initial = X_initial[:, idx, :]
    X_initial = jnp.concatenate([X_initial, jnp.zeros((batch_size, N_i, 1))], axis=-1)
    u_initial = u0[:, idx, :]
    return X_initial, u_initial

def fit(params, opt_state):
    print("Training")
    l = []
    for epoch in range(args.num_epochs):
        batch_loss = []
        for _, u_init_all, X_boundary, _ in data_loader:
            
            X_pinn = sample_X_pinn(args.N_pinn, args.batch_size)
            X_init, u_init = sample_X_init_and_u_init(args.N_i, args.batch_size, u_init_all)

            params, opt_state, loss_value = update(params, opt_state, X_init, u_init, 
                                                   X_boundary, X_pinn)
            batch_loss.append(loss_value)

        l.append(jnp.mean(jnp.array(batch_loss)))
        print("Epoch: {}, Loss: {}".format(epoch, l[-1]))
    return params, l

p_trained, l = fit(params, opt_state)

def pred(params, pde):
    X_init = X_initial[pde]
    u_init = u_initial[pde]
    return u_net.apply(params, fd, X_init, u_init).reshape((64, 64, 50))

pde_num = 900
preds = pred(p_trained, pde_num)
exact = Exact[pde_num]

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for i, t in enumerate([0, 10, 20, 30, 40]):
    axs[0, i].imshow(preds[:, :, t], cmap='jet')
    axs[0, i].set_title('Prediction at t = {}'.format(t))
    axs[1, i].imshow(exact[:, :, t], cmap='jet')
    axs[1, i].set_title('Exact at t = {}'.format(t))

plt.show()

# Relative L2 error for test data
def relative_L2_error(params):
    rel_error = []
    for pde in range(args.num_train_pdes, args.num_pdes):
        preds = pred(params, pde)
        exact = Exact[pde]
        rel_error.append(jnp.linalg.norm(preds - exact) / jnp.linalg.norm(exact))
    error = jnp.linalg.norm(preds - exact) / jnp.linalg.norm(exact)
    return jnp.mean(jnp.array(error)), jnp.std(jnp.array(error))

error, std = relative_L2_error(p_trained)

# Save error, std in text file
with open('experiments/PIT/error.txt', 'a') as f:
    f.write('Relative L2 error: {}\n'.format(error))
    f.write('Standard deviation: {}\n'.format(std))

f.close()

def save(state) -> None:
    with open('experiments/PIT/arrays_transformer_heat2D' + save_name + '.npy', 'wb') as f:
        for x in jax.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open('experiments/PIT/tree_transformer_heat2D' + save_name + '.pkl', 'wb') as f:
        pickle.dump(tree_struct, f)

save(p_trained)