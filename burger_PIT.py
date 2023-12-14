from jax import grad, jit, vmap, hessian
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
import load_data_burger
importlib.reload(load_data_burger)
from load_data_burger import PDEDataSet
from dataloader_burger import Dataset, DataLoader

import pickle

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

seq = hk.PRNGSequence(42)

class Args():
    def __init__(self):
        # for data generation
        self.num_pdes = 1000
        self.num_train_pdes = 600 
        self.N_in = 250 
        self.pde_name = 'burger'
        self.nu = 0.01

        # for transformer
        self.pos_enc_dim = 12
        self.num_attn_blocks = 4
        self.num_transformer_blocks = 1
        self.key_size = 100
        self.ff_hidden_dim = 100
        self.embedding_dim = 100
        self.decoder_hidden_dim = 100

        # for loss
        self.sup_weight = 30
        self.pinn_weight = 100

        # for training
        self.lr = 1e-4
        self.batch_size = 100
        self.num_epochs = 4000

args = Args()

dataset = PDEDataSet(args)
Exact = dataset.Exact

# Initial conditions 
X_initial = dataset.X_initial 
u_initial = dataset.u_initial 

# Random locations to enforce PINN loss 
X_unlabeled = dataset.X_unlabeled 

# Labeled data: initial + boundary
X_labeled = dataset.X_labeled 
u_labeled = dataset.u_labeled

x = dataset.x
t = dataset.t
fd = dataset.full_domain

data_loader = DataLoader(Dataset(X_initial[:args.num_train_pdes], u_initial[:args.num_train_pdes], 
                                 X_labeled[:args.num_train_pdes], u_labeled[:args.num_train_pdes], 
                                 X_unlabeled[:args.num_train_pdes]), args.batch_size)

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

params = u_net.init(next(seq), jnp.ones((1,2)), jnp.ones((1,2)), jnp.ones((1,1)))

optimizer = optax.adam(args.lr)
opt_state = optimizer.init(params)

def loss(params, X_init, u_init, X_labeled, u_labeled, X_pinn):
    
    # Initial and boundary loss
    preds = u_net.apply(params, X_labeled, X_init, u_init)
    intial_and_boundary_loss = jnp.mean(jnp.square(preds - u_labeled))
    
    # PDE loss
    u = lambda X: u_net.apply(params, X, X_init, u_init).squeeze(-1)
    u_val = u(X_pinn)
    u_x = vmap(grad(u))(X_pinn)[..., 0]
    u_t = vmap(grad(u))(X_pinn)[..., 1]
    u_xx = vmap(hessian(u))(X_pinn)[..., 0, 0]
    
    pde = u_t + u_val * u_x - args.nu * u_xx # burgers equation
    
    pde_loss = jnp.mean(jnp.square(pde))

    return args.sup_weight * intial_and_boundary_loss + args.pinn_weight * pde_loss

def batch_loss(params, X_init, u_init, X_labeled, u_labeled, X_pinn):
    return jnp.mean(vmap(loss, in_axes=(None, 0, 0, 0, 0, 0))(params, X_init, u_init, X_labeled, u_labeled, X_pinn))

@jit
def update(params, opt_state, X_init, u_init, X_labeled, u_labeled, X_pinn):
    loss_value = batch_loss(params, X_init, u_init, X_labeled, u_labeled, X_pinn)
    grads = grad(batch_loss)(params, X_init, u_init, X_labeled, u_labeled, X_pinn)

    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, loss_value

def fit(params, opt_state):
    print("Training")
    l = []
    for epoch in range(args.num_epochs):
        batch_loss = []
        for X_init, u_init, X_labeled, u_labeled, X_pinn in data_loader:
            
            params, opt_state, loss_value = update(params, opt_state, X_init, u_init, 
                                                   X_labeled, u_labeled, X_pinn)
            batch_loss.append(loss_value)

        l.append(jnp.mean(jnp.array(batch_loss)))
        print("Epoch: {}, Loss: {}".format(epoch, l[-1]))
    return params, l

p_trained, l = fit(params, opt_state)

def pred(params, pde):
    X_init = X_initial[pde]
    u_init = u_initial[pde]
    return u_net.apply(params, fd, X_init, u_init).reshape((100, 256))

pde_num = 554
preds_transformer = pred(p_trained, pde_num)
exact = Exact[pde_num]

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(preds_transformer.T, cmap='jet')
plt.colorbar()
plt.title('Prediction')
plt.subplot(1, 2, 2)
plt.imshow(exact.T, cmap='jet')
plt.colorbar()
plt.title('Exact')
plt.show()

def save(state) -> None:
    with open("experiments/PIT/arrays_transformer_burger.npy", "wb") as f:
        for x in jax.tree_leaves(state):
            np.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_map(lambda t: 0, state)
    with open("experiments/PIT/tree_transformer_burger.pkl", "wb") as f:
        pickle.dump(tree_struct, f)

save(p_trained)
