# Efficient WGAN-GP training script for time-series (vectorized, batched, tf.function)
# Assumptions: running in environment where TensorFlow GPU is available.
# Reads 'syn_norm_subset_2000_bins.csv' with index_col 'time_bin' and column 'syn_norm'.

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # hide INFO logs

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import wasserstein_distance

# --------- Config / Hyperparameters ---------
CSV_PATH = "syn_norm_subset_2000_bins.csv"
WINDOW_SIZE = 30
LATENT_DIM = 8
BATCH_SIZE = 32
EPOCHS = 2000
N_CRITIC = 5
GP_WEIGHT = 10.0
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 20
SEED = 1234

tf.random.set_seed(SEED)

# --------- Utility functions ---------

def make_windows_from_series(series, window_size, stride=1):
    """Return a 2D numpy array of sliding windows shape (num_windows, window_size)."""
    series = np.asarray(series, dtype=np.float32)
    if len(series) < window_size:
        return np.empty((0, window_size), dtype=np.float32)
    # use numpy stride_tricks via tf.signal.frame for simplicity
    tensor = tf.signal.frame(series, window_size, stride)
    return tensor.numpy()

# gradient penalty vectorized

def gradient_penalty(critic, real, fake):
    # real, fake: [batch, window_size]
    batch = tf.shape(real)[0]
    alpha = tf.random.uniform([batch, 1], 0.0, 1.0, dtype=real.dtype)
    interpolated = alpha * real + (1.0 - alpha) * fake
    # ensure shape [batch, window_size, 1] for Conv1D
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        interpolated_input = tf.expand_dims(interpolated, -1)
        pred = critic(interpolated_input)
    grads = gp_tape.gradient(pred, interpolated)
    grads = tf.reshape(grads, [batch, -1])
    grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
    gp = tf.reduce_mean((grads_norm - 1.0) ** 2)
    return gp

# distribution distance (kept for monitoring) -- uses numpy

def distribution_distance(real, fake):
    # small function to compute Wasserstein distance on histograms
    real = np.asarray(real).ravel()
    fake = np.asarray(fake).ravel()
    bin_edges = np.linspace(-1, 1, num=50)
    empirical_real, _ = np.histogram(real, bins=bin_edges, density=True)
    empirical_fake, _ = np.histogram(fake, bins=bin_edges, density=True)
    if empirical_real.sum() > 0:
        empirical_real = empirical_real / empirical_real.sum()
    if empirical_fake.sum() > 0:
        empirical_fake = empirical_fake / empirical_fake.sum()
    return wasserstein_distance(empirical_real, empirical_fake)

# --------- Model definitions (use float32)

def build_critic(window_length):
    inputs = tf.keras.Input(shape=(window_length, 1), dtype=tf.float32)
    x = tf.keras.layers.Conv1D(64, kernel_size=10, padding='same')(inputs)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=10, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=10, padding='same')(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=out, name='critic')


def build_generator(latent_dim, window_length):
    inputs = tf.keras.Input(shape=(latent_dim,), dtype=tf.float32)
    x = tf.keras.layers.Dense(30)(inputs)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dense(50)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dense(50)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(window_length)(x)
    # output shape [batch, window_length]
    return tf.keras.Model(inputs=inputs, outputs=x, name='generator')

# --------- Data loading

df = pd.read_csv(CSV_PATH, index_col='time_bin')
series = df['syn_norm'].values.astype(np.float32)
print('Series length:', len(series))

windows = make_windows_from_series(series, WINDOW_SIZE, stride=1)
if windows.shape[0] == 0:
    raise ValueError('Not enough data to create a single window.');

# normalize already expected to be in [-1,1] based on original code. Keep dtype float32.

dataset = tf.data.Dataset.from_tensor_slices(windows)
# shuffle sufficiently and batch
dataset = dataset.shuffle(buffer_size=min(10000, windows.shape[0]), seed=SEED).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# create one-shot iterator
train_iterator = iter(dataset.repeat())
num_elements = windows.shape[0]
print('Number of windows:', num_elements)

# --------- Instantiate models and optimizers

critic = build_critic(WINDOW_SIZE)
generator = build_generator(LATENT_DIM, WINDOW_SIZE)

c_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5, beta_2=0.9)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.5, beta_2=0.9)

# checkpoint manager
ckpt = tf.train.Checkpoint(critic=critic, generator=generator, c_optimizer=c_optimizer, g_optimizer=g_optimizer)
manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=5)

# --------- Training step functions (vectorized)

@tf.function
def critic_train_step(real_batch):
    # real_batch: [B, window]
    batch = tf.shape(real_batch)[0]
    real_batch = tf.cast(real_batch, tf.float32)
    noise = tf.random.normal([batch, LATENT_DIM], dtype=tf.float32)
    with tf.GradientTape() as tape:
        fake = generator(noise, training=True)  # [B, window]
        # Critic expects shape [B, window, 1]
        real_in = tf.expand_dims(real_batch, -1)
        fake_in = tf.expand_dims(fake, -1)
        real_out = critic(real_in, training=True)
        fake_out = critic(fake_in, training=True)
        # Wasserstein critic loss
        c_loss = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
        gp = gradient_penalty(critic, real_batch, fake)
        loss = c_loss + GP_WEIGHT * gp
    grads = tape.gradient(loss, critic.trainable_variables)
    c_optimizer.apply_gradients(zip(grads, critic.trainable_variables))
    return loss, c_loss, gp

@tf.function
def generator_train_step(batch_size):
    noise = tf.random.normal([batch_size, LATENT_DIM], dtype=tf.float32)
    with tf.GradientTape() as tape:
        fake = generator(noise, training=True)
        fake_in = tf.expand_dims(fake, -1)
        fake_out = critic(fake_in, training=True)
        g_loss = -tf.reduce_mean(fake_out)
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
    return g_loss

# --------- Training loop (keeps same logical algorithm but optimized)

start_time = time.time()
critic_loss_history = []
generator_loss_history = []
emd_history = []
steps_per_epoch = max(1, num_elements // BATCH_SIZE)
print('Steps per epoch (approx):', steps_per_epoch)

for epoch in range(1, EPOCHS + 1):
    epoch_c_loss = 0.0
    epoch_gp = 0.0
    epoch_g_loss = 0.0
    # For each epoch, do steps_per_epoch iterations but keep n_critic updates per generator step
    for step in range(steps_per_epoch):
        # perform n_critic critic updates
        for _ in range(N_CRITIC):
            real_batch = next(train_iterator)  # shape [B, window]
            loss_val, c_w_loss, gp_val = critic_train_step(real_batch)
            epoch_c_loss += c_w_loss.numpy()
            epoch_gp += gp_val.numpy()
        # generator update
        g_loss_val = generator_train_step(BATCH_SIZE)
        epoch_g_loss += g_loss_val.numpy()

    # average over steps
    epoch_c_loss /= (steps_per_epoch * N_CRITIC)
    epoch_gp /= (steps_per_epoch * N_CRITIC)
    epoch_g_loss /= steps_per_epoch

    critic_loss_history.append(epoch_c_loss)
    generator_loss_history.append(epoch_g_loss)

    # monitoring: generate enough samples to compute EMD against the original distribution
    # generate num_samples ~ length/window
    num_samples = max(1, len(series) // WINDOW_SIZE)
    noise = tf.random.normal([num_samples, LATENT_DIM], dtype=tf.float32)
    gen_batch = generator.predict(noise, verbose=0)
    generated_data = gen_batch.reshape(-1)
    emd = distribution_distance(series[:len(generated_data)], generated_data)
    emd_history.append(emd)

    if epoch % 2 == 0 or epoch == 1:
        print(f'Epoch {epoch}/{EPOCHS} | C_loss(avg) {epoch_c_loss:.6f} | GP(avg) {epoch_gp:.6f} | G_loss(avg) {epoch_g_loss:.6f} | EMD {emd:.6f}')

    if epoch % SAVE_EVERY == 0:
      gen_path = os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch}.weights.h5")
      crit_path = os.path.join(CHECKPOINT_DIR, f"critic_epoch_{epoch}.weights.h5")
      generator.save_weights(gen_path)
      critic.save_weights(crit_path)
      print(f"Saved checkpoints: {gen_path}, {crit_path}")

end_time = time.time()
print('Training finished. Total time (s):', end_time - start_time)

# Save final weights
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
generator.save_weights(os.path.join(CHECKPOINT_DIR, 'generator_final.weights.h5'))
critic.save_weights(os.path.join(CHECKPOINT_DIR, 'critic_final.weights.h5'))

# Optionally save training histories
np.save('critic_loss_history.npy', np.array(critic_loss_history))
np.save('generator_loss_history.npy', np.array(generator_loss_history))
np.save('emd_history.npy', np.array(emd_history))
