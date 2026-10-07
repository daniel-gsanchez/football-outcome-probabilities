"""Neural network models (Keras)."""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TensorFlow C++ logs

import tensorflow as tf  # noqa: E402
from tensorflow.keras import Model, layers  # noqa: E402

SEED = 42


def set_seed(seed=SEED):
    """Make training reproducible (same weights, same results on the same machine)."""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_lstm(n_teams, n_leagues, window, n_features=5, units=32):
    """Shared LSTM over each team's recent matches + team and league embeddings.

    The same LSTM encodes the home and the away history, so "form" is learned
    once and applied to both sides.
    """
    home_seq = layers.Input((window, n_features), name="home_seq")
    away_seq = layers.Input((window, n_features), name="away_seq")
    home_team = layers.Input((), dtype="int32", name="home_team")
    away_team = layers.Input((), dtype="int32", name="away_team")
    league = layers.Input((), dtype="int32", name="league")

    lstm = layers.LSTM(units, name="form_encoder")
    team_emb = layers.Embedding(n_teams, 16, name="team_embedding")
    league_emb = layers.Embedding(n_leagues, 4, name="league_embedding")

    x = layers.Concatenate()([
        lstm(home_seq), lstm(away_seq),
        team_emb(home_team), team_emb(away_team), league_emb(league),
    ])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(3, activation="softmax", name="probabilities")(x)

    model = Model([home_seq, away_seq, home_team, away_team, league], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy")
    return model


def build_mlp(n_inputs):
    """Feed-forward baseline on window averages."""
    inp = layers.Input((n_inputs,))
    x = layers.Dense(32, activation="relu")(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(3, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy")
    return model


def fit(model, x_train, y_train, x_val, y_val, class_weight=None):
    """Train with early stopping on validation log-loss; keep the best epoch."""
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                            restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=50, batch_size=256, class_weight=class_weight,
                        callbacks=[stop], verbose=0)
    return history
