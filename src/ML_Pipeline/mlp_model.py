import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.neural_network import MLPRegressor

# Define a function to build a TensorFlow model
def tf_template():
    model = tf.keras.models.Sequential()
    # Dense hidden layer
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    return model

# Function to generate the TensorFlow model
def mlp_tf_model(X_train, X_test, y_train, y_test):
    # Building a model
    tf_model = tf_template()

    # Compiling the model
    tf_model.compile(loss='mse', 
                 optimizer=tf.keras.optimizers.Adam(), 
                 metrics=['mse', 'mae'])
 
    callback=tf.keras.callbacks.EarlyStopping(
                                    monitor="val_loss",
                                    min_delta=1.e-4,
                                    patience=20,
                                    verbose=0,
                                    mode="auto",
                                    baseline=None,
                                    restore_best_weights=True
                                )    

    # Training the model
    n_epochs = 500
    batch_size = 8
    tf_model.fit(X_train, y_train, callbacks=[callback],
              epochs=n_epochs, batch_size=batch_size, 
              validation_data=(X_test, y_test),
              verbose=0, shuffle=True)

    return tf_model
