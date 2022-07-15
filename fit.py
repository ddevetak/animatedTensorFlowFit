
import tensorflow as tf
from tensorflow.keras import Model

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = np.linspace(-math.pi, math.pi, 1000)
y = np.sin(x**2) + 0.4*np.random.rand(1000)
#y = np.cos(x**3) + 0.4*np.random.rand(1000)

#x1 = np.linspace(-math.pi, math.pi, 10)
#y1 = np.sin(x**2) + 0.04*np.random.rand(10)

predictionFull, lossFull = [], []

fig, ax = plt.subplots(figsize=(12,7))
curve, = ax.plot(x, x, 'r-', linewidth=2)
time_text = ax.text(.5, .5, '', fontsize=15)

ax.scatter(x, y, color = "orange")
ax.set_xlim(-math.pi, math.pi)
ax.set_ylim(-math.pi, math.pi)

# update
#############################################

def update(i):
    #label = 'timestep {0}'.format(i)
    #curve.set_ydata(predictionFull[i].data.numpy())
    curve.set_ydata(np.array(predictionFull[i].tolist()).squeeze())
    #time_text.set_text('Loss = %.4f' % lossFull[i].tolist())
    time_text.set_text('Loss = %.4f' % lossFull[i])
    time_text.set_x(1.0)
    time_text.set_y(-3.0)
    time_text.set_color('red')

    return curve



# define your custom callback for prediction

class prediction_history(tf.keras.callbacks.Callback):
    def __init__(self):
        self.predhis = []
    def on_epoch_end(self, epoch, logs={}):
        #print(model.predict(x))
        self.predhis.append(model.predict(x))
        predictionFull.append(model.predict(x))
        lossFull.append(round(logs['loss'], 3))

#############################################
model = tf.keras.Sequential(
        [
        tf.keras.layers.Dense(500, input_shape = (1,), activation = tf.nn.relu ),
        tf.keras.layers.Dense(1)
        ]
        )

Iterations = 3000

#model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.1), loss = "mse", metrics = ["mae"])
model.compile(optimizer = "Adam", loss = "mse", metrics = ["mae"])
model.fit(x, y, epochs = Iterations, callbacks=[prediction_history()]) 


if __name__ == '__main__':
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, Iterations, 20), interval=2)
    anim.save('./an.gif', writer='imagemagick', fps=500)


