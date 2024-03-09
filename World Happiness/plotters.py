import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_loss(loss_data, name):
    plt.plot(loss_data.history["loss"], label = "Training Loss")
    plt.plot(loss_data.history["val_loss"], label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc = "best")
    plt.savefig("World Happiness"+f"\{name}.png")
    print("Saved to ")
    print("Saved to "+ "<World Happiness"+f"\{name}.png>")
    plt.close()

def plot_apply(real_x, real_y, pred_model, name, k):
    plt.scatter(real_x, real_y, label = "Test data")
    x = tf.linspace(np.min(real_x), np.max(real_x), num = 300)
    plt.plot(x, pred_model.predict(x), label = "Model Regression")
    plt.legend(loc = "best")
    plt.xlabel(name)
    plt.ylabel("Score")
    plt.savefig(f"World Happiness\predictions201{k}.png")
    plt.close()