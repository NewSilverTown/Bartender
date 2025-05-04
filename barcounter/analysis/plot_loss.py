# analysis/plot_loss.py
import matplotlib.pyplot as plt
import numpy as np

losses = np.load("training/loss_history.npy")
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("training/loss_curve.png")