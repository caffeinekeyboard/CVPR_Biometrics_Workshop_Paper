import json
import matplotlib.pyplot as plt

with open('./checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

epochs = range(1, len(history['train_loss']) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, history['train_loss'], label='Training Loss', marker='o')
plt.plot(epochs, history['val_loss'], label='Validation Loss', marker='s')

plt.title('2D GumNet Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Pearson Correlation Loss (1 - r)')
plt.legend()
plt.grid(True)
plt.show()