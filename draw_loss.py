import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(train_loss, val_loss, save_path='loss_curves.jpg'):
    """
    Plot training and validation loss curves
    
    Args:
        train_loss (list): Training loss history
        val_loss (list): Validation loss history
        save_path (str): Path to save the plot
    """
    epochs = range(1, len(train_loss) + 1)
    
    #plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300)
    plt.close()


# Example usage:
if __name__ == "__main__":
    # Sample data - replace with your actual loss values
    train_loss = [0.6730, 0.5576, 0.4765, 0.3860, 0.3180, 0.1883, 0.1530, 0.0925, 0.0433, 0.0204]
    val_loss = [0.7597, 0.7830, 1.0152, 0.9423, 0.6479, 0.9210, 0.6718, 0.8968, 1.2215, 1.1727]
    
    plot_loss_curves(train_loss, val_loss)