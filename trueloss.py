import tensorflow as tf
import matplotlib.pyplot as plt

class trueloss:
    """A class to compute and plot the true training loss and accuracy of a Keras model without regularization effects."""
    
    def __init__(self, model=None, plot=False, loss_after='epoch'):
        """
        Initialize the trueloss class.

        Parameters:
        - model: The Keras classification model to be trained.
        - plot: Boolean, whether to plot the loss and accuracy after training.
        - loss_after: String, whether to log the true loss and accuracy after each 'epoch' or 'batch'.
        """
        self.model = model
        self.loss_after = loss_after
        self.plot = plot

    def fit(self, *args, **kwargs):
        """
        Train the model and plot the training progress if required.

        Returns:
        - history: Training keras history object.
        """
        kwargs = self.prepare_input(*args, **kwargs)
        history = self.model.fit(**kwargs)
        if self.plot:
            self.plot_fn(history)
        return history

    def plot_fn(self, history):
        """
        Plot the training and validation loss and accuracy.

        Parameters:
        - history: Training history object returned by keras model.
        """
        plt.figure(figsize=(11, 5))
        epochs = range(1, len(history.history['loss']) + 1)

        # Plot for training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history.history['loss'], label='Train Loss', color='#1f77b4', linestyle='-')
        if 'val_loss' in history.history:
            plt.plot(epochs, history.history['val_loss'], label='Validation Loss', color='#ff7f0e', linestyle='-')
        if 'base_loss' in history.history:
            plt.plot(epochs, history.history['base_loss'], label='True Training Loss', color='#1f77b4', linestyle='-.')
        plt.title('Loss Over Epochs', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)

        # Plot for training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linestyle='-')
        if 'val_accuracy' in history.history:
            plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linestyle='-')
        if 'base_accuracy' in history.history:
            plt.plot(epochs, history.history['base_accuracy'], label='True Training Accuracy', color='#1f77b4', linestyle='-.')
        plt.title('Accuracy Over Epochs', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout(pad=2.0)
        
        # Display the plot
        plt.show()

    def prepare_input(self, *args, **kwargs):
        """
        Standardize input and add the TruePerformanceCallback to the callbacks.

        Returns:
        - params: Dictionary of standardized parameters.
        """
        # Default parameters
        params = {
            'x': None,
            'y': None,
            'batch_size': None,
            'epochs': 1,
            'verbose': 'auto',
            'callbacks': None,
            'validation_split': 0.0,
            'validation_data': None,
            'shuffle': True,
            'class_weight': None,
            'sample_weight': None,
            'initial_epoch': 0,
            'steps_per_epoch': None,
            'validation_steps': None,
            'validation_batch_size': None,
            'validation_freq': 1,
            'max_queue_size': 10,
            'workers': 1,
            'use_multiprocessing': False
        }

        # Handle input arguments and update parameters
        if len(args) > 0:
            if isinstance(args[0], (tf.data.Dataset, tf.keras.utils.Sequence)):
                # X is dataloader
                del params['y']
                for key, val in kwargs.items():
                    params[key] = val
                for arg, key in zip(args, params.keys()):
                    params[key] = arg
                callbacks = params['callbacks'] if params['callbacks'] else []
                callbacks.append(TruePerformanceCallback(params['x']))
                params['callbacks'] = callbacks
                return params
            else:
                # X and y are provided
                for key, val in kwargs.items():
                    params[key] = val
                for arg, key in zip(args, params.keys()):
                    params[key] = arg
                callbacks = params['callbacks'] if params['callbacks'] else []
                callbacks.append(TruePerformanceCallback(params['x'], params['y']))
                params['callbacks'] = callbacks
                return params
        else:
            # When args is empty
            for key, val in kwargs.items():
                params[key] = val
            callbacks = params['callbacks'] if params['callbacks'] else []
            
            if params['y']:
                callbacks.append(TruePerformanceCallback(params['x'], params['y']))
            else:
                callbacks.append(TruePerformanceCallback(params['x']))

            params['callbacks'] = callbacks
            return params

class TruePerformanceCallback(tf.keras.callbacks.Callback):
    """A custom callback to compute the primary loss and accuracy of the model on training data without regularization effects."""

    def __init__(self, x, y=None):
        """
        Initialize the TruePerformanceCallback.

        Parameters:
        - x: Training data.
        - y: Target data, optional.
        """
        super().__init__()
        self.x = x
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to log the true training loss and accuracy.

        Parameters:
        - epoch: Current epoch number.
        - logs: Dictionary of logs from the current epoch.
        """
        if logs is not None:
            # Evaluate the model on the training data
            if self.y is None:
                results = self.model.evaluate(self.x, verbose=0)
            else:
                results = self.model.evaluate(self.x, self.y, verbose=0)
            # Log the evaluation results
            logs['base_loss'] = results[0]
            logs['base_accuracy'] = results[1]
