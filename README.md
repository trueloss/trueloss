# trueloss

`trueloss` is a Python library designed to compute and plot the true training loss and accuracy of a Keras `classification` model without regularization effects. The purpose of creating this library is to address the discrepancy between the training loss displayed in Keras. This discrepancy can lead to misleading conclusions about model performance. `trueloss` allows you to gain a clearer understanding of your model's true performance on the training data.


When training machine learning models, the total training loss usually includes a base loss term and a regularization loss term.The base loss is problem-specific, guiding model predictions (e.g., Cross-Entropy Loss for classification, Mean Squared Error (MSE) for regression). But during testing there is no regularization term added to the base loss. Mathematically, this is represented as:

1. **During Training:**

$$
\text{Training Loss} = \text{Base Loss} + \text{Regularization Loss}
$$

$$
\text{Training Loss} (\mathcal{L}_{\text{training}}) = \mathcal{L}_{\text{base}} + \lambda \|\theta\|^2
$$

2. **During testing:**

$$
\text{Testing Loss} = \text{Base Loss}
$$

$$
\text{Testing Loss} (\mathcal{L}_{\text{testing}}) = \mathcal{L}_{\text{base}}
$$


Currently this version only supports classification models.


## Features

- Computes the true training loss and accuracy without regularization effects.
- Plots training and validation loss and accuracy curves.
- Seamlessly integrates with Keras, using the same `fit` method parameters and defaults.
- Ensures the model instance and history object work normally, with the addition of `base_loss` and `base_accuracy` in the training history.

## Installation

Install the library using pip:

```bash
pip install trueloss
```

## Usage

### Basic Usage

First, import the necessary libraries and the `trueloss` class:

```python
import tensorflow as tf
from trueloss import trueloss
```

### Creating and Compiling a Model

Create and compile your Keras model as usual:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Training the Model with trueloss

Create an instance of the `trueloss` class and train your model using the `fit` method. The parameters for `fit` are exactly the same as those for the Keras `fit` method, with the same defaults. This ensures seamless integration with Keras.

```python
# Initialize the trueloss instance with the model
true_loss_instance = trueloss(model=model, plot=True)

# Train the model
history = true_loss_instance.fit(x_train, y_train, 
                                 validation_data=(x_val, y_val),
                                 epochs=10, 
                                 batch_size=32, 
                                 verbose=1)
```

### Important Notes

1. **Model Instance Behavior**:
   - The model instance will behave normally, even when fitted using `true_loss_instance.fit()`.
   - You can use the model for predictions, evaluations, and other tasks just as you would with a standard Keras model.

2. **Training History**:
   - The training history object returned by the `fit` method is the same as the Keras history object.
   - The only addition is `history.history['base_loss']` and `history.history['base_accuracy']`, which log the true training loss and accuracy.

3. **Verbose Output**:
   - The verbose output during training will be the same as the Keras `fit` method.

### Plotting Only

If you only want to plot the training and validation curves without fitting the model, you can initialize the `trueloss` class without a model and use the `plot_fn` method directly:

```python
# Initialize the trueloss instance without a model
true_loss_instance = trueloss(plot=True)

# Assuming you have a history object from previous training
true_loss_instance.plot_fn(history)
```

## Example

Here is a complete example of how to use the `trueloss` library:

```python
import tensorflow as tf
from trueloss import trueloss

# Load and preprocess the data
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

# Create and compile the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Initialize the trueloss instance
true_loss_instance = trueloss(model=model, plot=True)

# Train the model
history = true_loss_instance.fit(x_train, y_train, 
                                 validation_data=(x_val, y_val),
                                 epochs=10, 
                                 batch_size=32, 
                                 verbose=1)
```

You can also visit this [notebook](https://github.com/trueloss/trueloss/tests/example.ipynb) for an example.



## Contributing

Contributions are welcome! If you have any improvements, suggestions, or bug fixes, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/trueloss/trueloss/LICENSE) file for details.

## Contact

For questions, feedback, or support, please reach out via [GitHub issues](https://github.com/trueloss/trueloss/issues) or email me(Siddique Abusaleh) at trueloss.py@gmail.com.

## Acknowledgements

This library is built on top of Keras and TensorFlow. We thank the contributors of these libraries for their excellent work.

## Citation

If you find this library useful in your research, please consider citing it:

```
@misc{trueloss,
  author = {Siddique Abusaleh},
  title = {trueloss: A library for computing true training loss in Keras models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/trueloss/trueloss}}
}
```

By using `trueloss`, you can gain deeper insights into your model's true performance on the training data while enjoying the seamless integration with Keras.