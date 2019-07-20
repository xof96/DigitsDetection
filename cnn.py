import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from image_generator import create_train_test_dataset

tf.logging.set_verbosity(tf.logging.INFO)

# Creating the Dataset
N_TRAIN = 5000
N_TEST = 1000

N_DIGITS = 8

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    with tf.variable_scope('net_scope'):
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        input_layer = tf.reshape(features["x"], [-1, 32, 256, 1])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], strides=2,
                                 padding="same", activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2, padding="same")

        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2, padding="same")

        # Convolutional Layer #3
        conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        # Pooling Layer #3
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3, 3], strides=2, padding="same")

        # Convolutional Layer #4
        conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

        # Pooling Layer #4
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3, 3], strides=2, padding="same")

    with tf.variable_scope('class_layer'):
        # Convolutional Layer #5
        conv5 = tf.layers.conv2d(inputs=pool4, filters=10, kernel_size=[1, 1], padding="same")

    logits = tf.reshape(conv5, [-1, N_DIGITS, 10], name="l")

    logits_1 = tf.reshape(tf.gather(logits, indices=[0], axis=1), [-1, 10], name="l1")
    logits_2 = tf.reshape(tf.gather(logits, indices=[1], axis=1), [-1, 10], name="l2")
    logits_3 = tf.reshape(tf.gather(logits, indices=[2], axis=1), [-1, 10], name="l3")
    logits_4 = tf.reshape(tf.gather(logits, indices=[3], axis=1), [-1, 10], name="l4")
    logits_5 = tf.reshape(tf.gather(logits, indices=[4], axis=1), [-1, 10], name="l5")
    logits_6 = tf.reshape(tf.gather(logits, indices=[5], axis=1), [-1, 10], name="l6")
    logits_7 = tf.reshape(tf.gather(logits, indices=[6], axis=1), [-1, 10], name="l7")
    logits_8 = tf.reshape(tf.gather(logits, indices=[7], axis=1), [-1, 10], name="l8")

    labels_1 = labels[:, 0]
    labels_2 = labels[:, 1]
    labels_3 = labels[:, 2]
    labels_4 = labels[:, 3]
    labels_5 = labels[:, 4]
    labels_6 = labels[:, 5]
    labels_7 = labels[:, 6]
    labels_8 = labels[:, 7]

    pred = tf.argmax(tf.nn.softmax(logits, axis=2), axis=2, name="predictions")

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss_1 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels_1, logits=logits_1))
    loss_2 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels_2, logits=logits_2))
    loss_3 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels_3, logits=logits_3))
    loss_4 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels_4, logits=logits_4))
    loss_5 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels_5, logits=logits_5))
    loss_6 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels_6, logits=logits_6))
    loss_7 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels_7, logits=logits_7))
    loss_8 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels_8, logits=logits_8))

    total_loss = tf.reduce_mean(loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7 + loss_8)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=2),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor", axis=2)
    }

    accuracy, _ = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name="acc")
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("loss", total_loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=total_loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    train_valid_data, test_data = create_train_test_dataset(N_TRAIN, N_TEST, N_DIGITS)
    x_train_valid = []
    y_train_valid = []

    for x, y in train_valid_data:
        x_train_valid.append(x)
        y_train_valid.append(y)

    x_train_valid = np.array(x_train_valid).astype(np.float32)
    y_train_valid = np.array(y_train_valid).astype(np.int32)

    x_test = []
    y_test = []

    for x, y in test_data:
        x_test.append(x)
        y_test.append(y)

    x_test = np.array(x_test).astype(np.float32)
    y_test = np.array(y_test).astype(np.int32)

    # Training and Validation data
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid)

    print("Data are Ready")

    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="digits_model")


    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {
        # "probabilities": "l1",
        # "predictions": "predictions",
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        batch_size=64,
        num_epochs=None,
        shuffle=True)

    classifier.train(
        input_fn=train_input_fn,
        steps=5000,
        hooks=[logging_hook])

    print("Trained...")

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_valid}, y=y_valid, batch_size=64, num_epochs=None, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn,
                                       steps=5000,
                                       hooks=[logging_hook])
    print(eval_results)

    # Evaluate the model and print results
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test}, y=y_test, batch_size=64, num_epochs=None, shuffle=False)
    test_results = classifier.evaluate(input_fn=test_input_fn,
                                       steps=5000,
                                       hooks=[logging_hook])
    print(eval_results)
