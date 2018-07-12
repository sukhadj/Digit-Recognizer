import cv2 as cv
import tensorflow as tf
import numpy as np

from train import download, cnn_model, neural_network

tf.logging.set_verbosity(tf.logging.INFO)


def main():
    # train_images, train_labels, test_images, test_labels = download()

    # normalize(train_images)
    # normalize(test_images)

    # Create the estimator
    print("================== Creating the estimator ============")
    mnist_classifier = tf.estimator.Estimator(model_fn=neural_network, model_dir='./temp/cnn_classifier/')
    print(mnist_classifier)

    # Setting up the logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)

    # Train the model
    print("=================== Training ===================")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.asarray(train_images, dtype=np.float32)},
                                                        y=train_labels, batch_size=500, num_epochs=None, shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn, hooks=[logging_hook], steps=1000)

    # Evaluation
    print("=================== Evaluating ===================")
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.asarray(test_images, dtype=np.float32)},
                                                       y=test_labels, batch_size=500, num_epochs=1, shuffle=False)
    eval = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval)

    # Prediction
    img = cv.imread("./test/5.png", 0)
    img = 255 - img

    #img = test_images[1252]
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img = img.reshape([1, 28, 28])

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.asarray(img, dtype=np.float32)},
                                                          y=None, batch_size=1, shuffle=False)
    prediction = mnist_classifier.predict(input_fn=predict_input_fn)
    for i, p in enumerate(prediction):
        print(i, p)
    #train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.asarray(img, dtype=np.float32)},
    #                                                    y=np.array([3], np.int32), batch_size=500, num_epochs=None, shuffle=True)
    #mnist_classifier.train(train_input_fn)

if __name__ == '__main__':
    main()
