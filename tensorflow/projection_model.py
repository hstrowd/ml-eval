#!/usr/bin/python

"""
Simple fantasy football score projection model developed using TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf

TRAINING_DATA = "model_data.train.csv"
TESTING_DATA = "model_data.test.csv"

def main(argv=None):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load datasets.
    print("Loading training data...")
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=TRAINING_DATA,
                                                                       target_dtype=np.int,
                                                                       features_dtype=np.float32)

    print("Loading testing data...")
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=TESTING_DATA,
                                                                   target_dtype=np.int,
                                                                   features_dtype=np.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=11)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 40, 20, 10],
                                                n_classes=5,
                                                model_dir="/tmp/ff_prediction_model")

    # Fit model.
    classifier.fit(x=training_set.data,
                   y=training_set.target,
                   steps=2000)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(x=test_set.data,
                                        y=test_set.target)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))

    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.initialize_all_variables()

    # Classify two new flower samples.
    new_samples = np.array(
        [[10.75, 120.34, 1.52, 18.63, 5, 75, 1, 2, 14, 0, 10.63], [6.23, 76.83, 1.10, 9.27, 3, 17, 0, 12, 101, 2, 19.83]], dtype=np.float32)
    y = classifier.predict(new_samples)
    print('Predictions: {}'.format(str(y)))

    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

if __name__ == "__main__":
   main(sys.argv[1:])
