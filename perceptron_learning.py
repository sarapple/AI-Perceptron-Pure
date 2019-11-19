import math
import numpy as np
from matplotlib import pyplot as plt

class PerceptronLearning:
  @staticmethod
  def run(training_inputs, results, initial_weights = None, iterations = 1):
    if (initial_weights != None):
      weights = initial_weights
    else:
      default_weights = [0, 0, 0]
      weights = default_weights

    for _ in range(iterations):
      for example_id, training_input in enumerate(PerceptronLearning.add_bias_to_inputs(training_inputs)):
        weights = PerceptronLearning.get_updated_weight_for_input(
          weights = weights,
          training_input = training_input,
          true_label = results[example_id]
        )
        
    return weights

  @staticmethod
  def add_bias_to_inputs(training_inputs):
    # Adds 1 for X0 so that intercept (a.k.a. w0) * 1 is the identity
    return [[1] + x for x in training_inputs]

  @staticmethod
  def miscategorized_inputs(training_inputs, results, weights):
    training_inputs_with_bias = PerceptronLearning.add_bias_to_inputs(training_inputs)
    return [
      x for (index, x) in enumerate(training_inputs_with_bias)
        if PerceptronLearning.has_error(
          results[index],
          PerceptronLearning.get_computed_output(weights, x)
        )
    ]

  @staticmethod
  def get_updated_weight_for_input(weights, training_input, true_label, iteration = 1):
    computed_output = PerceptronLearning.get_computed_output(
      training_input = training_input,
      weights = weights
    )

    if PerceptronLearning.has_error(true_label, computed_output) == False:
      return weights

    return PerceptronLearning.map_weight_to_adjusted_weights(
      weights = weights,
      true_label = true_label,
      training_input = training_input,
      computed_output = computed_output
    )

  @staticmethod
  def map_weight_to_adjusted_weights(weights, true_label, training_input, computed_output):
    return [
      PerceptronLearning.get_updated_weight(
        starting_weight_for_feature = weight,
        desired_output = true_label,
        computed_output = computed_output,
        input_for_feature = training_input[index],
      ) for (index, weight) in enumerate(weights)]
 
  @staticmethod
  def compute_error(true_label, computed_output):
    error = true_label * computed_output
    return error

  @staticmethod
  def has_error(true_label, computed_output):
    # negative means polarity is reversed
    return PerceptronLearning.compute_error(true_label, computed_output) <= 0

  @staticmethod
  def get_computed_output(weights, training_input):
    # TODO: There must be a dot product function
    x_feature_0 = training_input[0]
    x_feature_1 = training_input[1]
    x_feature_2 = training_input[2]

    weight_0 = weights[0]
    weight_1 = weights[1]
    weight_2 = weights[2]

    return (x_feature_0 * weight_0) + (x_feature_1 * weight_1) + (x_feature_2 * weight_2)

  @staticmethod
  def get_updated_weight(starting_weight_for_feature, desired_output, computed_output, input_for_feature):
    learning_rate = 1

    return starting_weight_for_feature + (learning_rate * desired_output * input_for_feature)
