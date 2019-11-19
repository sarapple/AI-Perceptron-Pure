from perceptron_learning import PerceptronLearning

def test_run():
  training_values = [
    [8, -11, 1]
  ]
  initial_weights = [0, 0, 0]
  iterations = 1

  result = PerceptronLearning.run(training_values, initial_weights = initial_weights, iterations = iterations)

  assert result == [1, 2, 3]

def test_get_updated_weight_for_input():
  weights = [1, 2, 3]
  training_value = [1, -2, -3, 1]

  result = PerceptronLearning.get_updated_weight_for_input(weights, training_value)
  assert result == [-2, -2, 3]

def test_compute_error():
  true_label = 10
  computed_output = -10

  result = PerceptronLearning.compute_error(true_label, computed_output)
  
  assert result == -100

def test_has_error():
  true_label = 10
  computed_output = -10

  result = PerceptronLearning.has_error(true_label, computed_output)
  
  assert result == True

def test_get_computed_output():
  weights = [1, 2, 3]
  training_value = [1, -2, 3]

  result = PerceptronLearning.get_computed_output(weights, training_value)
  assert result == 6

def test_get_updated_weight():
  starting_weight_for_feature = 2
  desired_output = 1
  input_for_feature = 8

  result = PerceptronLearning.get_updated_weight(starting_weight_for_feature, desired_output, input_for_feature)

  assert result == 10
