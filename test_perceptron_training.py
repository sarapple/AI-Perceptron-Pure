from perceptron_learning import PerceptronLearning

def test_run_1():
  training_inputs = [
    [8, -11]
  ]
  initial_weights = [0, 0, 0]
  results = [1]
  iterations = 1

  result = PerceptronLearning.run(training_inputs, results = results, initial_weights = initial_weights, iterations = iterations)

  # weights get updated because y * f(x) = 0 and error case is if y * f(x) <= 0
  # [x0, x1, x2] with x0 being always set to 1 for identity
  assert result == [1, 8, -11]

def test_run_2():
  training_inputs = [
    [8, -11],
    [7, 7]
  ]
  initial_weights = [0, 0, 0]
  results = [1, -1]
  iterations = 1

  result = PerceptronLearning.run(training_inputs, results = results, initial_weights = initial_weights, iterations = iterations)

  # [x0, x1, x2] with x0 being always set to 1 for identity
  # 7,7 will not be considered an error so will be unchanged compared to test_run_1
  assert result == [1, 8, -11]

def test_get_updated_weight_for_input():
  weights = [1, 2, 3]
  training_input = [1, -2, 3]
  true_label = -1

  result = PerceptronLearning.get_updated_weight_for_input(
    weights = weights,
    training_input = training_input,
    true_label = true_label
  )
  assert result == [0, 4, 0]

def test_has_error():
  true_label = 10
  computed_output = -10

  result = PerceptronLearning.has_error(true_label, computed_output)
  
  assert result == True

def test_get_computed_output():
  weights = [1, 2, 3]
  training_input = [1, -2, 3]

  result = PerceptronLearning.get_computed_output(weights, training_input)
  assert result == 6

def test_get_updated_weight():
  starting_weight_for_feature = 1
  desired_output = -1
  input_for_feature = 1
  computed_output = 6

  result = PerceptronLearning.get_updated_weight(
    starting_weight_for_feature = starting_weight_for_feature,
    desired_output = desired_output,
    input_for_feature = input_for_feature,
    computed_output = computed_output
  )

  assert result == 0
