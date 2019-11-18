import sys

from perceptron_learning import PerceptronLearning
from reporter import Reporter
from reader import Reader

def main():
  input_csv_file_name = sys.argv[1]
  output_csv_file_name = sys.argv[2]

  # input values are in the form of [feature_1, feature_2, label]
  input_values = Reader.csv(input_csv_file_name)

  # TODO: Remove after dev
  max_iterations = 10
  iterations = 0

  # Track previous weights and allow to compare against latest weight to check convergence
  previous_weights = [0, 0, 0]
  weights = None

  while (weights != previous_weights or iterations >= max_iterations):
    if (weights != None):
      # update previous weight
      previous_weights = weights
  
    # weights will be list in the form of [w_1, w_2, b]
    weights = PerceptronLearning.run(
      training_values = input_values,
      initial_weights = previous_weights,
      iterations = 1
    )
    Reporter.write_output(file_name = output_csv_file_name, values = weights)

if __name__ == '__main__':
    main()
