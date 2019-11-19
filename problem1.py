import sys

from perceptron_learning import PerceptronLearning
from reporter import Reporter
from reader import Reader
from visualizer import Visualizer

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

  Reporter.write_output(file_name = output_csv_file_name, content = "", should_overwrite_file = True)

  while (weights != previous_weights and iterations <= max_iterations):
    if (weights != None):
      # update previous weight
      previous_weights = weights

    # import ipdb; ipdb.set_trace()
  
    # weights will be list in the form of [b of w_0, w_1, w_2]
    weights = PerceptronLearning.run(
      training_inputs = input_values,
      initial_weights = previous_weights,
      iterations = 1
    )

    Reporter.write_output(
      file_name = output_csv_file_name,
      content = ','.join(map(str, [weights[1], weights[2], weights[0]])) + "\n",
    )

    Visualizer.draw_chart(
      input_values = input_values,
      weights = weights,
      file_name = "figures/figure_" + str(iterations)
    )
    
    iterations += 1

if __name__ == '__main__':
    main()
