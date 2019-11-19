import sys

from perceptron_learning import PerceptronLearning
from visualizer import Visualizer
from reporter import Reporter
from reader import Reader

def main():
  input_csv_file_name = sys.argv[1]
  output_csv_file_name = sys.argv[2]

  # input values are in the form of [feature_1, feature_2, label]
  input_values = Reader.csv(input_csv_file_name)

  # Track previous weights and allow to compare against latest weight to check convergence
  previous_weights = [0, 0, 0]
  weights = None

  Reporter.write_output(file_name = output_csv_file_name, content = "", should_overwrite_file = True)

  training_inputs = [[x[0], x[1]] for x in input_values]
  results = [x[2] for x in input_values]

  iterations = 0

  while (previous_weights != weights):
    # Past the initial condition, we want to track the previous_weight
    if (weights != None):
      # update previous weight so we can remember for comparison
      previous_weights = weights
      # import ipdb; ipdb.set_trace()

    # weights will be list in the form of [b or w_0, w_1, w_2]
    weights = PerceptronLearning.run(
      training_inputs = training_inputs,
      results = results,
      initial_weights = previous_weights,
      iterations = 1
    )

    # write lines to output file
    Reporter.write_output(
      file_name = output_csv_file_name,
      content = ','.join(map(str, [weights[1], weights[2], weights[0]])) + "\n",
    )

    # create png images of the figures
    Visualizer.draw_chart(
      input_values = input_values,
      weights = weights,
      file_name = "figures/figure_" + str(iterations)
    )

    iterations += 1

if __name__ == '__main__':
    main()
