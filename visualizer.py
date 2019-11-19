import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.get_cachedir()

class Visualizer:
  @staticmethod
  def draw_chart(input_values, weights, file_name):
    plt.figure(file_name)
    category_1 = [x for x in input_values if x[-1] == 1]
    category_2 = [x for x in input_values if x[-1] == -1]

    plt.plot(
      [x[0] for x in category_1],
      [x[1] for x in category_1],
      'ro'
    )

    plt.plot(
      [x[0] for x in category_2],
      [x[1] for x in category_2],
      'bo'
    )

    Visualizer.abline((weights[1]/weights[2] if weights[2] != 0 else 0), weights[0])

    plt.savefig(file_name)
    plt.clf()

  @staticmethod
  def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')