import math
from scipy.stats import norm
import random as rd
import matplotlib.pyplot as plt
from helpers import euclidianDist


class ExpectationMaximizer():
    def __init__(self, num_initial=4, num_points=30):
        self.num_experiments = 0
        self.experiments = []
        self.parameters = []
        self.points = []
        self.guesses = []

        for _ in range(num_initial):
            self.run_experiment(num_points)

    def run_experiment(self, n=30):
        data = experiment(n)

        self.experiments.append(data)
        self.num_experiments += 1

        self.parameters.append(self.estimate_parameters(data))

    def estimate_parameters(self, data):
        n = len(data)
        x_mean = sum([x[0] for x in data]) / n
        y_mean = sum([x[1] for x in data]) / n

        x_var = sum([(x[0] - x_mean)**2 for x in data]) / (n - 1)
        y_var = sum([(x[1] - y_mean)**2 for x in data]) / (n - 1)
                     
        x_stdev = x_var**0.5
        y_stdev = y_var**0.5

        return (x_mean, y_mean, x_stdev, y_stdev)

    def guess_class(self, point):
        self.points.append(point)
        guess = 0
        max_likelihood = 0
        x, y = point

        for i in range(self.num_experiments):
            x_mu, y_mu, x_sd, y_sd = self.parameters[i]
            
            x_likelihood = normal_pdf(x, x_mu, x_sd)
            y_likelihood = normal_pdf(y, y_mu, y_sd)

            likelihood = x_likelihood * y_likelihood

            if likelihood > max_likelihood:
                max_likelihood = likelihood
                guess = i

        self.guesses.append(guess)
        return guess

    def show_chart(self, show_means=True, show_guesses=True):
        colors = 'bgrcmy'
        for i, data in enumerate(self.experiments):
            plt.scatter(*zip(*data), color=colors[i % len(colors)])
        
        if show_guesses:
            for i, point in enumerate(self.points):
                plt.plot(*point, '*' + colors[self.guesses[i] % len(colors)])


        if show_means:
            for x_mu, y_mu, _, _ in self.parameters:
                plt.plot(x_mu, y_mu, 'xk')
        plt.show()

        
def normal_pdf(x, mean, stdev):
    # yuck
    return (1 / ((2 * math.pi * (stdev**2))**0.5)) * math.exp(-1 * ((x - mean)**2 / (2 * stdev**2)))


def experiment(n, mean_range=(10, 90), stdev_range=(1, 15)):
    points = []
    x_mean = rd.uniform(*mean_range)
    y_mean = rd.uniform(*mean_range)
    stdev = rd.uniform(*stdev_range)
    for _ in range(n):
        points.append((rd.gauss(x_mean, stdev), rd.gauss(y_mean, stdev)))

    return points


def main():
    em = ExpectationMaximizer(num_points=100)

    num_new_points = 100
    new_points = [(rd.uniform(0, 100), rd.uniform(0, 100)) for _ in range(num_new_points)]
    for point in new_points:
        em.guess_class(point)

    em.show_chart()


if __name__ == '__main__':
    main()
