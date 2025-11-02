import random as rd
import matplotlib.pyplot as plt
from helpers import euclidianDist


def findCentroid(vectors):
    vectors = list(vectors)
    sum_vector = [0 for _ in range(len(vectors[0]))]
    for vector in vectors:
        for i in range(len(vector)):
            sum_vector[i] += vector[i]

    return tuple([x / len(vectors) for x in sum_vector])



def kmeans(points, k):
   centroids = rd.sample(points, k)
   convergence = False
   while True:
       clusters = {centroid:set() for centroid in centroids}
       # categorize each point to a cluster:
       for point in points:
           closest = centroids[0]
           closest_dist = euclidianDist(point, centroids[0])

           for centroid in centroids[1:]:
               dist = euclidianDist(point, centroid)
               if dist < closest_dist:
                   closest_dist = dist
                   closest = centroid
           
           clusters[closest].add(point)

       if convergence:
           return clusters

       # find new centroids of each cluster
       new_centroids = [findCentroid(points) for points in clusters.values()]

       if set(centroids) == set(new_centroids):
           convergence = True

       centroids = new_centroids


def findWeightedCentroids(points, probs, m=2):
    k = len(probs[points[0]])
    dim = len(points[0])
    summed_points = [[0] * dim for _ in range(k)]
    total_weights = [0] * k
    
    for point in points:
        for centroid in range(k):
            w = probs[point][centroid] ** m
            for i in range(dim):
                summed_points[centroid][i] += point[i] * w
            total_weights[centroid] += w

    for centroid in range(k):
        for i in range(dim):
            summed_points[centroid][i] /= total_weights[centroid]

    for centroid in range(k):
        summed_points[centroid] = tuple(summed_points[centroid])

    return summed_points


def fuzzykmeans(points, k, m=2):
    centroids = rd.sample(points, k)
    point_probs = {point:[-1]*k for point in points}
    while True:
        clusters = [[] for _ in range(k)]
        # categorize each point to a cluster:
        for point in points:
            summed_dist = 0
            for i, centroid in enumerate(centroids):
                dist = max(euclidianDist(point, centroid), 1e-9)
                inv_dist = 1 / dist ** (2 / (m - 1))
                point_probs[point][i] = inv_dist
                summed_dist += inv_dist
    
            point_probs[point] = [x/summed_dist for x in point_probs[point]]
    
            closest_centroid = point_probs[point].index(max(point_probs[point]))
            clusters[closest_centroid].append(point)
    
        new_centroids = findWeightedCentroids(points, point_probs)
    
        for i in range(len(centroids)):
            if euclidianDist(centroids[i], new_centroids[i]) >= .01:
                continue

        return {new_centroids[i]:clusters[i] for i in range(k)}

           

def main():
    data_in = []
    num_experiments = 3
    num_samples = 300
    
    for _ in range(num_experiments):
        x_mean = rd.randrange(100)
        y_mean = rd.randrange(100)
        stdev = 5 + rd.random() * 8

        for _ in range(num_samples):
            x = rd.gauss(x_mean, stdev)
            y = rd.gauss(y_mean, stdev)
            data_in.append((x, y))

    clusters = fuzzykmeans(data_in, 3)

    for centroid, points in clusters.items():
        plt.plot(*centroid, 'xk')
        plt.scatter(*zip(*points))

    plt.show()


if __name__ == '__main__':
    main()
