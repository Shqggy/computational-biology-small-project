import random as rd
import matplotlib.pyplot as plt

def euclidianDist(x, y):
    if len(x) != len(y):
        return -1
    
    ss = 0 # sum of squared distance
    for i in range(len(x)):
        ss += (y[i] - x[i])**2

    return ss**.5


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


def findWeightedCentroids(points, probs):
    k = len(probs[points[0]])
    summed_point = [[0] * len(points) for _ in range(k)]
    cluster_weights = [0] * k
    for point in points:
        for i in range(k):
            for j in range(len(point)):
                summed_point[j][i] += point[j] * probs[point][j]
            cluster_weights[i] += probs[point][j]

    for i in range(k):
        for j in range(len(point)):
            summed_point[j][i] /= cluster_weights[j]

    for i in range(k):
        summed_point[i] = tuple(summed_point[i])
    
    return list(summed_point)

        


       
def fuzzykmeans(points, k):
   centroids = rd.sample(points, k)
   point_probs = {point:[-1]*k for point in points}
   clusters = [[] for _ in range(k)]
   converged = False
   while True:
       # categorize each point to a cluster:
       for point in points:
           summed_dist = 0
           for i, centroid in enumerate(centroids):
               dist = euclidianDist(point, centroid)
               point_probs[point][i] = dist
               summed_dist += dist

           point_probs[point] = [x/summed_dist for x in point_probs[point]]

           closest_centroid = point_probs[point].index(max(point_probs[point]))
           clusters[closest_centroid].append(point)
       
       if converged:
           return {centroids[i]:clusters[i] for i in range(k)}

       new_centroids = findWeightedCentroids(points, point_probs)
       print(new_centroids)

       converged = True
       for i in range(len(centroids)):
           if euclidianDist(centroids[i], new_centroids[i]) >= .01:
               converged = False

       centroids = new_centroids
           

        


def main():
    data_in = []
    for _ in range(100):
        data_in.append(tuple([rd.randrange(100) for _ in range(2)]))

    clusters = fuzzykmeans(data_in, 3)

    for centroid, points in clusters.items():
        plt.plot(*centroid, 'xk')
        plt.scatter(*zip(*points))

    plt.show()


if __name__ == '__main__':
    main()
