import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import math

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i+=step

def function_a(x):
    output = math.sin((2*math.pi*x))+math.sin((5*math.pi*x))
    return output


def generate_a_data():
    values = []
    with open('a.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter =',')
        for x in frange(-1.0, 1.0, 0.002):
            calc = [x, function_a(x)]
            writer.writerow(calc)
            values.append(calc)
    file.close()

    plt.axis([-1, 1, -2, 2])
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    x = [x[0] for x in values]
    y = [x[1] for x in values]
    plt.plot(x, y)
    plt.show()


def function_b(x, y):
    output = math.exp(-(x**2 + y**2)/0.1)
    return output


def generate_b_data():
    values = []
    with open('b.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter =',')
        for x in frange(-1.0, 1.0, 0.05):
            for y in frange(-1.0, 1.0, 0.05):
                calc = [x, y, function_b(x, y)]
                writer.writerow(calc)
                values.append(calc)
    file.close()

    x = [x[0] for x in values]
    y = [x[1] for x in values]
    z = [x[2] for x in values]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(x, y, z)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


generate_a_data()