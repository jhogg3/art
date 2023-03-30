import multiprocessing
import random
import statistics
import random

import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImagePath
from deap import creator, base, tools, algorithms


def draw(solution):
	image = Image.new("RGB", (200, 200))
	canvas = ImageDraw.Draw(image, "RGBA")
	for polygon in solution:
		fill_color = polygon[0]
		vertices = polygon[1:]
		canvas.polygon(vertices, fill=fill_color, )
	return image


MAX = 255 * 200 * 200
TARGET = Image.open(r"C:\Users\Josh\Downloads\davin.png")
TARGET.load()  # read image and close the file

def mutate(solution, indpb):
	if random.random() < 0.5:
		# mutate points
		polygon = random.choice(solution)
		coords = [x for point in polygon[1:] for x in point]
		tools.mutGaussian(coords, 0, 10, indpb)
		coords = [max(0, min(int(x), 200)) for x in coords]
		as_list = list(polygon)
		as_list[1:] = list(zip(coords[::2], coords[1::2]))
		polygon = tuple(as_list)


	else:
		# reorder polygons
		tools.mutShuffleIndexes(solution, indpb)

	return (solution,)  # DEAP expects a tuple here

def evaluate(solution):
	image = draw(solution)
	diff = ImageChops.difference(image, TARGET)
	hist = diff.convert("L").histogram()
	count = sum(i * n for i, n in enumerate(hist))
	print((MAX - count) / MAX)
	return (MAX - count) / MAX,  # DEAP expects a tuple here


def make_polygon():
	x1 = random.randrange(0, 200)
	y1 = random.randrange(0, 200)
	x2 = random.randrange(0, 200)
	y2 = random.randrange(0, 200)
	x3 = random.randrange(0, 200)
	y3 = random.randrange(0, 200)
	R = random.randrange(0, 256)
	G = random.randrange(0, 256)
	B = random.randrange(0, 256)
	A = random.randrange(30, 61)

	return (R, G, B, A), (x1, y1), (x2, y2), (x3, y3)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initRepeat, creator.Individual, make_polygon, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=2)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxMessyOnePoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.6)
toolbox.register("select", tools.selBest, k = 50)
toolbox.register("mutate2", mutate, indpb=0.1)

stats = tools.Statistics(evaluate)
stats.register("avg", statistics.mean)
stats.register("std", statistics.stdev)




def run(generations=1000, population_size=10, seed=60, cxpb=0.5, mutpb=0.3):
	# for reproducibility
	random.seed(seed)
	population = toolbox.population(n=population_size)
	fitness = list(map(toolbox.evaluate, population))

	# main evolution loop
	pool = multiprocessing.Pool(4)
	toolbox.register("map", pool.map)
	individuals = []
	for i in range(len(population)):
		individuals.append(i)

	hof = tools.HallOfFame(1)

	for i in range(generations):

		offspring = toolbox.select(population)
		offspring = list(map(toolbox.clone, offspring))
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < cxpb:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:
			if random.random() < mutpb:
				toolbox.mutate2(mutant)
				del mutant.fitness.values

		# The population is entirely replaced by the offspring
		population[:] = offspring
		hof.update(population)

		draw(hof[0]).save("solution.png")



		# record = stats.compile(population[0])

		#population, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1,
					# ngen=50, stats=stats, halloffame=hof, verbose=False_)

		# print("gen:", i, "avg:", record["avg"], "std:", record["std"], "best:", hof[0].fitness.values[0])

		pool.close()
		print(fitness)
		print("\nbest 3 global solutions:\n", hof)
		print("\nbest 3 in last population:\n", tools.selBest(population, k=4))


if __name__ == "__main__":
	run()
	print("all done")
