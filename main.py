import random
import math
from functools import partial

from particula import Particula
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib import animation
import statistics
from matplotlib import cm

num_particles = 10
max_iterations = 50
semillas = [5678910, 123456, 6598324, 23971446, 58301398]
bounds_velocity = [-0.3, 0.3]


def rosenbrock(position):
    return 100 * (position[1] - position[0] ** 2) ** 2 + (position[0] - 1) ** 2


def rastrigin(position):
    return 20 + (position[0] ** 2) + (position[1] ** 2) - 10 * (
            np.cos(2 * math.pi * position[0]) + np.cos(2 * math.pi * position[1]))


def pso_local(num_particles, fitness_function, max_iterations, bounds_velocity, semilla):
    evaluaciones = 0
    GraficaFitness = []
    particles = []
    random.seed(semilla)
    t = 0
    if (fitness_function == rosenbrock):
        limitesX = [-1.5, 2]
        limitesY = [-0.5, 3]
    else:
        limitesX = [-5, 5]
        limitesY = [-5, 5]
    for i in range(num_particles):
        x = random.uniform(limitesX[0], limitesX[1])
        y = random.uniform(limitesY[0], limitesY[1])
        initial_position = [x, y]
        particles.append(Particula(initial_position, bounds_velocity))

    fitness_mejor = math.inf
    posiciones = []
    while t < max_iterations:
        t += 1
        posiciones_t = []
        for i in range(num_particles):
            fitnessAct = fitness_function(particles[i].position)
            fitnessPBest = fitness_function(particles[i].pbest)
            evaluaciones += 2
            if fitnessAct < fitnessPBest:
                particles[i].set_pbest(particles[i].position)
            if fitnessAct < fitness_mejor:
                fitness_mejor = fitnessAct
                mejor_posicion = copy.copy(particles[i].pbest)
            posiciones_t.append(copy.copy(particles[i].position))
        posiciones.append(copy.copy(posiciones_t))

        for i in range(num_particles):
            entorno = particles[i].getEntorno(i, num_particles)
            fitnessMejorEntorno = fitness_function(particles[i].pbest)
            evaluaciones += 1
            lbest = copy.copy(particles[entorno[0]].pbest)
            for j in entorno:
                fitnessParticulaActual = fitness_function(particles[j].pbest)
                evaluaciones += 1
                if fitnessParticulaActual < fitnessMejorEntorno:
                    fitnessMejorEntorno = fitnessParticulaActual
                    lbest = copy.copy(particles[j].pbest)
            particles[i].update_velocity_local(lbest)
            particles[i].update_position(limitesX, limitesY)
        GraficaFitness.append(fitness_mejor)

    return fitness_mejor, mejor_posicion, GraficaFitness, posiciones, evaluaciones


def pso_global(num_particles, fitness_function, max_iterations, bounds_velocity, semilla):
    t = 0
    evaluaciones = 0
    particles = []
    GraficaFitness = []
    random.seed(semilla)
    if (fitness_function == rosenbrock):
        limitesX = [-2, 2]
        limitesY = [-1, 3]
    else:
        limitesX = [-5, 5]
        limitesY = [-5, 5]
    for i in range(num_particles):
        x = random.uniform(limitesX[0], limitesX[1])
        y = random.uniform(limitesY[0], limitesY[1])
        initial_position = [x, y]
        particles.append(Particula(initial_position, bounds_velocity))

    fitness_mejor = math.inf
    fitness_gbest = math.inf
    posicion_gbest = particles[0].pbest
    posiciones = []
    while t < max_iterations:
        t += 1
        posiciones_t = []
        for i in range(num_particles):
            fitnessAct = fitness_function(particles[i].position)
            fitnessPBest = fitness_function(particles[i].pbest)
            evaluaciones += 2
            if fitnessAct < fitnessPBest:
                particles[i].set_pbest(particles[i].position)
                fitnessPBest = fitnessAct
            if fitnessPBest < fitness_gbest:
                fitness_gbest = fitnessPBest
                posicion_gbest = copy.copy(particles[i].pbest)
            posiciones_t.append(copy.copy(particles[i].position))
        posiciones.append(copy.copy(posiciones_t))

        for i in range(num_particles):
            if (fitness_function(particles[i].pbest) < fitness_mejor):
                mejor_posicion = copy.copy(particles[i].pbest)
                fitness_mejor = fitness_function(particles[i].pbest)
                evaluaciones += 2
            particles[i].update_velocity_global(posicion_gbest)
            particles[i].update_position(limitesX, limitesY)
        GraficaFitness.append(fitness_mejor)
    return fitness_mejor, mejor_posicion, GraficaFitness, posiciones, evaluaciones


def busqueda_local(funcion, inicial):
    solucion_inicial = inicial
    evaluaciones = 0
    solucion_actual = solucion_inicial
    max_evaluaciones = 3000
    while evaluaciones < max_evaluaciones:

        mejor_vecino = solucion_actual
        for vecino in generar_vecinos(solucion_actual):
            if funcion(vecino) < funcion(mejor_vecino):
                mejor_vecino = vecino
            evaluaciones += 2
        if funcion(mejor_vecino) < funcion(solucion_actual):
            solucion_actual = mejor_vecino
        evaluaciones += 2
        if funcion(mejor_vecino) >= funcion(solucion_actual):
            break
    return solucion_actual, evaluaciones


def generar_vecinos(solucion):
    vecinos = []
    granularidad = 0.1
    vecinos.append([solucion[0] + granularidad, solucion[1] + granularidad])
    vecinos.append([solucion[0] + granularidad, solucion[1] - granularidad])
    vecinos.append([solucion[0] - granularidad, solucion[1] + granularidad])
    vecinos.append([solucion[0] - granularidad, solucion[1] - granularidad])

    for i in range(len(vecinos)):
        if vecinos[i][0] > 10:
            vecinos[i][0] = 10
        elif vecinos[i][0] < -10:
            vecinos[i][0] = -10

        if vecinos[i][1] > 10:
            vecinos[i][1] = 10
        elif vecinos[i][1] < -10:
            vecinos[i][1] = -10
    return vecinos


def representar(funcion):
    if funcion == rosenbrock:
        x = np.linspace(-1.5, 2)
        y = np.linspace(-0.5, 3)
    else:
        x = np.linspace(-5, 5)
        y = np.linspace(-5, 5)
    X, Y = np.meshgrid(x, y)
    Z = funcion([X, Y])
    ax.plot_surface(X, Y, Z, cmap=cm.plasma, alpha=0.8)
    ax.view_init(elev=90, azim=180)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(f'{"ROS" if funcion == rosenbrock else "RAS"}(X,Y)')


def update(frame, posiciones, funcion, nombre, fitness, semilla):
    datos = posiciones[frame]
    datosX = []
    datosY = []
    for j in range(len(datos)):
        datosX.append(datos[j][0])
        datosY.append(datos[j][1])
    plt.cla()
    representar(funcion)
    if funcion == rosenbrock:
        if nombre == "PSOLOCALROS":
            ax.set_title(f"Rosenbrock Local con Semilla {semilla}\nFitness mínimo:{fitness}")
        else:
            ax.set_title(f"Rosenbrock Global con Semilla {semilla}\nFitness mínimo:{fitness}")
    else:
        if nombre == "PSOLOCALRAS":
            ax.set_title(f"Rastrigin Local con Semilla {semilla}\nFitness mínimo:{fitness}")
        else:
            ax.set_title(f"Rastrigin Global con Semilla {semilla}\nFitness mínimo:{fitness}")
    image = ax.scatter(datosX, datosY, c='black', s=10, alpha=1)

    images.append([image])
    return images


def guardarGif(funcion, posiciones, semilla, nombre, fitness):
    global ax
    global images
    fig = plt.figure(figsize=[12, 8])
    ax = plt.axes(projection='3d')

    images = []
    ani = animation.FuncAnimation(fig, partial(update, posiciones=posiciones, funcion=funcion, nombre=nombre,
                                               fitness=fitness, semilla=semilla), frames=int(len(posiciones)),
                                  interval=1, blit=False)

    my_writer = animation.PillowWriter(fps=20, codec='libx264', bitrate=2)

    ani.save(filename=nombre + str(semilla) + '.gif', writer=my_writer)


if __name__ == '__main__':
    ListaEvaluacionesROS = []
    ListaValoresROS = []
    ListaEvaluacionesRAS = []
    ListaValoresRAS = []
    for semilla in semillas:
        print()
        print(f"Semilla {semilla}")
        print("PSO LOCAL ROSENBROCK")
        fitness_best, ms, Grafica, posiciones, evaluaciones = pso_local(num_particles, rosenbrock, max_iterations,
                                                                        bounds_velocity, semilla)
        ListaEvaluacionesROS.append(evaluaciones)
        ListaValoresROS.append(fitness_best)
        print("MEJOR POSICION: ", ms)
        print("Fitness: ", fitness_best)
        fig2, ax2 = plt.subplots()
        ax2.set_xlabel("Iteraciones")
        ax2.set_ylabel("Fitness")
        ax2.set_title(f"PSO LOCAL ROS con Semilla: {semilla}")
        iteraciones = range(0, len(Grafica))
        ax2.plot(iteraciones, Grafica, label="Rosenbrock(x,y)")
        ax2.plot(iteraciones, np.zeros(len(Grafica)), label="Mínimo global")
        ax2.legend()
        ax2.set_xticks(np.arange(0, len(Grafica), step=5))
        plt.savefig(f"PSOLOCALROSENBROCK{semilla}.jpg")
        plt.show()
        guardarGif(rosenbrock, posiciones, semilla, "PSOLOCALROS", fitness_best)
        print()
        print("PSO LOCAL RASTRIGIN")
        fitness_best, ms, Grafica, posiciones, evaluaciones = pso_local(num_particles, rastrigin, max_iterations,
                                                                        bounds_velocity, semilla)
        ListaEvaluacionesRAS.append(evaluaciones)
        ListaValoresRAS.append(fitness_best)
        print("MEJOR POSICION: ", ms)
        print("Fitness: ", fitness_best)
        fig3, ax3 = plt.subplots()
        ax3.set_xlabel("Iteraciones")
        ax3.set_ylabel("Fitness")
        iteraciones = range(0, len(Grafica))
        ax3.plot(iteraciones, Grafica, label="Rastrigin(x,y)")
        ax3.plot(iteraciones, np.zeros(len(Grafica)), label="Mínimo global")
        ax3.legend()
        ax3.set_title(f"PSO LOCAL RAS con Semilla: {semilla}")
        ax3.set_xticks(np.arange(0, len(Grafica), step=5))
        plt.savefig(f"PSOLOCALRASTRIGIN{semilla}.jpg")
        plt.show()
        guardarGif(rastrigin, posiciones, semilla, "PSOLOCALRAS", fitness_best)
    print("------------------------------------------------------------------------------")
    print("ESTADISTICAS PARA ROSENBROCK LOCAL:")
    print("Ev. Medias: ", statistics.mean(ListaEvaluacionesROS))
    print("Ev. Mejor: ", min(ListaEvaluacionesROS))
    print("Ev. Desv: ", statistics.stdev(ListaEvaluacionesROS))
    print("Mejor Valor: ", min(ListaValoresROS))
    print("Media Valores: ", statistics.mean(ListaValoresROS))
    print("Desv Valores: ", statistics.stdev(ListaValoresROS))
    print("------------------------------------------------------------------------------")
    print("ESTADISTICAS PARA RASTRIGIN LOCAL:")
    print("Ev. Medias: ", statistics.mean(ListaEvaluacionesRAS))
    print("Ev. Mejor: ", min(ListaEvaluacionesRAS))
    print("Ev. Desv: ", statistics.stdev(ListaEvaluacionesRAS))
    print("Mejor Valor: ", min(ListaValoresRAS))
    print("Media Valores: ", statistics.mean(ListaValoresRAS))
    print("Desv Valores: ", statistics.stdev(ListaValoresRAS))
    print("------------------------------------------------------------------------------")
    ListaEvaluacionesROS = []
    ListaValoresROS = []
    ListaEvaluacionesRAS = []
    ListaValoresRAS = []
    for semilla in semillas:
        print()
        print(f"Semilla {semilla}")
        print("PSO GLOBAL ROSENBROCK")
        fitness_best, ms, Grafica, posiciones, evaluaciones = pso_global(num_particles, rosenbrock, max_iterations,
                                                                         bounds_velocity,
                                                                         semilla)
        ListaEvaluacionesROS.append(evaluaciones)
        ListaValoresROS.append(fitness_best)
        print("MEJOR POSICION: ", ms)
        print("Fitness: ", fitness_best)
        fig2, ax2 = plt.subplots()
        ax2.set_xlabel("Iteraciones")
        ax2.set_ylabel("Fitness")
        ax2.set_title(f"PSO GLOBAL ROS con Semilla: {semilla}")
        iteraciones = range(0, len(Grafica))
        ax2.plot(iteraciones, Grafica, label="Rosenbrock(x,y)")
        ax2.plot(iteraciones, np.zeros(len(Grafica)), label="Mínimo global")
        ax2.legend()
        ax2.set_xticks(np.arange(0, len(Grafica), step=5))
        plt.savefig(f"PSOGLOBALROSENBROCK{semilla}.jpg")
        plt.show()
        guardarGif(rosenbrock, posiciones, semilla, "PSOGLOBALROS", fitness_best)
        print()
        print("PSO GLOBAL RASTRIGIN")
        fitness_best, ms, Grafica, posiciones, evaluaciones = pso_global(num_particles, rastrigin, max_iterations,
                                                                         bounds_velocity,
                                                                         semilla)
        ListaEvaluacionesRAS.append(evaluaciones)
        ListaValoresRAS.append(fitness_best)
        print("MEJOR POSICION: ", ms)
        print("Fitness: ", fitness_best)
        fig3, ax3 = plt.subplots()
        ax3.set_xlabel("Iteraciones")
        ax3.set_ylabel("Fitness")
        iteraciones = range(0, len(Grafica))
        ax3.plot(iteraciones, Grafica, label="Rastrigin(x,y)")
        ax3.plot(iteraciones, np.zeros(len(Grafica)), label="Mínimo global")
        ax3.legend()
        ax3.set_title(f"PSO GLOBAL RAS con Semilla: {semilla}")
        ax3.set_xticks(np.arange(0, len(Grafica), step=5))
        plt.savefig(f"PSOGLOBALRASTRIGIN{semilla}.jpg")
        plt.show()
        guardarGif(rastrigin, posiciones, semilla, "PSOGLOBALRAS", fitness_best)
    print("------------------------------------------------------------------------------")
    print("ESTADISTICAS PARA ROSENBROCK GLOBAL:")
    print("Ev. Medias: ", statistics.mean(ListaEvaluacionesROS))
    print("Ev. Mejor: ", min(ListaEvaluacionesROS))
    print("Ev. Desv: ", statistics.stdev(ListaEvaluacionesROS))
    print("Mejor Valor: ", min(ListaValoresROS))
    print("Media Valores: ", statistics.mean(ListaValoresROS))
    print("Desv Valores: ", statistics.stdev(ListaValoresROS))
    print("------------------------------------------------------------------------------")
    print("ESTADISTICAS PARA RASTRIGIN GLOBAL:")
    print("Ev. Medias: ", statistics.mean(ListaEvaluacionesRAS))
    print("Ev. Mejor: ", min(ListaEvaluacionesRAS))
    print("Ev. Desv: ", statistics.stdev(ListaEvaluacionesRAS))
    print("Mejor Valor: ", min(ListaValoresRAS))
    print("Media Valores: ", statistics.mean(ListaValoresRAS))
    print("Desv Valores: ", statistics.stdev(ListaValoresRAS))
    print("------------------------------------------------------------------------------")
    print("BUSQUEDA LOCAL MEJOR VECINO ROSENBROCK")
    mejor_solucion,evaluaciones = busqueda_local(rosenbrock, [0, 0])
    print("EVALUACIONES: ",evaluaciones)
    print("POSICION: ", mejor_solucion)
    print("Fitness: ", rosenbrock(mejor_solucion))
    print("BUSQUEDA LOCAL MEJOR VECINO RASTRIGIN")
    mejor_solucion,evaluaciones = busqueda_local(rastrigin, [1, 1])
    print("EVALUACIONES: ",evaluaciones)
    print("POSICION: ", mejor_solucion)
    print("Fitness: ", rastrigin(mejor_solucion))

