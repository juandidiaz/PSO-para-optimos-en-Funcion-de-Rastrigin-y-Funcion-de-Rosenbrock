import random
import math

omega = 0.729
phi1 = phi2 = 1.49445

Vecindad = 2


class Particula:
    def __init__(self, position, bounds_velocity):
        self.position = position
        self.velocity = [None] * len(position)
        for i in range(len(position)):
            self.velocity[i] = random.uniform(bounds_velocity[0], bounds_velocity[1])
        self.pbest = position
        self.gbest = position

    def update_velocity_local(self, lbest):
        r1 = random.random()
        r2 = random.random()
        for i in range(len(self.velocity)):
            self.velocity[i] = (omega * self.velocity[i]) + (phi1 * r1 * (self.pbest[i] - self.position[i])) + (
                    phi2 * r2 * (lbest[i] - self.position[i]))

    def update_velocity_global(self, gbest):
        r1 = random.random()
        r2 = random.random()
        for i in range(len(self.velocity)):
            self.velocity[i] = (omega * self.velocity[i]) + (phi1 * r1 * (self.pbest[i] - self.position[i])) + (
                    phi2 * r2 * (gbest[i] - self.position[i]))

    def update_position(self, limitesX, limitesY):
        for i in range(len(self.position)):
            suma = self.position[i] + self.velocity[i]
            if (i == 0):
                if suma >= limitesX[0] and suma <= limitesX[1]:
                    self.position[i] = suma
                elif suma < limitesX[0]:
                    self.position[i] = limitesX[0]
                else:
                    self.position[i] = limitesX[1]

            else:
                if suma >= limitesY[0] and suma <= limitesY[1]:
                    self.position[i] = suma
                elif suma < limitesY[0]:
                    self.position[i] = limitesY[0]
                else:
                    self.position[i] = limitesY[1]

    def set_pbest(self, pbest):
        self.pbest = pbest

    def set_gbest(self, gbest):
        self.gbest = gbest

    def set_position(self, position):
        self.position = position

    def get_pbest(self):
        return self.pbest

    def getEntorno(self, indice, num_particles):
        entorno = []
        for i in range(-Vecindad, Vecindad + 1):
            if (i != 0):
                vecino = (indice + i) % num_particles
                entorno.append(vecino)
        return entorno
