import sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from PyQt6.QtCore import QTimer, Qt

# Globale Parameter
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_TO_DELTA = {
    0: (0, -10),   # UP
    1: (0, 10),    # DOWN
    2: (-10, 0),   # LEFT
    3: (10, 0)     # RIGHT
}

# Q-Netzwerk
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Deep Q-Learning Agent
class DeepQLearningAgent:
    def __init__(self, alpha=0.001, gamma=0.9, epsilon=0.2):
        self.model = QNetwork(input_dim=2, output_dim=4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.pos = self.random_pos()

    def random_pos(self):
        return [random.randint(100, 700), random.randint(100, 700)]

    def get_state(self, goal_pos):
        dx = goal_pos[0] - self.pos[0]
        dy = goal_pos[1] - self.pos[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = (math.degrees(angle_rad) + 450) % 360
        norm_angle = angle_deg / 360.0
        dist = min(math.sqrt(dx**2 + dy**2), 400)
        norm_dist = dist / 400.0
        return np.array([norm_angle, norm_dist], dtype=np.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        state_tensor = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())

    def update_q(self, state, action, reward, next_state):
        state_tensor = torch.tensor(state).float().unsqueeze(0)
        next_state_tensor = torch.tensor(next_state).float().unsqueeze(0)

        q_values = self.model(state_tensor)
        with torch.no_grad():
            next_q_values = self.model(next_state_tensor)
            max_next_q = torch.max(next_q_values)

        target_q_values = q_values.clone()
        target_q_values[0][action] = reward + self.gamma * max_next_q

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def move(self, action):
        dx, dy = ACTION_TO_DELTA[action]
        self.pos[0] = min(max(self.pos[0] + dx, 0), 800)
        self.pos[1] = min(max(self.pos[1] + dy, 0), 800)

# GUI und Simulation
class QLearningDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Q-Learning Visualisierung")
        self.setGeometry(100, 100, 800, 800)

        self.agent = DeepQLearningAgent()
        self.goal = self.agent.random_pos()
        self.prev_distance = self.distance(self.agent.pos, self.goal)
        self.start_distance = self.prev_distance
        self.last_state = self.agent.get_state(self.goal)

        self.distance_history = []
        self.episode_efficiency = []
        self.episode_step_counter = 0
        self.plot_counter = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_env)
        self.timer.start(5)

    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def update_env(self):
        state = self.agent.get_state(self.goal)
        action = self.agent.choose_action(state)
        self.agent.move(action)

        new_state = self.agent.get_state(self.goal)
        dist = self.distance(self.agent.pos, self.goal)
        self.distance_history.append(dist)
        self.episode_step_counter += 1

        if dist < 15:
            reward = 100
            efficiency = self.episode_step_counter / max(self.start_distance, 1)
            self.episode_efficiency.append(efficiency)
            self.goal = self.agent.random_pos()
            self.prev_distance = self.distance(self.agent.pos, self.goal)
            self.start_distance = self.prev_distance
            self.episode_step_counter = 0
        else:
            reward = self.prev_distance - dist
            self.prev_distance = dist

        self.agent.update_q(state, action, reward, new_state)
        self.last_state = new_state
        self.repaint()

        self.plot_counter += 1
        if self.plot_counter % 100 == 0:
            self.plot_learning_progress()

    def plot_learning_progress(self):
        plt.figure("Lernfortschritt: Distanz zum Ziel")
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(self.distance_history[-500:])
        plt.xlabel("Zeit (Schritte)")
        plt.ylabel("Distanz zum Ziel")
        plt.title("Distanzentwicklung")

        if len(self.episode_efficiency) > 1:
            plt.subplot(2, 1, 2)
            plt.plot(self.episode_efficiency)
            plt.xlabel("Episode")
            plt.ylabel("Schritte / Startdistanz")
            plt.title("Effizienz pro Episode")

        plt.tight_layout()
        plt.pause(0.001)

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setBrush(QColor(255, 0, 0))
        qp.drawEllipse(self.goal[0] - 10, self.goal[1] - 10, 20, 20)
        qp.setBrush(QColor(0, 0, 0))
        qp.drawEllipse(self.agent.pos[0] - 10, self.agent.pos[1] - 10, 20, 20)
        qp.setPen(QPen(QColor(0, 255, 0), 2))
        qp.drawLine(self.agent.pos[0], self.agent.pos[1], self.goal[0], self.goal[1])

        if self.last_state is not None:
            qp.setPen(QPen(QColor(0, 0, 255)))
            qp.setFont(QFont("Arial", 10))
            x, y = self.agent.pos
            offset = 25
            angle_label = int(self.last_state[0] * 360)
            qp.drawText(x + 30, y - 50, f"Zustand: Winkel={angle_label}°, Distanz={self.last_state[1]:.2f}")
            for i, action in enumerate(ACTIONS):
                state_tensor = torch.tensor(self.last_state).float().unsqueeze(0)
                with torch.no_grad():
                    q_val = self.agent.model(state_tensor)[0][i].item()
                qp.drawText(x + 30, y + i * offset - 30, f"{action}: {q_val:.2f}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = QLearningDemo()
    demo.show()
    plt.ion()
    plt.show()
    sys.exit(app.exec())
