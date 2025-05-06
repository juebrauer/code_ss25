import sys
import random
import math
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
from PyQt6.QtCore import QTimer, Qt, QPointF

# Globale Parameter für die Segmentierung
NUM_ANGLE_SECTORS = 10
NUM_DISTANCE_LEVELS = 10

# Parameter für Q-Learning
alpha = 0.1
gamma = 0.9
epsilon = 0.2

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_TO_DELTA = {
    'UP': (0, -10),
    'DOWN': (0, 10),
    'LEFT': (-10, 0),
    'RIGHT': (10, 0)
}
ACTION_TO_ANGLE = {
    'UP': -90,
    'DOWN': 90,
    'LEFT': 180,
    'RIGHT': 0
}

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.pos = self.random_pos()

    def random_pos(self):
        return [random.randint(100, 700), random.randint(100, 700)]

    def get_state(self, goal_pos):
        dx = goal_pos[0] - self.pos[0]
        dy = goal_pos[1] - self.pos[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = (math.degrees(angle_rad) + 450) % 360
        angle_sector = int(angle_deg / (360 / NUM_ANGLE_SECTORS))

        dist = min(math.sqrt(dx**2 + dy**2), 400)
        dist_level = int(dist / (400 / NUM_DISTANCE_LEVELS))
        return (angle_sector, dist_level)

    def choose_action(self, state):
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        q_vals = [self.q_table.get((state, a), 0.0) for a in ACTIONS]
        return ACTIONS[q_vals.index(max(q_vals))]

    def update_q(self, state, action, reward, next_state):
        old_q = self.q_table.get((state, action), 0.0)
        future_q = max([self.q_table.get((next_state, a), 0.0) for a in ACTIONS])
        new_q = old_q + alpha * (reward + gamma * future_q - old_q)
        self.q_table[(state, action)] = new_q

    def move(self, action):
        dx, dy = ACTION_TO_DELTA[action]
        self.pos[0] = min(max(self.pos[0] + dx, 0), 800)
        self.pos[1] = min(max(self.pos[1] + dy, 0), 800)

class ActionMapWindow(QWidget):
    def __init__(self, q_table, get_current_state_func):
        super().__init__()
        self.q_table = q_table
        self.get_current_state_func = get_current_state_func
        self.setWindowTitle("Best Actions (Polar View)")
        self.setGeometry(920, 450, 400, 400)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(500)

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.width() / 2, self.height() / 2)
        max_radius = min(self.width(), self.height()) / 2 - 10

        current_state = self.get_current_state_func()

        for i in range(NUM_ANGLE_SECTORS):
            angle_deg = (i * (360 / NUM_ANGLE_SECTORS) + 90) % 360
            angle_rad = math.radians(angle_deg)

            x = center.x() + max_radius * math.cos(angle_rad)
            y = center.y() + max_radius * math.sin(angle_rad)
            qp.setPen(QPen(Qt.GlobalColor.lightGray, 1))
            qp.drawLine(center, QPointF(x, y))

            lx = center.x() + (max_radius + 10) * math.cos(angle_rad)
            ly = center.y() + (max_radius + 10) * math.sin(angle_rad)
            qp.setPen(QPen(Qt.GlobalColor.black))
            qp.setFont(QFont("Arial", 8))
            angle_label = int(i * (360 / NUM_ANGLE_SECTORS))
            qp.drawText(QPointF(lx - 10, ly), f"{angle_label}°")

        for i in range(1, NUM_DISTANCE_LEVELS):
            r = i * (max_radius / NUM_DISTANCE_LEVELS)
            qp.setPen(QPen(Qt.GlobalColor.lightGray, 1))
            qp.drawEllipse(center, r, r)
            qp.setPen(QPen(Qt.GlobalColor.black))
            qp.setFont(QFont("Arial", 8))
            qp.drawText(QPointF(center.x() + 5, center.y() - r + 12), f"D{i}")

        for angle_sector in range(NUM_ANGLE_SECTORS):
            start_angle = angle_sector * (360 / NUM_ANGLE_SECTORS)
            sector_angle_width = (360 / NUM_ANGLE_SECTORS)
            for dist_level in range(NUM_DISTANCE_LEVELS):
                radius_inner = dist_level * (max_radius / NUM_DISTANCE_LEVELS)
                radius_outer = (dist_level + 1) * (max_radius / NUM_DISTANCE_LEVELS)

                angle_center_deg = start_angle + sector_angle_width / 2
                angle_rad = math.radians(angle_center_deg + 90)
                state = (angle_sector, dist_level)

                if state == current_state:
                    qp.setBrush(QColor(144, 238, 144, 160))  # hellgrün mit Transparenz
                    qp.setPen(Qt.PenStyle.NoPen)
                    path = QPolygonF()
                    for edge_angle in [start_angle, start_angle + sector_angle_width]:
                        for r in [radius_inner, radius_outer]:
                            a_rad = math.radians(edge_angle + 90)
                            xx = center.x() + r * math.cos(a_rad)
                            yy = center.y() + r * math.sin(a_rad)
                            path.append(QPointF(xx, yy))
                    qp.drawPolygon(path)

                best_action = self.get_best_action(state)
                if best_action:
                    r = (radius_inner + radius_outer) / 2
                    xx = center.x() + r * math.cos(angle_rad)
                    yy = center.y() + r * math.sin(angle_rad)
                    self.draw_arrow(qp, QPointF(xx, yy), ACTION_TO_ANGLE[best_action], 8)

    def get_best_action(self, state):
        best_action = None
        best_q = -float('inf')
        for a in ACTIONS:
            q = self.q_table.get((state, a), None)
            if q is not None and q > best_q:
                best_q = q
                best_action = a
        return best_action

    def draw_arrow(self, qp, pos, angle_deg, size):
        angle_rad = math.radians(angle_deg)
        dx = size * math.cos(angle_rad)
        dy = size * math.sin(angle_rad)
        p1 = QPointF(pos.x(), pos.y())
        p2 = QPointF(pos.x() + dx, pos.y() + dy)
        qp.setPen(QPen(Qt.GlobalColor.black, 2))
        qp.drawLine(p1, p2)

        head_size = size / 2
        left_angle = angle_rad + math.radians(150)
        right_angle = angle_rad - math.radians(150)
        left = QPointF(p2.x() + head_size * math.cos(left_angle), p2.y() + head_size * math.sin(left_angle))
        right = QPointF(p2.x() + head_size * math.cos(right_angle), p2.y() + head_size * math.sin(right_angle))
        arrow_head = QPolygonF([p2, left, right])
        qp.setBrush(Qt.GlobalColor.black)
        qp.drawPolygon(arrow_head)

class QLearningDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-Learning Visualisierung")
        self.setGeometry(100, 100, 800, 800)

        self.counter = 0
        self.agent = QLearningAgent()
        self.goal = self.agent.random_pos()
        self.prev_distance = self.distance(self.agent.pos, self.goal)
        self.last_state = self.agent.get_state(self.goal)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_env)
        self.timer.start(5)

        self.polar_map = ActionMapWindow(self.agent.q_table, lambda: self.last_state)
        self.polar_map.show()

    def distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def update_env(self):
        self.counter += 1

        state = self.agent.get_state(self.goal)
        action = self.agent.choose_action(state)
        self.agent.move(action)

        new_state = self.agent.get_state(self.goal)
        dist = self.distance(self.agent.pos, self.goal)

        if False and self.counter % 100 == 0:
            print(self.agent.q_table)

        if dist < 15:
            reward = 100
            self.goal = self.agent.random_pos()
            self.prev_distance = self.distance(self.agent.pos, self.goal)
            #print(self.agent.q_table)
        else:
            reward = self.prev_distance - dist
            #reward = 0.0
            self.prev_distance = dist

        self.agent.update_q(state, action, reward, new_state)
        self.last_state = new_state
        self.repaint()

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setBrush(QColor(255, 0, 0))
        qp.drawEllipse(self.goal[0] - 10, self.goal[1] - 10, 20, 20)
        qp.setBrush(QColor(0, 0, 0))
        qp.drawEllipse(self.agent.pos[0] - 10, self.agent.pos[1] - 10, 20, 20)
        qp.setPen(QPen(QColor(0, 255, 0), 2))
        qp.drawLine(self.agent.pos[0], self.agent.pos[1], self.goal[0], self.goal[1])

        if self.last_state:
            qp.setPen(QPen(QColor(0, 0, 255)))
            qp.setFont(QFont("Arial", 10))
            x, y = self.agent.pos
            offset = 25
            angle_label = self.last_state[0] * (360 // NUM_ANGLE_SECTORS)
            qp.drawText(x + 30, y - 50,
                        f"Zustand: Winkel={angle_label}°, Distanz-Level={self.last_state[1]}")
            for i, action in enumerate(ACTIONS):
                q_val = self.agent.q_table.get((self.last_state, action), 0.0)
                qp.drawText(x + 30, y + i * offset - 30, f"{action}: {q_val:.2f}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = QLearningDemo()
    demo.show()
    sys.exit(app.exec())
