# simulator.py
import tkinter as tk
import random
from agent import SearchAgent, GreedyGridAgent, SimpleReflexAgent, ModelBasedAgent


class VisualGridHuntGame:
    """Environment class that maintains grid state and percepts."""

    def __init__(self, width=10, height=8, num_food=10, num_opponents=0, num_traps=3, custom_walls=None):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]
        self.facing = 'Right'

        self.walls = set(custom_walls) if custom_walls is not None else {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        self.food_positions = set()
        while len(self.food_positions) < num_food:
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if pos != (0, 0) and pos not in self.walls:
                self.food_positions.add(pos)

        self.toxic_traps = set()
        while len(self.toxic_traps) < num_traps:
            pos = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if pos != (0, 0) and pos not in self.walls and pos not in self.food_positions:
                self.toxic_traps.add(pos)

        self.opponents = []
        while len(self.opponents) < num_opponents:
            op_pos = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
            if tuple(op_pos) != (0, 0) and tuple(op_pos) not in self.walls and tuple(op_pos) not in self.food_positions:
                self.opponents.append(op_pos)

        self.score = 0
        self.steps = 0
        self.collision = False

    def get_percept(self) -> dict:
        """Provides both local boolean percepts and global map data for search algorithms."""
        ahead_x, ahead_y = self.agent_pos
        if self.facing == 'Up':
            ahead_y += 1
        elif self.facing == 'Down':
            ahead_y -= 1
        elif self.facing == 'Left':
            ahead_x -= 1
        elif self.facing == 'Right':
            ahead_x += 1

        is_wall_ahead = (
            ahead_x < 0 or ahead_x >= self.width or
            ahead_y < 0 or ahead_y >= self.height or
            (ahead_x, ahead_y) in self.walls
        )

        return {
            'wall_ahead': is_wall_ahead,
            'food_here': tuple(self.agent_pos) in self.food_positions,
            'toxin_here': tuple(self.agent_pos) in self.toxic_traps,
            'collision': self.collision,
            'score': self.score,
            'remaining_food': len(self.food_positions),
            # Required by SearchAgent:
            'agent_pos': list(self.agent_pos),
            'all_food': list(self.food_positions),
            'walls': list(self.walls),
            'grid_size': (self.width, self.height)
        }

    def execute_action(self, action: str):
        self.steps += 1
        if action in ['Up', 'Down', 'Left', 'Right']:
            self.facing = action

        new_pos = list(self.agent_pos)
        if action == 'Up':
            new_pos[1] = min(self.height - 1, new_pos[1] + 1)
        elif action == 'Down':
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 'Left':
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 'Right':
            new_pos[0] = min(self.width - 1, new_pos[0] + 1)

        if tuple(new_pos) in self.walls:
            self.score -= 5
        else:
            self.agent_pos = new_pos

        tuple_pos = tuple(self.agent_pos)
        if tuple_pos in self.toxic_traps:
            self.score -= 15

        if tuple_pos in self.food_positions:
            self.food_positions.remove(tuple_pos)
            self.score += 20

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 60 or self.collision


class GridGameGUI:
    """Tkinter GUI wrapper to render the environment and run agent decisions."""

    def __init__(self, root, width=10, height=8, num_food=10, num_opponents=0):
        self.root = root
        self.root.title("IT3012 - AI Agent Grid Hunt")

        self.env = VisualGridHuntGame(width=width, height=height, num_food=num_food, num_opponents=num_opponents)
        
        # Select Agent: SearchAgent (BFS/DFS/UCS), ModelBasedAgent, SimpleReflexAgent, or GreedyGridAgent
        self.agent = SearchAgent()
        self.agent.active_algo = 'BFS'  # Options: 'BFS', 'DFS', 'UCS'

        max_canvas_dim = 600
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        self.canvas = tk.Canvas(root, width=self.env.width * self.cell_size, height=self.env.height * self.cell_size, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Start Simulation", command=self.run_loop, font=("Arial", 12), bg="#000066", fg="white")
        self.btn.pack(pady=5)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        # Draw Grid & Walls
        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

        # Draw Traps
        for tx, ty in self.env.toxic_traps:
            offset = self.cell_size * 0.25
            x1 = tx * self.cell_size + offset
            y1 = (self.env.height - 1 - ty) * self.cell_size + offset
            self.canvas.create_rectangle(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="purple")

        # Draw Food
        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="#f59e0b", outline="#d97706")

        # Draw Agent
        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.7, y1 + self.cell_size * 0.7, fill="#000066", outline="#1e3a8a")

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():
                percept = self.env.get_percept()
                action = self.agent.sense_and_act(percept)
                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(text=f"Score: {self.env.score} | Steps: {self.env.steps} | Action: {action}")
                self.root.after(250, step)
            else:
                end_text = f"Finished! Final Score: {self.env.score}"
                self.label.config(text=end_text)
                self.btn.config(state="normal")

        step()


if __name__ == "__main__":
    root = tk.Tk()
    app = GridGameGUI(root, width=10, height=8, num_food=10, num_opponents=0)
    root.mainloop()