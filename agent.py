import random
from collections import deque   # Practical 03: FIFO frontier for BFS
import heapq                    # Practical 03: priority-queue frontier for UCS


class GreedyGridAgent:
    """A simple agent that wanders randomly to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """Condition-action rules only."""

    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            return 'Up'
        if percept['wall_ahead']:
            return 'Left'
        return 'Up'


class ModelBasedAgent:
    """Reflex rules + an internal model of the world."""

    DIRECTIONS = ['Up', 'Right', 'Down', 'Left']
    DIR_VECTORS = {
        'Up': (0, 1),
        'Right': (1, 0),
        'Down': (0, -1),
        'Left': (-1, 0)
    }

    def __init__(self):
        self.est_pos = (0, 0)
        self.visited_cells = {(0, 0)}
        self.last_action = None
        self.tried_while_blocked = []

    def _update_state(self, percept: dict):
        if self.last_action in self.DIR_VECTORS and not percept.get('bumped', False):
            dx, dy = self.DIR_VECTORS[self.last_action]
            self.est_pos = (self.est_pos[0] + dx, self.est_pos[1] + dy)
            self.visited_cells.add(self.est_pos)

    def _decide(self, percept: dict) -> str:
        if percept['wall_ahead']:
            if (
                self.last_action in self.DIRECTIONS
                and self.last_action not in self.tried_while_blocked
            ):
                self.tried_while_blocked.append(self.last_action)

            candidates = [
                d for d in self.DIRECTIONS
                if d not in self.tried_while_blocked
            ]

            if not candidates:
                self.tried_while_blocked = []
                candidates = list(self.DIRECTIONS)

            def leads_to_new_cell(direction):
                dx, dy = self.DIR_VECTORS[direction]
                next_pos = (self.est_pos[0] + dx, self.est_pos[1] + dy)
                return next_pos not in self.visited_cells

            unexplored = [d for d in candidates if leads_to_new_cell(d)]
            action = (unexplored or candidates)[0]

            self.tried_while_blocked.append(action)
            return action

        self.tried_while_blocked = []
        return (
            self.last_action
            if self.last_action in self.DIRECTIONS
            else 'Up'
        )

    def sense_and_act(self, percept: dict) -> str:
        self._update_state(percept)
        action = self._decide(percept)
        self.last_action = action
        return action


class SearchAgent:
    """Practical 03 -- Goal-Based / Planning Agent."""

    DIR_VECTORS = {
        'Up': (0, 1),
        'Down': (0, -1),
        'Left': (-1, 0),
        'Right': (1, 0)
    }

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

    def _successors(self, state, walls, grid_size):
        """Return legal (action, next_state) pairs."""
        width, height = grid_size
        x, y = state

        for action, (dx, dy) in self.DIR_VECTORS.items():
            nx, ny = x + dx, y + dy

            if (
                0 <= nx < width
                and 0 <= ny < height
                and (nx, ny) not in walls
            ):
                yield action, (nx, ny)

    def bfs_search(self, start, goal, walls, grid_size):
        """Breadth-First Search using a FIFO queue."""
        frontier = deque([(start, [])])
        reached = {start}

        while frontier:
            state, path = frontier.popleft()

            if state == goal:
                return path

            for action, nxt in self._successors(state, walls, grid_size):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))

        return []

    def dfs_search(self, start, goal, walls, grid_size):
        """Depth-First Search using a LIFO stack."""
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            state, path = frontier.pop()

            if state == goal:
                return path

            for action, nxt in self._successors(state, walls, grid_size):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))

        return []

    def ucs_search(self, start, goal, walls, grid_size):
        """Uniform-Cost Search using a priority queue ordered by g(n)."""
        tie = 0
        frontier = [(0, tie, start, [])]
        best_cost = {start: 0}

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)

            if state == goal:
                return path

            if cost > best_cost.get(state, float('inf')):
                continue

            for action, nxt in self._successors(state, walls, grid_size):
                step_cost = 1
                new_cost = cost + step_cost

                if new_cost < best_cost.get(nxt, float('inf')):
                    best_cost[nxt] = new_cost
                    tie += 1
                    heapq.heappush(
                        frontier,
                        (new_cost, tie, nxt, path + [action])
                    )

        return []

    def sense_and_act(self, percept: dict) -> str:
        """Build a plan to the closest food and execute one action."""
        if not self.plan:
            all_food = percept.get('all_food', [])

            if not all_food:
                return random.choice(['Up', 'Down', 'Left', 'Right'])

            start = tuple(percept['agent_pos'])
            walls = set(map(tuple, percept['walls']))
            grid_size = percept['grid_size']

            # Closest food by Manhattan distance.
            goal = min(
                all_food,
                key=lambda food:
                    abs(food[0] - start[0]) + abs(food[1] - start[1])
            )
            goal = tuple(goal)

            search = {
                'BFS': self.bfs_search,
                'DFS': self.dfs_search,
                'UCS': self.ucs_search,
            }[self.active_algo]

            self.plan = search(start, goal, walls, grid_size)

            if not self.plan:
                return random.choice(['Up', 'Down', 'Left', 'Right'])

        return self.plan.pop(0)