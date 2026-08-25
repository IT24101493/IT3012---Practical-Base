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
    """Condition-action rules only -- deliberately no __init__ storing
    history, per the Step 1.2 spec. Given the same percept, it will
    always return the same action; that's the whole point.

        IF food_here      THEN stay put (already collected on arrival)
        IF wall_ahead      THEN try 'Left'
        ELSE                    keep going 'Up'
    """

    def sense_and_act(self, percept: dict) -> str:
        if percept['food_here']:
            return 'Up'
        if percept['wall_ahead']:
            return 'Left'
        return 'Up'


class ModelBasedAgent:
    """Reflex rules + an internal model of the world.

    The agent is never told its (x, y) position -- that's the whole
    point of partial observability -- so it keeps its own belief about
    where it is (est_pos) built purely from the actions it chose and
    whether the environment's bump sensor ('bumped' in the percept)
    says they actually succeeded. That's the Transition Model: "if I
    moved Up and I wasn't bumped, my y went up by one."

    tried_while_blocked remembers which directions it has already
    attempted since it last got stuck at the current spot, so a wall
    that keeps producing the same wall_ahead=True percept doesn't make
    it repeat the same doomed action -- unlike SimpleReflexAgent.
    """

    DIRECTIONS = ['Up', 'Right', 'Down', 'Left']
    DIR_VECTORS = {'Up': (0, 1), 'Right': (1, 0), 'Down': (0, -1), 'Left': (-1, 0)}

    def __init__(self):
        self.est_pos = (0, 0)           # believed position, dead-reckoned
        self.visited_cells = {(0, 0)}   # cells the agent believes it has occupied
        self.last_action = None         # action chosen on the previous turn
        self.tried_while_blocked = []   # directions already tried since getting stuck here

    def _update_state(self, percept: dict):
        """Transition model: fold the effect of our last action into our
        belief state before deciding what to do next. percept.get(...)
        defaults 'bumped' to False so this also works against the bare
        {'wall_ahead', 'food_here'} percepts used in unit tests, where
        position tracking isn't what's being exercised."""
        if self.last_action in self.DIR_VECTORS and not percept.get('bumped', False):
            dx, dy = self.DIR_VECTORS[self.last_action]
            self.est_pos = (self.est_pos[0] + dx, self.est_pos[1] + dy)
            self.visited_cells.add(self.est_pos)

    def _decide(self, percept: dict) -> str:
        if percept['wall_ahead']:
            # We know at least one direction is blocked: whichever way we
            # were last facing -- that's *why* wall_ahead is True now.
            if self.last_action in self.DIRECTIONS and self.last_action not in self.tried_while_blocked:
                self.tried_while_blocked.append(self.last_action)

            candidates = [d for d in self.DIRECTIONS if d not in self.tried_while_blocked]
            if not candidates:
                # Tried every direction from this spot -- boxed in, start over.
                self.tried_while_blocked = []
                candidates = list(self.DIRECTIONS)

            def leads_to_new_cell(d):
                dx, dy = self.DIR_VECTORS[d]
                return (self.est_pos[0] + dx, self.est_pos[1] + dy) not in self.visited_cells

            unexplored = [d for d in candidates if leads_to_new_cell(d)]
            action = (unexplored or candidates)[0]

            self.tried_while_blocked.append(action)
            return action

        # Either the way ahead is clear, or we're standing on food that's
        # already been auto-collected -- either way, keep heading the way
        # we were already going instead of resetting to some fixed
        # direction every turn (that fixed-default version is what got
        # ModelBasedAgent stuck oscillating in corners during testing --
        # it kept walking back the way it came instead of continuing on).
        self.tried_while_blocked = []
        return self.last_action if self.last_action in self.DIRECTIONS else 'Up'

    def sense_and_act(self, percept: dict) -> str:
        self._update_state(percept)
        action = self._decide(percept)
        self.last_action = action
        return action


class SearchAgent:
    """Practical 03 -- Goal-Based / Planning Agent.

    Unlike the reflex agents above, this agent doesn't react to the
    current percept: it uses the *world model* exposed in the percept
    (grid_size, walls, all_food, agent_pos) to SIMULATE future states
    offline, build a complete plan of actions to the nearest food, and
    then execute that plan one step at a time.

    All three algorithms share the exact same node-expansion skeleton;
    the ONLY thing that changes is the Frontier data structure:

        BFS -> FIFO queue  (deque.popleft)  : shallowest node first
        DFS -> LIFO stack  (list.pop)       : deepest node first
        UCS -> Priority Q  (heapq.heappop)  : cheapest g(n) first

    Each also keeps a `reached` set, turning Tree Search into Graph
    Search: a state is expanded at most once, so cycles in the grid
    can never trap the algorithm in an infinite loop.
    """

    DIR_VECTORS = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}

    def __init__(self):
        self.plan = []              # Step 1.3: queued actions from the last search
        self.active_algo = 'BFS'    # Step 1.3: switch to 'DFS' or 'UCS' to compare

    # ------------------------------------------------------------------
    # Successor function: given a state (x, y), which (action, state')
    # pairs are legal? This is the agent's transition model of the world.
    # ------------------------------------------------------------------
    def _successors(self, state, walls, grid_size):
        width, height = grid_size
        x, y = state
        for action, (dx, dy) in self.DIR_VECTORS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                yield action, (nx, ny)

    # ------------------------------------------------------------------
    # Step 1.2 -- the three uninformed search strategies
    # Each returns a list of actions from start to goal, or [] if the
    # goal is unreachable.
    # ------------------------------------------------------------------
    def bfs_search(self, start, goal, walls, grid_size):
        """Breadth-First Search: FIFO frontier -> shallowest node first.
        Optimal here because every step costs the same (1)."""
        frontier = deque([(start, [])])     # (state, actions-so-far)
        reached = {start}                   # graph search: no re-expansion
        while frontier:
            state, path = frontier.popleft()    # FIFO
            if state == goal:
                return path
            for action, nxt in self._successors(state, walls, grid_size):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))
        return []

    def dfs_search(self, start, goal, walls, grid_size):
        """Depth-First Search: LIFO frontier -> deepest node first.
        Complete (thanks to `reached`) but NOT optimal: it commits to
        one branch all the way down before backtracking, producing the
        winding routes you see in the simulation."""
        frontier = [(start, [])]            # plain list used as a stack
        reached = {start}
        while frontier:
            state, path = frontier.pop()        # LIFO
            if state == goal:
                return path
            for action, nxt in self._successors(state, walls, grid_size):
                if nxt not in reached:
                    reached.add(nxt)
                    frontier.append((nxt, path + [action]))
        return []

    def ucs_search(self, start, goal, walls, grid_size):
        """Uniform-Cost Search: priority queue ordered by total path
        cost g(n). With a uniform step cost of 1 this expands nodes in
        the same order as BFS, but the machinery generalises to any
        (non-negative) step costs -- e.g. weighting toxic-trap cells.
        `tie` is a counter so heapq never has to compare tuples of
        equal cost by their (uncomparable) contents."""
        tie = 0
        frontier = [(0, tie, start, [])]    # (g(n), tie, state, actions)
        best_cost = {start: 0}              # reached, with cheapest g(n) found
        while frontier:
            cost, _, state, path = heapq.heappop(frontier)   # cheapest first
            if state == goal:
                return path
            if cost > best_cost.get(state, float('inf')):
                continue    # stale entry: a cheaper route was found already
            for action, nxt in self._successors(state, walls, grid_size):
                step_cost = 1               # uniform grid: every move costs 1
                new_cost = cost + step_cost
                if new_cost < best_cost.get(nxt, float('inf')):
                    best_cost[nxt] = new_cost
                    tie += 1
                    heapq.heappush(frontier, (new_cost, tie, nxt, path + [action]))
        return []

    # ------------------------------------------------------------------
    # Step 1.3 -- offline planning + step-by-step plan execution
    # ------------------------------------------------------------------
    def sense_and_act(self, percept: dict) -> str:
        if not self.plan:
            all_food = percept.get('all_food', [])
            if not all_food:
                return random.choice(['Up', 'Down', 'Left', 'Right'])

            start = tuple(percept['agent_pos'])
            walls = set(map(tuple, percept['walls']))
            grid_size = percept['grid_size']

            # Closest food pellet by Manhattan distance
            goal = min(all_food, key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1]))
            goal = tuple(goal)

            search = {
                'BFS': self.bfs_search,
                'DFS': self.dfs_search,
                'UCS': self.ucs_search,
            }[self.active_algo]

            self.plan = search(start, goal, walls, grid_size)

            if not self.plan:   # goal unreachable -- don't crash, wander
                return random.choice(['Up', 'Down', 'Left', 'Right'])

        return self.plan.pop(0)