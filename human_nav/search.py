"""Spatial A* for pedestrians; wall-only distance fields for robot heuristics."""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import count

from .config import Grid, Point


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(
    grid: Grid,
    start: Point,
    goal: Point,
    blocked: set[Point] | None = None,
    order: str = "NESW",
) -> list[Point] | None:
    """Return a shortest route including start and goal, or None.

    Blocks are current occupied cells, not future reservations. The searcher's
    own cell is always allowed. Equal f scores prefer lower h, then direction order.
    """
    blocked = (blocked or set()) - {start}
    if not grid.walkable(start) or not grid.walkable(goal) or goal in blocked:
        return None
    serial = count()
    frontier = [(manhattan(start, goal), manhattan(start, goal), next(serial), 0, start)]
    costs = {start: 0}
    parent: dict[Point, Point] = {}
    while frontier:
        _, _, _, cost, p = heappop(frontier)
        if cost != costs[p]:
            continue
        if p == goal:
            route = [p]
            while p != start:
                p = parent[p]
                route.append(p)
            return route[::-1]
        for q in grid.neighbors(p, order):
            new_cost = cost + 1
            if q in blocked or new_cost >= costs.get(q, float("inf")):
                continue
            costs[q] = new_cost
            parent[q] = p
            h = manhattan(q, goal)
            heappush(frontier, (new_cost + h, h, next(serial), new_cost, q))
    return None


def distance_field(
    grid: Grid,
    goal: Point,
    blocked: set[Point] | None = None,
) -> dict[Point, int]:
    """Reverse distances for heuristics or frozen-map feasibility checks."""
    blocked = blocked or set()
    if goal in blocked or not grid.walkable(goal):
        return {}
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        p = queue.popleft()
        for q in grid.neighbors(p):
            if q not in distances and q not in blocked:
                distances[q] = distances[p] + 1
                queue.append(q)
    return distances
