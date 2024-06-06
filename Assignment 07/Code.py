# Q1

import heapq
import sys


def apply_dijkstra(n, start, adj_list):
    shortest_path = []

    for x in range(n):
        shortest_path.append(sys.maxsize)

    shortest_path[start] = 0
    visited_set = set()

    pq = []
    heapq.heapify(pq)

    heapq.heappush(pq, [0, start])

    while pq:
        cost, node = heapq.heappop(pq)
        shortest_path[node] = min(shortest_path[node], cost)
        visited_set.add(node)

        for adj_ele in adj_list[node]:
            adj_node, adj_cost = adj_ele
            if adj_node not in visited_set:
                heapq.heappush(pq, [adj_cost + cost, adj_node])

    return shortest_path


def specialHub(n, edges, distanceThreshold):
    adj_list = []

    for x in range(n):
        adj_list.append([])

    for edge in edges:
        start, end, cost = edge
        adj_list[start].append([end, cost])
        adj_list[end].append([start, cost])

    min_count = n + 1
    max_city = -1

    for x in range(n):
        shortest_path = apply_dijkstra(n, x, adj_list)
        print(shortest_path)
        count = 0
        for y in range(len(shortest_path)):
            if shortest_path[y] <= distanceThreshold:
                count += 1

        if count <= min_count:
            min_count = count
            max_city = x

    return max_city


# print(
#     specialHub(
#         n=5,
#         edges=[[0, 1, 2], [0, 4, 8], [1, 2, 3], [1, 4, 2], [2, 3, 1], [3, 4, 1]],
#         distanceThreshold=2,
#     )
# )


# Q2

import heapq


def challenge(n, connections):
    number_of_edges = 0
    pq = []
    heapq.heapify(pq)

    for edge in connections:
        node1, node2, cost = edge
        heapq.heappush(pq, [cost, [node1, node2]])

    minimum_cost = 0
    adj_list = []

    for x in range(n):
        adj_list.append([])

    while pq:
        edge = heapq.heappop(pq)
        cost, nodes = edge
        node1, node2 = nodes

        if not formsCycle(node1 - 1, node2 - 1, adj_list):
            number_of_edges += 1
            adj_list[node1 - 1].append(node2 - 1)
            adj_list[node2 - 1].append(node1 - 1)
            minimum_cost += cost

        if number_of_edges == n - 1:
            break

    if number_of_edges != n - 1:
        return -1

    return minimum_cost


def formsCycle(node1, node2, adj_list):
    stack = []
    stack.append(node1)

    visited_set = set()

    while stack:
        node = stack.pop()
        visited_set.add(node)

        for neighbor in adj_list[node]:
            if neighbor == node2:
                return True
            if neighbor not in visited_set:
                visited_set.add(neighbor)
                stack.append(neighbor)

    return False


# print(
#     challenge(n=6, connections=[[1, 2, 2], [2, 3, 3], [3, 4, 1], [4, 5, 2], [5, 6, 1]])
# )


# Q3

import sys


def min_cost_to_supply_water(n, wells, pipes):
    adj_matrix = [[-1] * n for _ in range(n)]

    if len(pipes) == 0:
        return sum(wells)

    for pipe in pipes:
        if adj_matrix[pipe[0] - 1][pipe[1] - 1] != -1:
            adj_matrix[pipe[0] - 1][pipe[1] - 1] = min(
                adj_matrix[pipe[0] - 1][pipe[1] - 1], pipe[2]
            )
            adj_matrix[pipe[1] - 1][pipe[0] - 1] = min(
                adj_matrix[pipe[1] - 1][pipe[0] - 1], pipe[2]
            )
        else:
            adj_matrix[pipe[0] - 1][pipe[1] - 1] = pipe[2]
            adj_matrix[pipe[1] - 1][pipe[0] - 1] = pipe[2]

    adj_list = []
    for x in range(n):
        adj_list.append([])

    for x in range(n):
        for y in range(x + 1, n):
            if adj_matrix[x][y] != -1:
                adj_list[x].append([y, adj_matrix[x][y]])
                adj_list[y].append([x, adj_matrix[y][x]])

    min_cost = 0

    for x in range(1, n):
        min_temp = sys.maxsize
        if len(adj_list[x]) == 0:
            min_cost += wells[x]
        for y in range(len(adj_list[x])):
            if adj_list[x][y][0] < x:
                min_temp = min(min_temp, adj_list[x][y][1])
                if x == 1:
                    if min_temp == sys.maxsize:
                        min_cost += wells[0] + wells[1]
                    else:
                        min_cost += min(
                            wells[0] + wells[1], min_temp + min(wells[0], wells[1])
                        )
                else:
                    min_cost += min(min_temp, wells[x])

    return min_cost


# print(min_cost_to_supply_water(n=2, wells=[4, 5], pipes=[]))
# print(min_cost_to_supply_water(n=3, wells=[1, 2, 2], pipes=[[1, 2, 1], [2, 3, 1]]))


# Q4

import heapq
import sys


def cheapestRoutes(s, prices):
    # Write your code
    n = len(prices)
    adj_list = []

    for x in range(n):
        adj_list.append([])

    for x in range(n):
        for y in range(n):
            if prices[x][y] != -1:
                adj_list[x].append([y, prices[x][y]])

    pq = []
    visited_set = set()

    shortest_path = []
    for x in range(n):
        shortest_path.append(sys.maxsize)

    shortest_path[s] = 0

    heapq.heapify(pq)
    heapq.heappush(pq, [0, s])

    while pq:
        cost, node = heapq.heappop(pq)
        visited_set.add(node)
        shortest_path[node] = min(shortest_path[node], cost)

        for adj_ele in adj_list[node]:
            adj_node, adj_cost = adj_ele
            if adj_node not in visited_set:
                heapq.heappush(pq, [adj_cost + cost, adj_node])

    for x in range(len(shortest_path)):
        if shortest_path[x] == sys.maxsize:
            shortest_path[x] = -1

    return shortest_path


# print(
#     cheapestRoutes(
#         s=0,
#         prices=[
#             [0, -1, 4, 1, -1],
#             [-1, 0, -1, -1, -1],
#             [4, -1, 0, -1, 2],
#             [1, -1, -1, 0, 3],
#             [-1, -1, 2, 3, 0],
#         ],
#     )
# )


# Q5

from collections import deque


def isDividePossible(n, connected_houses):
    adj_list = []

    for x in range(n):
        adj_list.append([])

    for connected_house in connected_houses:
        adj_list[connected_house[0]].append(connected_house[1])
        adj_list[connected_house[1]].append(connected_house[0])

    explored_set = set()
    red_set = set()
    blue_set = set()

    queue = deque()

    # starting with 0, True)
    queue.append([0, True])
    red_set.add(0)

    while queue:
        node, red_group = queue.popleft()
        explored_set.add(node)

        for neighbor in adj_list[node]:
            if neighbor in explored_set:
                continue
            if red_group:
                new_group = False
                parent_set = red_set
                child_set = blue_set
            else:
                new_group = True
                parent_set = blue_set
                child_set = red_set

            if neighbor not in explored_set and neighbor in parent_set:
                return False
            child_set.add(neighbor)
            queue.append([neighbor, new_group])

    return True


# print(
#     isDividePossible(
#         n=8,
#         connected_houses=[
#             [0, 1],
#             [0, 3],
#             [1, 4],
#             [1, 5],
#             [3, 4],
#             [3, 7],
#             [4, 6],
#             [5, 6],
#             [6, 7],
#         ],
#     )
# )


# Q6

import heapq


def minCostToConnectHubs(hubs):
    n = len(hubs)
    edge_list = []

    for x in range(n):
        for y in range(x + 1, n):
            manhattan_dist = abs(hubs[x][0] - hubs[y][0]) + abs(hubs[x][1] - hubs[y][1])
            edge_list.append([x, y, manhattan_dist])

    pq = []
    heapq.heapify(pq)

    for edge in edge_list:
        heapq.heappush(pq, [edge[2], edge[0], edge[1]])

    visited_set = set()
    number_of_edges = 0
    min_cost = 0

    adj_list = []
    for x in range(n):
        adj_list.append([])

    while pq:
        edge = heapq.heappop(pq)
        if not formsCycle(edge[1], edge[2], adj_list):
            number_of_edges += 1
            min_cost += edge[0]
            adj_list[edge[1]].append(edge[2])
            adj_list[edge[2]].append(edge[1])

        if number_of_edges == n - 1:
            break

    return min_cost


def formsCycle(node1, node2, adj_list):
    stack = []
    stack.append(node1)

    visited_set = set()

    while stack:
        node = stack.pop()
        visited_set.add(node)

        for neighbor in adj_list[node]:
            if neighbor == node2:
                return True
            if neighbor not in visited_set:
                visited_set.add(neighbor)
                stack.append(neighbor)

    return False


# print(minCostToConnectHubs(hubs=[[1, 6], [8, 0], [14, 7], [9, 3], [2, 9], [5, 5]]))


# Q7

from collections import deque


def findMaxSuccessPath(n, edges, prob, start_node, end_node):
    number_of_edges = len(edges)
    adj_list = []

    for x in range(n):
        adj_list.append([])

    for x in range(number_of_edges):
        edge_start, edge_end = edges[x]
        edge_prob = prob[x]

        adj_list[edge_start].append([edge_end, edge_prob])
        adj_list[edge_end].append([edge_start, edge_prob])

    queue = deque()
    visited_list = []
    queue.append([start_node, visited_list, 1])

    max_prob = 0

    while queue:
        node, visited_list, prob = queue.popleft()

        if node == end_node:
            max_prob = max(max_prob, prob)
        else:
            for neighbor in adj_list[node]:
                new_prob = prob
                if neighbor[0] not in visited_list:
                    visited_list.append(neighbor[0])
                    new_prob *= neighbor[1]
                    queue.append([neighbor[0], visited_list, new_prob])

    return max_prob


# print(
#     findMaxSuccessPath(
#         n=3,
#         edges=[[0, 1], [1, 2], [0, 2]],
#         prob=[0.5, 0.5, 0.8],
#         start_node=0,
#         end_node=2,
#     )
# )


# Q8

from collections import deque


def minimumTimeToVisit(grid):
    grid_row = len(grid)
    grid_col = len(grid[0])

    queue = deque()
    queue.append([0, 0, 0])

    visited_set = set()

    while queue:
        grid_loc = queue.popleft()
        row, col, threshold = grid_loc
        visited_set.add((row, col))

        if row == grid_row - 1 and col == grid_col - 1:
            return threshold

        visited_set_temp = []
        unvisited_count = 0

        if col + 1 < grid_col and grid[row][col + 1] <= threshold + 1:
            if (row, col + 1) in visited_set:
                visited_set_temp.append([row, col + 1])
            else:
                queue.append([row, col + 1, threshold + 1])
                unvisited_count += 1

        if row - 1 >= 0 and grid[row - 1][col] <= threshold + 1:
            if (row - 1, col) in visited_set:
                visited_set_temp.append([row - 1, col])
            else:
                queue.append([row - 1, col, threshold + 1])
                unvisited_count += 1

        if col - 1 >= 0 and grid[row][col - 1] <= threshold + 1:
            if (row, col - 1) in visited_set:
                visited_set_temp.append([row, col - 1])
            else:
                queue.append([row, col - 1, threshold + 1])
                unvisited_count += 1

        if row + 1 < grid_row and grid[row + 1][col] <= threshold + 1:
            if (row, col - 1) in visited_set:
                visited_set_temp.append([row + 1, col])
            else:
                queue.append([row + 1, col, threshold + 1])
                unvisited_count += 1

        if unvisited_count == 0:
            for cell in visited_set_temp:
                row, col = cell
                queue.append([row, col, threshold + 1])

    return -1


# print(minimumTimeToVisit(grid=[[0, 1, 1], [1, 2, 1], [1, 3, 0]]))

# print(
#     minimumTimeToVisit(
#         grid=[
#             [0, 0, 0, 0, 1],
#             [0, 1, 1, 1, 1],
#             [0, 1, 0, 0, 0],
#             [0, 1, 0, 1, 1],
#             [0, 0, 0, 1, 0],
#         ]
#     )
# )


# Q9

import heapq
import sys
from collections import deque


def WeightLimitedPathsExist(n, edgelist, querylist):
    adj_list = []

    for x in range(n):
        adj_list.append([])

    for edge in edgelist:
        adj_list[edge[0]].append([edge[1], edge[2]])
        adj_list[edge[1]].append([edge[0], edge[2]])

    adj_matrix = []
    list = []

    for x in range(n):
        list.append(-1)

    for x in range(n):
        adj_matrix.append(list)

    for x in range(n):
        for y in range(n):
            if x == y:
                adj_matrix[x][y] = 0

    for x in range(n):
        queue = deque()
        visited_list = []

        queue.append([x, visited_list, sys.maxsize])
        visited_list.append(x)

        while queue:
            node, visited_list, min_weight = queue.popleft()

            for neighbor in adj_list[node]:
                if neighbor[0] not in visited_list:
                    visited_list.append(neighbor[0])
                    new_weight = min(min_weight, neighbor[1])
                    queue.append([neighbor[0], visited_list, new_weight])
                    adj_matrix[node][neighbor[0]] = max(
                        adj_matrix[node][neighbor[0]], new_weight
                    )

    result = []
    for query in querylist:
        if adj_matrix[query[0]][query[1]] >= query[2]:
            result.append(True)
        else:
            result.append(False)

    return result


# print(
#     WeightLimitedPathsExist(
#         n=6,
#         edgelist=[[0, 2, 4], [0, 3, 2], [1, 2, 3], [2, 3, 1], [4, 5, 5]],
#         querylist=[[2, 3, 1], [1, 3, 3], [2, 0, 3], [0, 5, 6]],
#     )
# )


# Q10


def shortestFareRoute(start, target, specialRoads):
    if start == target:
        return 0

    node_list = []
    edge_list = []

    for bridge in specialRoads:
        from_node = [bridge[0], bridge[1]]
        to_node = [bridge[2], bridge[3]]
        cost = bridge[4]

        if from_node not in node_list:
            node_list.append(from_node)

        if to_node not in node_list:
            node_list.append(to_node)

        n = len(node_list)

        edge_list.append([n - 2, n - 1, cost])

    for x in range(len(node_list)):
        manhattan_dist = abs(node_list[x][0] - start[0]) + abs(
            node_list[x][1] - start[1]
        )
        edge_list.append([x, len(node_list), manhattan_dist])

    node_list.append(start)
    start_index = len(node_list) - 1

    for x in range(len(node_list)):
        manhattan_dist = abs(node_list[x][0] - target[0]) + abs(
            node_list[x][1] - target[1]
        )
        edge_list.append([x, len(node_list), manhattan_dist])

    node_list.append(target)
    target_index = len(node_list) - 1

    n = len(node_list)
    adj_list = []

    for x in range(n):
        adj_list.append([])

    for edge in edge_list:
        adj_list[edge[0]].append([edge[1], edge[2]])
        adj_list[edge[1]].append([edge[0], edge[2]])

    pq = []
    heapq.heapify(pq)

    min_fare = []
    for x in range(n):
        min_fare.append(sys.maxsize)

    heapq.heappush(pq, [0, start_index])
    visited_set = set()

    while pq:
        cost, node = heapq.heappop(pq)
        min_fare[node] = min(min_fare[node], cost)
        visited_set.add(node)
        for adj in adj_list[node]:
            if adj[0] not in visited_set:
                heapq.heappush(pq, [cost + adj[1], adj[0]])

    return min_fare[target_index]


# print(
#     shortestFareRoute(
#         start=[5, 5],
#         target=[10, 10],
#         specialRoads=[[3, 3, 4, 4, 5], [5, 5, 7, 5, 2], [7, 5, 7, 10, 6]],
#     )
# )
