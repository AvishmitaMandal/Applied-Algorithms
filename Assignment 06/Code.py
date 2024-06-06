# Q1


def MaximumDeafPeople(PeopleShout):
    n = len(PeopleShout)
    if n == 0:
        return 0

    adj_list = []
    for x in range(n):
        adj_list.append([])

    for i in range(n):
        for j in range(n):
            if i != j:
                person1 = PeopleShout[i]
                person2 = PeopleShout[j]

                person1_x, person1_y, person1_radius = person1
                person2_x, person2_y, person2_radius = person2

                dist_calc = pow(person1_x - person2_x, 2) + pow(
                    person1_y - person2_y, 2
                )
                if dist_calc <= pow(person1_radius, 2):
                    adj_list[i].append(j)

    print(adj_list)

    maximum_connected_components = 1
    stack = []

    for x in range(len(adj_list)):
        visited_nodes = set()
        if x not in visited_nodes:
            connected_components = 0
            stack.append(x)

            while stack:
                ele = stack.pop()
                if ele not in visited_nodes:
                    visited_nodes.add(ele)
                    connected_components += 1

                for neighbor in adj_list[ele]:
                    if neighbor not in visited_nodes:
                        stack.append(neighbor)

            maximum_connected_components = max(
                maximum_connected_components, connected_components
            )

    return maximum_connected_components


# PeopleShout = [[2, 1, 3], [6, 1, 4]]
# print(MaximumDeafPeople(PeopleShout))


# Q2

import heapq
import sys


def cheapest_path(n, costs, start):
    # return list of ints
    prices = []
    for x in range(n):
        prices.append(sys.maxsize)
    prices[start] = 0

    adj_list = []
    for x in range(n):
        adj_list.append([])

    for i in range(n):
        for j in range(n):
            if i != j and costs[i][j] != 0:
                adj_list[i].append((j, costs[i][j]))

    pq = []
    heapq.heapify(pq)
    heapq.heappush(pq, (0, start))
    visited_set = set()

    while len(visited_set) < n:
        cost, node = heapq.heappop(pq)
        visited_set.add(node)

        for x in range(len(adj_list[node])):
            child_node, child_node_cost = adj_list[node][x]
            if child_node not in visited_set:
                new_cost = min(cost + child_node_cost, prices[child_node])
                prices[child_node] = new_cost
                heapq.heappush(pq, (new_cost, child_node))

    return prices


# print(cheapest_path(n=3, costs=[[0, 15, 0], [1, 0, 1], [1, 1, 0]], start=0))

# Q3

import copy


def calc_overlap(i, j, words):
    word1 = words[i]
    word2 = words[j]

    max_overlap = 0

    for i in range(1, min(len(word1), len(word2)) + 1):
        overlap1 = word1[-i:] == word2[:i]
        overlap2 = word2[-i:] == word1[:i]

        max_overlap = max(max_overlap, i) if overlap1 or overlap2 else max_overlap

    if word1 in word2:
        max_overlap = max(max_overlap, len(word1))
    elif word2 in word1:
        max_overlap = max(max_overlap, len(word2))

    return max_overlap


def overlap(i, j, words):
    word1 = words[i]
    word2 = words[j]

    max_overlap = 0
    overlap_position = None

    if word1 in word2:
        return word2
    elif word2 in word1:
        return word1

    for i in range(1, min(len(word1), len(word2)) + 1):
        overlap1 = word1[-i:] == word2[:i]
        overlap2 = word2[-i:] == word1[:i]

        if overlap1 or overlap2:
            max_overlap = i
            overlap_position = "suffix1_prefix2" if overlap1 else "suffix2_prefix1"

    if overlap_position == "suffix1_prefix2":
        return word1 + word2[max_overlap:]
    elif overlap_position == "suffix2_prefix1":
        return word2 + word1[max_overlap:]
    else:
        return word1 + word2


def generate_password(words):
    while len(words) >= 2:
        n = len(words)
        max_overlap = 0
        first_string_index, second_string_index = -1, -1
        for i in range(n):
            for j in range(i + 1, n):
                overlap_count = calc_overlap(i, j, words)
                if overlap_count >= max_overlap:
                    max_overlap = overlap_count
                    first_string_index = i
                    second_string_index = j

        overlapping_string = overlap(first_string_index, second_string_index, words)

        new_words = []
        new_words.append(overlapping_string)

        for x in range(len(words)):
            if x != first_string_index and x != second_string_index:
                new_words.append(words[x])

        words = copy.deepcopy(new_words)

    return words[0]


# print(generate_password(words=["XYY", "YYX"]))

# Q4

import copy


def maximumPeople(personHeight, roomHeight):
    visited_set = set()

    personHeight_copy = copy.deepcopy(personHeight)
    personHeight_copy.sort()

    roomHeight_copy = copy.deepcopy(roomHeight)

    count = 0
    for person in personHeight_copy:
        i = 0
        while i < len(roomHeight_copy) and person <= roomHeight_copy[i]:
            i += 1
        if i == 0:
            return count
        else:
            j = i - 1
            while j >= 0 and j in visited_set:
                j -= 1
            if j >= 0:
                visited_set.add(j)
                count += 1

    return count


# print(
#     maximumPeople(
#         personHeight=[12, 6, 32, 33, 20, 33, 22, 18, 26, 3, 29, 24, 31, 1, 32, 8, 14],
#         roomHeight=[
#             9,
#             12,
#             28,
#             12,
#             19,
#             27,
#             4,
#             20,
#             15,
#             13,
#             1,
#             12,
#             16,
#             30,
#             5,
#             24,
#             22,
#             6,
#             32,
#             8,
#         ],
#     )
# )

# Q5


def decoded(first_string_list, second_string_list, n, encoded):
    first_string = "".join(first_string_list)
    second_string = "".join(second_string_list)

    if n % 2 == 0:
        return first_string + second_string
    else:
        return first_string + encoded[n // 2] + second_string


def smallestString(encoded: str) -> str:
    n = len(encoded)
    if n % 2 == 0:
        first_string = encoded[0 : n // 2]
        second_string = encoded[n // 2 : n]
    else:
        first_string = encoded[0 : n // 2]
        second_string = encoded[n // 2 + 1 : n]

    first_string_list = []
    second_string_list = []

    for c in first_string:
        first_string_list.append(c)

    for c in second_string:
        second_string_list.append(c)

    for x in range(len(first_string_list)):
        if first_string_list[x] != "a":
            first_string_list[x] = "a"
            return decoded(first_string_list, second_string_list, n, encoded)

    for x in range(len(second_string_list)):
        if second_string_list[x] != "a":
            second_string_list[x] = "a"
            return decoded(first_string_list, second_string_list, n, encoded)

    # if all are a's then
    return encoded[0 : len(encoded) - 1] + "b"


# print(smallestString(encoded="deified"))


# Q6


def grandTour(checkpoints):
    n = len(checkpoints)
    edge_vector = []
    for x in range(n):
        edge_vector.append(0)
    edge_vector[0] = 1
    paths = []

    paths = hamiltonian_path(1, edge_vector, n, checkpoints, paths)
    if len(paths) != 0:
        return True
    return False


def hamiltonian_path(k, edge_vector, n, graph, paths):
    while k < n:
        next_vertex(k, edge_vector, n, graph)
        if edge_vector[k] == 0:
            return paths
        if k == n - 1:
            paths.append(edge_vector)
            return paths
        else:
            hamiltonian_path(k + 1, edge_vector, n, graph, paths)


def next_vertex(k, edge_vector, n, graph):
    while k < n:
        edge_vector[k] = (edge_vector[k] + 1) % (n + 1)
        if edge_vector[k] == 0:
            return
        if graph[edge_vector[k - 1] - 1][edge_vector[k] - 1] == 1:
            j = 0
            while j < k:
                if edge_vector[j] == edge_vector[k]:
                    break
                j += 1
            if j == k:
                if (
                    k == n - 1
                    and graph[edge_vector[n - 1] - 1][edge_vector[0] - 1] == 1
                ) or k < n - 1:
                    return


# print(
#     grandTour(
#         checkpoints=[
#             [0, 1, 1, 1, 0],
#             [1, 0, 1, 0, 1],
#             [1, 1, 0, 1, 1],
#             [1, 0, 1, 0, 1],
#             [0, 1, 1, 1, 0],
#         ]
#     )
# )
