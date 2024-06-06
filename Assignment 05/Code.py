# Q1

import heapq


def busRemaining(busStation):
    n = len(busStation)
    if n == 0 or n == 1:
        return n

    heap = []
    for station in busStation:
        heapq.heappush(heap, (station[0], station))

    busRemainingList = []
    station1 = []

    while heap:
        if len(station1) == 0:
            (station1_start, station1) = heapq.heappop(heap)
        (station2_start, station2) = heapq.heappop(heap)

        if isOverlapping(station1, station2):
            station1 = overlappedStation(station1, station2)
        else:
            busRemainingList.append(station1)
            station1 = station2

    busRemainingList.append(station1)

    return len(busRemainingList)


def isOverlapping(station1, station2):
    return station1[1] >= station2[0]


def overlappedStation(station1, station2):
    new_station = []
    new_station.append(min(station1[0], station2[0]))
    new_station.append(max(station1[1], station2[1]))
    return new_station


# Q2

import heapq


def solvePuzzle(numbers):
    mark_numbers = []
    for num in numbers:
        mark_numbers.append(0)

    answerToPuzzle = 0

    while True:
        heap = []
        for x in range(len(numbers)):
            if mark_numbers[x] == 0:
                heapq.heappush(heap, (numbers[x], x))

        if len(heap) == 0:
            break

        (smallest_num, smallest_index) = heapq.heappop(heap)
        answerToPuzzle += smallest_num

        mark_numbers[smallest_index] = 1
        if smallest_index - 1 >= 0:
            mark_numbers[smallest_index - 1] = 1
        if smallest_index + 1 < len(numbers):
            mark_numbers[smallest_index + 1] = 1

    return answerToPuzzle


# Q3


import heapq


def findMedianPrice(prices, k):
    start, end = 0, k - 1
    n = len(prices)
    median_list = []

    while end < n:
        heap = []
        x = start
        while x <= end:
            heapq.heappush(heap, prices[x])
            x += 1

        median_list.append(calculate_median(heap, k))
        start += 1
        end += 1

    return median_list


def calculate_median(heap, k):
    if k % 2 != 0:
        iter = int((k - 1) / 2)
        for x in range(iter):
            heapq.heappop(heap)
        return heapq.heappop(heap)
    else:
        iter = int(k / 2 - 1)
        for x in range(iter):
            heapq.heappop(heap)
        num1 = heapq.heappop(heap)
        num2 = heapq.heappop(heap)
        return (num1 + num2) / 2


# Q4


def shorterBuildingsHelper(heights, count_list):
    n = len(heights)
    if n == 1:
        return

    mid = n // 2

    left_heights = heights[0:mid]
    right_heights = heights[mid:n]

    shorterBuildingsHelper(left_heights, count_list)
    shorterBuildingsHelper(right_heights, count_list)

    i, j = 0, 0
    for i in range(len(left_heights)):
        for j in range(len(right_heights)):
            left_building_height, left_index = left_heights[i]
            right_building_height, right_index = right_heights[j]
            if left_building_height > right_building_height:
                count_list[left_index] += 1

    return


def shorterBuildings(heights):
    n = len(heights)
    count_list = []
    new_heights = []
    for x in range(n):
        count_list.append(0)
        new_heights.append([heights[x], x])

    shorterBuildingsHelper(new_heights, count_list)
    return count_list


# Q5

import sys


def determineStandardRadius(houses, heaters):
    houses.sort()
    heaters.sort()

    nearest_heater_distance_list = []

    if len(heaters) == 1:
        for house in houses:
            nearest_heater_distance_list.append(abs(house - heaters[0]))

    if len(heaters) > 1:
        heater1 = heaters[0]
        heater2 = heaters[1]
        heater_iterator = 1

        for house in houses:
            if abs(house - heater1) <= abs(house - heater2):
                nearest_heater_distance_list.append(abs(house - heater1))
            else:
                while heater_iterator < len(heaters):
                    heater1 = heater2
                    heater_iterator += 1
                    if heater_iterator >= len(heaters):
                        nearest_heater_distance_list.append(abs(house - heater2))
                        break
                    else:
                        heater1 = heater2
                        heater2 = heaters[heater_iterator]
                        if abs(house - heater1) <= abs(house - heater2):
                            nearest_heater_distance_list.append(abs(house - heater1))
                            break

    print(nearest_heater_distance_list)

    max_dist = -1 * sys.maxsize
    for dist in nearest_heater_distance_list:
        max_dist = max(max_dist, dist)

    return max_dist


# Q6


def build_heap(list):
    n = len(list) // 2
    heapified_list = list
    for x in range(n - 1, -1, -1):
        heapified_list = heapify(heapified_list, x)
    return heapified_list


def heapify(list, x):
    left = 2 * x + 1
    right = 2 * x + 2

    if left < len(list) and list[left] > list[x]:
        largest_index = left
    else:
        largest_index = x

    if right < len(list) and list[right] > list[x]:
        largest_index = right

    if largest_index != x:
        list[largest_index], list[x] = list[x], list[largest_index]
        heapified_list = heapify(list, largest_index)
        return heapified_list
    return list


def heap_pop(list):
    new_list = list[1 : len(list)]
    build_heap(new_list)
    return list[0]


def isRearrangePossible(s, k):
    n = len(s)
    char_dict = {}
    for c in s:
        if c in char_dict:
            char_dict[c] += 1
        else:
            char_dict[c] = 1

    frequency_list = []
    for f in char_dict.values():
        frequency_list.append(f)

    heapified_list = build_heap(frequency_list)

    ele = heap_pop(heapified_list)

    return n - ele >= (ele - 1) * (k - 1)


# Q7

import heapq


class Huffman:
    def __init__(self):
        self.huffman_codes = {}
        self.source_string = ""

    def set_source_string(self, src_str):
        self.source_string = src_str

    def generate_codes(self):
        huffman_codes = {}
        source_string_char_freq_dict = {}

        # Maintaining a dictionary to store frequencies
        for c in self.source_string:
            if c in source_string_char_freq_dict:
                source_string_char_freq_dict[c] += 1
            else:
                source_string_char_freq_dict[c] = 1

        # Pushing the frequecies to the minheap
        heap = []
        for char, freq in source_string_char_freq_dict.items():
            heapq.heappush(heap, (freq, char))

        # Algorithm to create the huffman codes
        while len(heap) > 1:
            left_freq, left_string = heapq.heappop(heap)
            right_freq, right_string = heapq.heappop(heap)

            if left_freq == right_freq:
                heapq.heappush(heap, (right_freq, right_string))

                list_same_freq_item = []
                list_same_freq_item.append((left_freq, left_string))
                freq = left_freq

                while heap:
                    f, s = heapq.heappop(heap)
                    if f != freq:
                        heapq.heappush(heap, (f, s))
                        break
                    list_same_freq_item.append((f, s))

                sorted_list = sorted(list_same_freq_item, key=lambda x: x[0])

                if len(sorted_list) > 2:
                    x = 2
                    while x < len(sorted_list):
                        heapq.heappush(heap, sorted_list[x])
                        x += 1

                left_freq, left_string = sorted_list[0]
                right_freq, right_string = sorted_list[1]

            # now the left_string and right_string are available

            for c in left_string:
                if c in huffman_codes:
                    huffman_codes[c] = "0" + huffman_codes[c]
                else:
                    huffman_codes[c] = "0"

            for c in right_string:
                if c in huffman_codes:
                    huffman_codes[c] = "1" + huffman_codes[c]
                else:
                    huffman_codes[c] = "1"

            heapq.heappush(heap, (left_freq + right_freq, left_string + right_string))

        self.huffman_codes = huffman_codes

    def encode_message(self, message_to_encode):
        encoded_msg = ""
        for c in message_to_encode:
            encoded_msg += self.huffman_codes[c]

        return encoded_msg

    def decode_message(self, encoded_msg):
        decoded_msg = ""
        start, end = 0, 1
        while end < len(encoded_msg):
            if encoded_msg[start:end] in self.huffman_codes.values():
                for char, code in self.huffman_codes.items():
                    if code == encoded_msg[start:end]:
                        decoded_msg += char
                start = end
                end = start + 1
            else:
                end += 1
        if encoded_msg[start:end] in self.huffman_codes.values():
            for char, code in self.huffman_codes.items():
                if code == encoded_msg[start:end]:
                    decoded_msg += char
        return decoded_msg


# Q8

from collections import deque


class Node:
    def __init__(self, list, left=None, right=None):
        self.data = ""  # string
        self.list = list
        self.left = left
        self.right = right
        self.pivot_element = -1


class Wavelet_Tree:
    def __init__(self, A: list[int] = []):
        self.root = Node(A)

    def calculate_binary_string(self, list):
        str = ""
        max, min = -1, 10

        left_list = []
        right_list = []

        if len(list) <= 1:
            str += "X"
            return -1, left_list, right_list, str

        # special case - when all the elements of the list are same
        list_set = set()
        for item in list:
            list_set.add(item)

        if len(list_set) == 1 and len(list) != 1:
            for x in range(len(list)):
                str += "X"
            return -1, left_list, right_list, str

        for item in list:
            if item > max:
                max = item
            if item < min:
                min = item

        pivot_element = (max + min) / 2
        for item in list:
            if item <= pivot_element:
                str += "0"
                left_list.append(item)
            else:
                str += "1"
                right_list.append(item)

        return pivot_element, left_list, right_list, str

    def generate_tree(self):
        n = self.root
        queue = deque()
        queue.append(n)

        while queue:
            n = queue.popleft()
            pivot_element, left_list, right_list, data = self.calculate_binary_string(
                n.list
            )
            n.data = data
            n.pivot_element = pivot_element
            if len(left_list):
                n.left = Node(left_list)
                queue.append(n.left)
            if len(right_list):
                n.right = Node(right_list)
                queue.append(n.right)

    def get_wavelet_level_order(self):
        self.generate_tree()

        main_list = []
        queue = deque()
        queue.append(self.root)

        while queue:
            sublist = []
            for x in range(len(queue)):
                node = queue.popleft()
                sublist.append(node.data)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            main_list.append(sublist)

        return main_list

    def rank(self, character, position):
        self.generate_tree()

        n = self.root
        return self.rank_helper(character, position, n)

    def rank_helper(self, character, position, n):
        if n.pivot_element == -1:
            return min(len(n.list), position)

        if character <= n.pivot_element:
            new_position = 0
            for x in range(position):
                if n.data[x] == "0":
                    new_position += 1
            return self.rank_helper(character, new_position, n.left)

        else:
            new_position = 0
            for x in range(position):
                if n.data[x] == "1":
                    new_position += 1
            return self.rank_helper(character, new_position, n.right)
