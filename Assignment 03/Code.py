"""                                                                       Assignment - 3                                                               """
""" 
Q1. Solution Approach :
    * Constructed a circular linked list and deleted every k node. The remaining node is the winner.
    TC : O(N^2/K)
    SC : O(N)
"""


class ListNode:
    def __init__(self, value):
        self.val = value
        self.next = None


def who_wins(n, k):
    if n == 1:
        return 1
    head = construct_circular_linked_list(n)
    curr_node = head
    while curr_node.val != curr_node.next.val:
        for x in range(k - 2):
            curr_node = curr_node.next
        if curr_node.next.val == head.val:
            head = head.next
        curr_node.next = curr_node.next.next
        curr_node = curr_node.next

    return head.val


def construct_circular_linked_list(n):
    head = ListNode(1)
    curr_node = head
    for x in range(2, n + 1):
        curr_node.next = ListNode(x)
        curr_node = curr_node.next
    curr_node.next = head

    return head


""" 
Q2. Solution Approach :
    * Used BFS approach to see if the coordinate can be reached. 
    * Stopping condition : If the coordinates exceed the school coordinates.
    TC : O(N*(V+E))
    SC : O(N*(V+E))
"""

from collections import deque


def count_successful_school_commutes(home_coords, school_coords, N):
    result = 0
    for x in range(N):
        x1, y1 = home_coords[x]
        x2, y2 = school_coords[x]

        if x1 == x2 and y1 == y2:
            result += 1

        queue = deque()
        queue.append([x1, y1])

        while True:
            if len(queue) == 0:
                break
            coord = queue.popleft()
            x, y = coord
            sum_coords = x + y
            if sum_coords <= x2:
                if sum_coords == x2 and y == y2:
                    result += 1
                    break
                queue.append([sum_coords, y])
            if sum_coords <= y2:
                if x == x2 and sum_coords == y2:
                    result += 1
                    break
                queue.append([x, sum_coords])

    return result


""" 
Q3. Solution Approach :
    * Used BFS approach to see all the possible sequence. 
    * Stopping condition : When the height of the tree is N as long as the numbers are positive unit digit integers.
    TC : O(9*(V+E)) -> O(V+E)
    SC : O(9*(V+E)) -> O(V+E)
"""

from collections import deque


def zenthar_puzzle(N, K):
    result = []
    if N == 1:
        for x in range(1, 10):
            result.append(x)
        return result

    for x in range(1, 10):
        queue = deque()
        level = 1
        queue.append([x, str(x), level])
        level_reached = False
        while queue and not level_reached:
            for x in range(len(queue)):
                curr_node_num, curr_node_str, level = queue.popleft()
                if is_valid_number(curr_node_num - K):
                    num = curr_node_num - K
                    queue.append([num, curr_node_str + str(num), level + 1])
                    if (level + 1) == N:
                        result.append(int(curr_node_str + str(num)))
                if is_valid_number(curr_node_num + K):
                    num = curr_node_num + K
                    queue.append([num, curr_node_str + str(num), level + 1])
                    if (level + 1) == N:
                        result.append(int(curr_node_str + str(num)))
                if level + 1 == N:
                    level_reached = True

    return result


def is_valid_number(num):
    if num >= 0 and num <= 9:
        return True
    return False


""" 
Q4. Solution Approach :
    * Used recursive approach where the first layer of parenthesis is evaluated and the inner parenthesis are evaluated in the consequtive recursive calls. 
    * Stopping condition : When there are no parenthesis, simply return the string.
"""

from curses.ascii import isalpha, isdigit


def decompress(s):
    decompressed_string = ""
    if not contains_digit(s):
        return s

    parenthesis_stack = []
    x = 0
    while x < len(s):
        num_seq = ""
        while isdigit(s[x]):
            num_seq += s[x]
            num = int(num_seq)
            x += 1
        temp = ""
        if s[x] == "(":
            parenthesis_stack.append("(")
            while True:
                x += 1
                if x >= len(s):
                    break
                if s[x] == "(":
                    parenthesis_stack.append("(")
                elif s[x] == ")":
                    parenthesis_stack.pop()
                    if len(parenthesis_stack) == 0:
                        break
                temp += s[x]
            temp_decompressed_string = decompress(temp)
            str = temp_decompressed_string
            for i in range(num - 1):
                str += temp_decompressed_string
            decompressed_string += str
        elif isalpha(s[x]):
            decompressed_string += s[x]
        x += 1

    return decompressed_string


def contains_digit(s):
    for x in range(len(s)):
        if isdigit(s[x]):
            return True
    return False


""" 
Q5. Solution Approach :
    * Doing a BFS Traversal and storing the max_value in path, whenever finding node value greater than the max_value incrementing the counter.
    * If the total count > k return True.
    TC : O(N) 
    SC : O(N)
"""

from collections import deque


class TreeNode:
    def __init__(self, value: int):
        self.value = value
        self.left = None
        self.right = None


def create_bt_count_oracles_extract(values, k):
    root = create_binary_tree(values)
    tree_queue = deque()
    tree_queue.append([root, root.value])
    count = 0

    while tree_queue:
        node, max_val = tree_queue.popleft()
        if node.value >= max_val:
            count += 1
            max_val = node.value
        if node.left:
            tree_queue.append([node.left, max_val])
        if node.right:
            tree_queue.append([node.right, max_val])

    return count >= k


def create_binary_tree(values):
    if len(values) == 0:
        return None

    root = TreeNode(values[0])
    curr_node = root
    queue = deque()
    queue.append(curr_node)
    x = 1

    while x < len(values):
        curr_node = queue.popleft()
        curr_node.left = TreeNode(values[x])
        queue.append(curr_node.left)
        x += 1
        if x >= len(values):
            break
        curr_node.right = TreeNode(values[x])
        queue.append(curr_node.right)
        x += 1

    return root


""" 
Q6. Solution Approach :
    * Doing a level order traversal and storing the level with the max product in the level.
    TC : O(N) 
    SC : O(N)
"""

from collections import deque


def solve_puzzle(root):
    level = 1
    level_with_max_prod = 1
    max_prod = -(2**31)
    queue = deque()
    queue.append([root, level])
    while queue:
        prod = 1
        for x in range(len(queue)):
            curr_node, level = queue.popleft()
            prod *= curr_node.val
            if curr_node.left:
                queue.append([curr_node.left, level + 1])
            if curr_node.right:
                queue.append([curr_node.right, level + 1])
        if prod > max_prod:
            max_prod = prod
            level_with_max_prod = level

    return level_with_max_prod


""" 
Q7. Solution Approach :
    * Using recursion approach / preorder traversal.
    * Return the element if it is a leaf node and the list returned from the children, put the node value infront of each element and merge the items.
    * Take the sum when the final list is returned.
    TC : O(N) 
    SC : O(N)
"""

from collections import deque


def TreeOfNumbers(root) -> int:
    curr_node = root
    list = []
    list = getListOfNumbers(curr_node)
    sum = 0
    for x in range(len(list)):
        sum += int(list[x])
    return sum


def getListOfNumbers(curr_node):
    list = []
    if curr_node.left is None and curr_node.right is None:
        list.append(str(curr_node.data))
        return list

    if curr_node.left:
        list1 = getListOfNumbers(curr_node.left)
        for x in range(len(list1)):
            list1[x] = str(curr_node.data) + list1[x]
            list.append(list1[x])

    if curr_node.right:
        list2 = getListOfNumbers(curr_node.right)
        for x in range(len(list2)):
            list2[x] = str(curr_node.data) + list2[x]
            list.append(list2[x])

    return list


""" 
Q8. Solution Approach :
    * Insert - if the list[0] is empty then insert it in a sorted manner there and if filled merge it.
    * Continue this as long as list[i] is exhausted or empty. And merge the sorted lists.
    * Search - go through every sub list and do a binary search
    TC : Insert - O(logN) , Search - O(logN^2)
    SC : O(N)
"""


from math import ceil


class amor_dict:
    def __init__(self, num_list=[]):
        self.list = [[]]
        for num in num_list:
            self.insert(num)

    def insert(self, num):
        list = []
        level = 0
        list.append(num)
        if len(self.list) == 0:
            self.list.append(list)
        elif len(self.list[0]) == 0:
            self.list[0] = list
        else:
            if num >= self.list[0][0]:
                temp = [self.list[0][0], num]
            else:
                temp = [num, self.list[0][0]]
            level += 1
            while level < len(self.list) and len(self.list[level]) != 0:
                temp = merge_two_lists(temp, self.list[level])
                level += 1
            if level < len(self.list):
                self.list[level] = temp
            else:
                self.list.append(temp)
            for i in range(level):
                self.list[i] = []

    def search(self, num):
        for i in range(len(self.list)):
            if binary_search(self.list[i], num):
                return i
        return -1

    def print(self):
        self.printed_list = []
        for i in self.list:
            self.printed_list.append(i)
        return self.printed_list


def binary_search(list, num):
    start = 0
    end = len(list) - 1

    while start <= end:
        mid = ceil((start + end) / 2)
        if list[mid] == num:
            return True
        elif num > list[mid]:
            start = mid + 1
        else:
            end = mid - 1
    return False


def merge_two_lists(list1, list2):
    list = []
    m = 0
    n = 0
    while m < len(list1) and n < len(list2):
        if list1[m] < list2[n]:
            list.append(list1[m])
            m += 1
        else:
            list.append(list2[n])
            n += 1

    if m != len(list1):
        for i in range(m, len(list1)):
            list.append(list1[i])
    if n != len(list2):
        for i in range(n, len(list2)):
            list.append(list2[i])

    return list
