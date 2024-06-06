def amountPoliceGets(people):
    amount = 0
    stack = []

    for person in people:
        if len(stack) == 0:
            stack.append(person)
        else:
            person_1 = stack.pop()
            person_2 = person
            stack, amount = evaluate_interaction(person_1, person_2, stack, amount)

    return amount


def evaluate_interaction(first, second, stack, amount):
    if will_interact(first[0], second[0]) == False:
        stack.append(first)
        stack.append(second)
        return stack, amount

    while will_interact(first[0], second[0]):
        first_move = first[0]
        first_money = first[1]

        second_move = second[0]
        second_money = second[1]

        if first_move == 0 and second_move == -1:
            amount += second_money
            if len(stack):
                second = first
                first = stack.pop()
                if will_interact(first[0], second[0]) == False:
                    stack.append(first)
                    stack.append(second)
            else:
                stack.append(first)
                break
        elif first_move == 1 and second_move == -1:
            amount += first_money + second_money
            if len(stack):
                first = stack.pop()
                second = [0, 0]
                if will_interact(first[0], second[0]) == False:
                    stack.append(first)
                    stack.append(second)
            else:
                stack.append([0, 0])
                break
        else:
            amount += first_money
            if len(stack):
                first = stack.pop()
                if will_interact(first[0], second[0]) == False:
                    stack.append(first)
                    stack.append(second)
            else:
                stack.append(second)
                break

    return stack, amount


def will_interact(first_move, second_move):
    return first_move > second_move


people = [
    [0, 1],
    [0, 10],
    [1, 1],
    [0, 7],
    [0, 12],
    [1, 1],
    [-1, 1],
    [-1, 1],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 1],
    [1, 1],
    [0, 1],
    [1, 1],
    [0, 1],
    [0, 1],
    [1, 1],
    [-1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [-1, 1],
    [-1, 1],
    [1, 1],
    [1, 1],
    [-1, 1],
    [0, 1],
    [0, 1],
    [1, 1],
    [1, 1],
]
print(amountPoliceGets(people))

import random
import sys


class Node:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.prev = None
        self.down = None
        self.above = None


MIN_VALUE = -sys.maxsize - 1
MAX_VALUE = sys.maxsize
head = Node(MIN_VALUE)
tail = Node(MAX_VALUE)
height_of_skiplist = 0


class SkipList:
    def __init__(self):
        # Any variables for intialization
        head.next = tail
        tail.prev = head
        return None

    def search(self, target: int) -> bool:
        # Returns True if the element is present in skip list else False
        node_found = search_node(target)
        if node_found.val == target:
            return True
        return False

    def insert(self, num: int) -> None:
        # Inserts the element into the skip list
        if self.search(num) == True:
            print("The number is already present, so no insert")
            return None

        n = search_node(num)
        print("The maximum node with value just below num is :")
        print(n.val)
        level = -1

        while True:
            level += 1

            can_increase_level(level, height_of_skiplist, head, tail)

            new_node = Node(num)
            new_next = n.next

            n.next = new_node
            new_node.prev = n
            new_node.next = new_next
            new_next.prev = new_node

            while n.prev is not None and n.above is None:
                n = n.prev
            n = n.above
            if random_boolean() == False:
                break


# This method is to return the node at level 0 if found, else find the node just before it.
def search_node(target):
    n = head
    while n.down:
        n = n.down
        while n.next.val <= target:
            n = n.next

    return n


def random_boolean():
    val = random.choice([True, False])
    print(val)


def can_increase_level(level, height_of_skiplist, head, tail):
    if level >= height_of_skiplist:
        height_of_skiplist += 1
        add_empty_level(head, tail)


def add_empty_level(head, tail):
    new_head_node = Node(MIN_VALUE)
    new_tail_node = Node(MAX_VALUE)

    new_head_node.next = new_tail_node
    new_tail_node.prev = new_head_node

    head.above = new_head_node
    tail.above = new_tail_node

    new_head_node.down = head
    new_tail_node.down = tail

    head = new_head_node
    tail = new_tail_node


# sl = SkipList()
# print(head.val, tail.val)
# print(head.next.val)
# print(tail.prev.val)
# sl.insert(1)  # None
# sl.insert(2)  # None
# sl.insert(3)  # None
# print(sl.search(4))  # False
# sl.insert(4)  # None
# print(sl.search(4))  # True
# print(sl.search(1))  # True


stack_push = []
stack_pop = []


class Queue:
    DEFAULT_SIZE = 10

    # Initialization step
    def __init__(self):
        pass

    # Implement the enque() function to insert values into the queue.
    # It should return True on successful insertion and False on unsuccessful insertion.
    def enque(self, value):
        if len(stack_push) + len(stack_pop) == 10:
            return False
        else:
            stack_push.append(value)
            return True

    # Implement the deque() function to retrieve values from the queue.
    # It should return the next value in the queue; if the queue is empty, return -1.
    def deque(self):
        if len(stack_pop) != 0:
            return stack_pop.pop()
        else:
            if len(stack_push) == 0:
                return -1
            while len(stack_push):
                stack_pop.append(stack_push.pop())
            return stack_pop.pop()


queue = Queue()
print(queue.deque())
print(queue.enque(1))
print(queue.enque(2))
print(queue.enque(3))
print(queue.deque())
print(queue.deque())
print(queue.deque())
print(queue.deque())


def isItPossible(initial: list[str], final: list[str]) -> bool:
    # Your code here
    initial_stack = []
    for x in range(len(initial) - 1, -1, -1):
        initial_stack.append(initial[x])

    required_stack = []
    for color in final:
        required_stack.append(color)

    intermediate_stack = []
    while len(initial_stack):
        initial_top = initial_stack.pop()
        required_top = required_stack.pop()
        intermediate_top = None
        if len(intermediate_stack):
            intermediate_top = intermediate_stack.pop()

        if initial_top != required_top:
            if intermediate_top and (intermediate_top == required_top):
                initial_stack.append(initial_top)
            elif intermediate_top:
                intermediate_stack.append(intermediate_top)
                intermediate_stack.append(initial_top)
                required_stack.append(required_top)
            else:
                intermediate_stack.append(initial_top)
                required_stack.append(required_top)
        else:
            if intermediate_top:
                intermediate_stack.append(intermediate_top)

    while True:
        if len(intermediate_stack) == 0:
            return True

        intermediate_top = intermediate_stack.pop()
        required_top = required_stack.pop()

        if intermediate_top != required_top:
            return False
        else:
            continue


initial = ["Red", "Blue", "Green", "Yellow", "Orange"]
final = ["Yellow", "Orange", "Green", "Blue", "Red"]
print(isItPossible(initial, final))
