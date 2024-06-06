# --------------------------------------------------Assignment_01-------------------------------------------------#

# -------------------------------------------------------Q3-------------------------------------------------------#


from curses.ascii import isalpha

"""
Q3.
SOLUTION APPROACH :
    1. The worst case (with minimum balls) in having p balls of same color would be :
        1.1. To have p-1 balls of four categories and p balls of fifth category.
        1.2. However if for ex. balls in c1 < p, then we consider balls in c1 to be c1, otherwise p-1.
        1.3. Then we add the total balls in the list + 1, which is the result.
    
    For example : [c1,c2,c3,c4,c5] = [3,4,3,9,23] and p = 4
    So we would potentially like p-1 in each category and the result should be + 1.
    [3,3,3,3,3] + 1 = 16 balls 

    However if for ex. : [c1,c2,c3,c4,c5] = [3,4,3,9,23] and p = 4
    [3,4,3,4,4] + 1 = 19 balls

    Time Complexity : O(1) (O(5) - number of categories is same)
    Space Complexity : O(1) (O(5) - number of categories is same)
"""


def minimum_balls(c1, c2, c3, c4, c5, p):
    categories_list = [c1, c2, c3, c4, c5]
    total_balls_taken_out = greater_than_p(categories_list, p)
    return total_balls_taken_out


""" 
Helper function to find the categories whose number of balls is > p and calculating total balls required. 
"""


def greater_than_p(categories_list, p):
    total_balls = 0
    for x in range(0, len(categories_list)):
        # if num_of_balls in a particular category is > p update with p-1 otherwise keep the default value
        if categories_list[x] >= p:
            categories_list[x] = p - 1

    for num in categories_list:
        total_balls += num

    return total_balls + 1


""" 
Q3. 
TESTING :
c1 = 3
c2 = 4
c3 = 3
c4 = 9
c5 = 23
p = 4

print(minimum_balls(c1, c2, c3, c4, c5, p)) 
"""

# -------------------------------------------------------Q4-------------------------------------------------------#

"""
Q4.
SOLUTION APPROACH :
    1. Creating a encoded list such that blue is repesented as 0 and pink as 1.

        For ex. tiles = ["pink","pink","blue","blue","pink","blue","blue","pink"] 
        then tiles_encoded = [1,1,0,0,1,0,0,1].

    2. This would mean that we need to find the longest subarray whose cumulative sum is k or less.
    3. Using two pointers start and end, we will traverse the list.
    4. We keep incrementing end as long as the sum is less than k. We keep a max of the longest subarray.
    5. If the sum exceeds k we increment the start pointer and decrement the sum by tiles_encoded[start_pointer].
    6. We return the maximum subarray.

    Time Complexity : O(N)
    Space Complexity : O(N)
"""


def longestBlues(tiles, k):
    # Creating a new list to work with numbers
    tiles_encoded = []
    for color in tiles:
        if color == "blue":
            tiles_encoded.append(0)
        else:
            tiles_encoded.append(1)

    start_pointer = 0
    end_pointer = 1

    sum = tiles_encoded[start_pointer] + tiles_encoded[end_pointer]
    number_of_tiles = end_pointer - start_pointer + 1

    # Traversing the encoded list
    while end_pointer < len(tiles_encoded):
        # If sum <= k we keep incrementing the end pointer and update the sum & maximum subarray size.
        if sum <= k:
            number_of_tiles = max(number_of_tiles, end_pointer - start_pointer + 1)
            end_pointer += 1
            if end_pointer >= len(tiles_encoded):
                break
            sum += tiles_encoded[end_pointer]

        # If the sum > k we increment the start pointer and update the sum.
        else:
            sum -= tiles_encoded[start_pointer]
            start_pointer += 1

    # return the number of largest continous sequence of blue tiles
    return number_of_tiles


"""
Q4.
TESTING: 
tiles = ["blue","blue","blue","pink","pink","pink","blue","blue","blue","blue","pink"]
k = 2

tiles = ["pink", "pink", "blue", "blue", "pink", "blue", "blue", "pink"]
k = 1

print(longestBlues(tiles, k)) 
"""

# -------------------------------------------------------Q5-------------------------------------------------------#

"""
Q5.
SOLUTION APPROACH :
    1. Iterate through the string and increase total_kid count when encountering a alphabet
    2. Then iterate to store the numbers for each category. And adding to the total_candy count
    3. Maintaining a list to store number of candies of each category.
    4. Formatting the result by traversing though the candy_category list.

    Time Complexity : O(N)
    Space Complexity : O(1)
"""


def CandiesLog(s):
    total_candies = 0
    total_kids = 0

    # Creating a list of size 26 (total lowercase alphabets) to store the number of candies in each category
    # For ex. [1,2,3 ......] would mean 1 candy of type a, 2 candies of b, 3 candies of c and so on.
    candy_types_list = [0] * 26

    total_kids, total_candies = candies_log_helper(
        s, total_candies, total_kids, candy_types_list
    )

    # Iterating over the candy_type_list to format total candies in each category
    types_num = ""
    for num_candies_per_category in range(0, len(candy_types_list)):
        if candy_types_list[num_candies_per_category] != 0:
            types_num += chr(num_candies_per_category + ord("a"))
            types_num += str(candy_types_list[num_candies_per_category])

    # Formatting the result as desired
    res = "K" + str(total_kids) + "T" + str(total_candies) + types_num
    return res


"""
Helper function to find total kids, total candies and number of candies in each category 
"""


def candies_log_helper(s, total_candies, total_kids, candy_types_list):
    iterator = 0

    while iterator < len(s):
        # Incrementing the total_kids count when encountering a alphabet
        if isalpha(s[iterator]):
            total_kids += 1
            num_str = ""

            # Storing the candy type
            category = s[iterator]
            iterator += 1

            # Iterating further to find number of candies for the category
            while iterator < len(s) and not isalpha(s[iterator]):
                num_str += s[iterator]
                iterator += 1
            iterator -= 1
            num_candies = int(num_str)

            # Updating the total_candies count and updating the candy_types_list
            total_candies += num_candies
            candy_types_list[ord(category) - ord("a")] += num_candies
        iterator += 1

    return total_kids, total_candies


# s = 'a1a5b2c3'
# s = "c18d4d13b6a14c5"
# print(CandiesLog(s))

# -------------------------------------------------------Q6-------------------------------------------------------#

"""
Q6.
SOLUTION APPROACH :
    1. Recursion approach:
        1.1. Reversing the first k elements using the pointers : prev_node, curr_node and next_node
        1.2. Calling the reverse list method again by passing the list starting from the (k+1)th element
        1.3. Base case being if length of the list is less than k, we simply return the head as it is.

    Time Complexity : O(N)
    Space Complexity : O(1)
"""


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class HeimdallQuest:
    def reverse_k_steps(self, head, k):
        # Write your code here
        node_count = 0
        curr = head
        while curr:
            node_count += 1
            curr = curr.next
        if node_count < k:
            return head
        steps = k
        prev_node = None
        curr_node = head
        next_node = head.next
        while steps:
            curr_node.next = prev_node
            prev_node = curr_node
            curr_node = next_node
            next_node = curr_node.next
            steps -= 1
        hq = HeimdallQuest()
        head.next = hq.reverse_k_steps(curr_node, k)

        return prev_node

    def get_linked_list(self, head):
        # Write your code here to return list
        node = head
        res = []
        while node:
            res.append(node.value)
            node = node.next

        return res

    def create_linked_list(self, lst):
        # Write your code here to create linked list and return head of Linked list
        head = Node(lst[0])
        node = head
        for x in range(1, len(lst)):
            node.next = Node(lst[x])
            node = node.next

        return head


hq = HeimdallQuest()
head = hq.create_linked_list([12, 23, 221, 12, 12])
new_head = hq.reverse_k_steps(head, 0)
print(hq.get_linked_list(new_head))
