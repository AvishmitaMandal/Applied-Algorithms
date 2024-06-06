# Q1


def MinimumExpenditure(AppleTreePrice):
    smallest = 100000
    second_smallest = 100000
    smallest_index = []
    second_smallest_index = []

    person = AppleTreePrice[0]

    for i in range(3):
        if person[i] == smallest:
            smallest_index.append(i)
            break

        if person[i] == second_smallest:
            second_smallest_index.append(i)
            break

        if person[i] < smallest:
            second_smallest = smallest
            second_smallest_index = smallest_index

            smallest = person[i]
            smallest_index = []
            smallest_index.append(i)

        elif person[i] < second_smallest:
            second_smallest = person[i]
            second_smallest_index = []
            second_smallest_index.append(i)

    for i in range(1, len(AppleTreePrice)):
        new_smallest = 100000
        new_second_smallest = 100000
        new_smallest_index = []
        new_second_smallest_index = []
        person = AppleTreePrice[i]

        for x in range(3):
            if x not in smallest_index or len(smallest_index) == 0:
                if smallest + person[x] == new_smallest:
                    if len(new_smallest_index) == 0:
                        new_smallest_index.append(x)
                    elif new_smallest_index[0] != x:
                        new_smallest_index = []
                    break

                if smallest + person[x] == new_second_smallest:
                    if len(new_second_smallest_index) == 0:
                        new_second_smallest_index.append(x)
                    elif new_second_smallest_index[0] != x:
                        new_second_smallest_index = []
                    break

                if smallest + person[x] < new_smallest:
                    new_second_smallest = new_smallest
                    new_second_smallest_index = new_smallest_index
                    new_smallest = smallest + person[x]
                    new_smallest_index = []
                    new_smallest_index.append(x)

                elif smallest + person[x] < new_second_smallest:
                    new_second_smallest = smallest + person[x]
                    new_second_smallest_index = []
                    new_second_smallest_index.append(x)

        for x in range(3):
            if x not in second_smallest_index or len(second_smallest_index) == 0:
                if second_smallest + person[x] == new_smallest:
                    if len(new_smallest_index) == 0:
                        new_smallest_index.append(x)
                    elif new_smallest_index[0] != x:
                        new_smallest_index = []
                    break

                if second_smallest + person[x] == new_second_smallest:
                    if len(new_second_smallest_index) == 0:
                        new_second_smallest_index.append(x)
                    elif new_second_smallest_index[0] != x:
                        new_second_smallest_index = []
                    break

                if second_smallest + person[x] < new_smallest:
                    new_second_smallest = new_smallest
                    new_second_smallest_index = new_smallest_index
                    new_smallest = second_smallest + person[x]
                    new_smallest_index = []
                    new_smallest_index.append(x)

                elif second_smallest + person[x] < new_second_smallest:
                    new_second_smallest = second_smallest + person[x]
                    new_second_smallest_index = []
                    new_second_smallest_index.append(x)

        if new_smallest > new_second_smallest:
            temp = new_smallest
            new_smallest = new_second_smallest
            new_second_smallest = temp

            temp = new_smallest_index
            new_smallest_index = new_second_smallest_index
            new_second_smallest_index = temp

        smallest = new_smallest
        second_smallest = new_second_smallest
        smallest_index = new_smallest_index
        second_smallest_index = new_second_smallest_index

    return smallest


# list = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
# list2 = [[2, 5, 6]]
# list3 = [[4, 12, 7], [2, 34, 13]]
# print(MinimumExpenditure(list3))

# Q2


def alignments(A, B):
    dp = [[1 for i in range(B + 1)] for j in range(A + 1)]

    for i in range(1, A + 1):
        for j in range(1, B + 1):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1] + dp[i - 1][j - 1]

    return dp[A][B]


# print(alignments(A=1, B=2))


# Q3


def smallestMissingNumber(streetNumbers):
    if streetNumbers[0] != 0:
        return 0
    return smallestMissingNumberRecursive(streetNumbers)


def smallestMissingNumberRecursive(streetNumbers):
    n = len(streetNumbers)
    if n == 1:
        return streetNumbers[n - 1] + 1
    if n == 2:
        if streetNumbers[0] + 1 == streetNumbers[1]:
            return streetNumbers[1] + 1
        else:
            return streetNumbers[0] + 1

    mid = int(n / 2)
    left_list_smallest_element = smallestMissingNumberRecursive(
        streetNumbers[: mid + 1]
    )
    right_list_smallest_element = smallestMissingNumberRecursive(
        streetNumbers[mid + 1 :]
    )

    # checking for middle condition

    if left_list_smallest_element != streetNumbers[mid + 1]:
        return left_list_smallest_element
    else:
        return right_list_smallest_element


# streetNumbers = [
#     0,
#     1,
#     2,
#     3,
#     4,
#     5,
#     6,
#     7,
#     8,
#     9,
#     10,
#     11,
#     12,
#     13,
#     20,
#     21,
#     22,
#     23,
#     24,
#     25,
#     30,
# ]
# print(smallestMissingNumber(streetNumbers))

# Q4


def solution_inheritance(num_items, num_boxes, children):
    if num_boxes < children:
        return -1

    dp = [[0 for i in range(children)] for j in range(num_boxes)]

    # construct prefix sums
    p = []
    p.append(num_items[0])
    for i in range(1, num_boxes):
        p.append(p[i - 1] + num_items[i])

    if children == 0:
        return -1

    for j in range(children):
        dp[0][j] = 1

    for k in range(num_boxes):
        dp[k][0] = p[k]

    for i in range(1, num_boxes):
        for j in range(1, children):
            dp[i][j] = 10000000
            for x in range(i):
                cost = max(dp[x][j - 1], p[i] - p[x])
                if cost < dp[i][j]:
                    dp[i][j] = cost

    return dp[num_boxes - 1][children - 1]


num_items = [1356, 1420, 1750, 1650, 1715, 1496]
num_boxes = 6
children = 0
print(solution_inheritance(num_items, num_boxes, children))

# Q5


def place_max_speedbump(len_road, bump_int1, bump_int2, bump_int3):
    dp = []
    x = 0
    while x <= len_road:
        dp.append(-1)
        x += 1
    dp[0] = 0

    for i in range(1, len(dp), 1):
        index1, index2, index3 = -1, -1, -1
        if i - bump_int1 >= 0:
            index1 = i - bump_int1
        if i - bump_int2 >= 0:
            index2 = i - bump_int2
        if i - bump_int3 >= 0:
            index3 = i - bump_int3

        if max(dp[index1], dp[index2], dp[index3]) != -1:
            dp[i] = max(dp[index1], dp[index2], dp[index3]) + 1
        else:
            dp[i] = -1

    if dp[len_road] == -1:
        return 0
    return dp[len_road]


# len_road = 7
# bump_int1 = 5
# bump_int2 = 3
# bump_int3 = 2
# print(place_max_speedbump(len_road, bump_int1, bump_int2, bump_int3))


# Q6


def find_path(stone_inscription_list):
    n = len(stone_inscription_list)
    x = 0
    optimal_path_list = []
    while x < n:
        optimal_path_list.append(10000)
        x += 1

    optimal_path_list[0] = 0
    iterator = 0

    while iterator < n:
        j = iterator + 1
        x = 0
        while x < stone_inscription_list[iterator] and j < n:
            optimal_path_list[j] = min(
                optimal_path_list[iterator] + 1, optimal_path_list[j]
            )
            x += 1
            j += 1
        iterator += 1
    print(optimal_path_list)
    return optimal_path_list[n - 1]


# list = [1, 1, 1, 4, 1, 1, 3]
# print(find_path(list))


# Q7


def decode_cryptic_message(lists):
    if len(lists) == 1:
        return lists[0]

    x = 0
    new_list = []
    while x < len(lists):
        first_list = lists[x]
        x += 1
        if x >= len(lists):
            new_list.append(first_list)
            break
        second_list = lists[x]
        x += 1
        list = merge_two_list(first_list, second_list)
        new_list.append(list)

    return decode_cryptic_message(new_list)


def merge_two_list(first_list, second_list):
    list = []
    i, j = 0, 0

    while i < len(first_list) and j < len(second_list):
        if first_list[i] <= second_list[j]:
            list.append(first_list[i])
            i += 1
        else:
            list.append(second_list[j])
            j += 1

    if i == len(first_list):
        while j < len(second_list):
            list.append(second_list[j])
            j += 1

    if j == len(second_list):
        while i < len(first_list):
            list.append(first_list[i])
            i += 1

    return list


# lists = [[1, 4, 5], [1, 6, 7], [3, 3]]
# print(decode_cryptic_message(lists))
