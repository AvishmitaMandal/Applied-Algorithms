# Q1


def translate_message(arr):
    arr1 = []

    for ele in arr:
        arr1.append((ele % 7, ele // 7))

    hash_table = {}

    for ele in arr1:
        if ele[0] not in hash_table:
            hash_table[ele[0]] = (1, [ele[1]])
        else:
            new_list = hash_table[ele[0]][1]
            new_list.append(ele[1])
            hash_table[ele[0]] = (hash_table[ele[0]][0] + 1, new_list)

    max_length = 1
    for key, value in hash_table.items():
        list = value[1]
        sorted_list = sorted(list)
        count = 1
        for x in range(1, len(sorted_list)):
            if sorted_list[x] - sorted_list[x - 1] == 1:
                count += 1
            else:
                count = 1
            max_length = max(max_length, count)

    return max_length


# print(translate_message(arr=[1, 8, 15, 23, 29, 36, 43]))


# Q2


def find_order_size(orders):
    hash_table = {}
    result = len(orders)

    for order in orders:
        order_tuple = (order[0], order[1])
        if order_tuple in hash_table:
            hash_table[order_tuple] += 1
        else:
            hash_table[order_tuple] = 1

    count = 0
    for key, value in hash_table.items():
        rev = (key[1], key[0])
        if rev in hash_table:
            if hash_table[key] >= hash_table[rev]:
                count += hash_table[key]
            else:
                count += hash_table[rev]
            hash_table[key] = 0
            hash_table[rev] = 0
        else:
            count += value

    return count


# print(find_order_size(orders=[[5, 8], [3, 2], [5, 8]]))

# Q3


def findCircusStrings(circusString):
    hash_table = {}
    start = 0
    end = start + 10

    n = len(circusString)
    if n <= 10:
        return []

    output = []
    while end <= n:
        substring = circusString[start:end]

        if hash(substring) in hash_table:
            hash_table[hash(substring)][0] += 1
        else:
            hash_table[hash(substring)] = [1, substring]

        start += 1
        end += 1

    for key, value in hash_table.items():
        if value[0] > 1:
            output.append(value[1])

    sorted_output = sorted(output)

    return sorted_output


print(findCircusStrings(circusString="YWYWYWYWYWYWY"))

# Q4


def checkIfAllDistinct(sum1, sum2, hash_table):
    list1 = hash_table[sum1]
    list2 = hash_table[sum2]

    for list1_item in list1:
        for list2_item in list2:
            index_set = set()
            index_set.add(list1_item[0])
            index_set.add(list1_item[1])
            index_set.add(list2_item[0])
            index_set.add(list2_item[1])

            if len(index_set) == 4:
                return True

    return False


def aliveOrDead(trees, tigersWish):
    trees_list = []

    for tree in trees:
        if tree != "X":
            trees_list.append(int(tree))

    hash_table = {}

    for x in range(len(trees_list) - 1):
        for y in range(x + 1, len(trees_list)):
            pair_sum = trees_list[x] + trees_list[y]
            if pair_sum in hash_table:
                new_list = hash_table[pair_sum]
                new_list.append((x, y))
                hash_table[pair_sum] = new_list
            else:
                hash_table[pair_sum] = [(x, y)]

    for key, value in hash_table.items():
        if (tigersWish - key) in hash_table:
            if checkIfAllDistinct(key, tigersWish - key, hash_table):
                return "Alive"

    return "Dead"


# print(
#     aliveOrDead(
#         trees=["X", "X", "X", "1"],
#         tigersWish=8,
#     )
# )
