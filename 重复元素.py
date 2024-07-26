list = [5, 5, 9, 9, 13, 25]


def func(list):
    new_list = []
    for i in list:
        if i not in new_list:
            new_list.append(i)

    return new_list


print(func(list))
print(set(list))