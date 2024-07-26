list = [2, 3, 4, 5, 6, 9, 10]
num = 3
count = 0
left = 0
right = len(list) - 1
while left <= right:
    mid = (left + right) // 2
    count += 1
    if list[mid] < num:
        left = mid + 1
    elif list[mid] > num:
        right = mid - 1
    else:
        print(f"要找的数字是{num}， 索引{mid},共查询{count}次")
        break
else:
    print("没有该元素")
