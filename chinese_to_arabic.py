def chinese_to_arabic(chinese_number):
    chinese_digits = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    chinese_units = {'十': 10, '百': 100, '千': 1000}

    total = 0
    current_number = 0
    for char in chinese_number:
        if char in chinese_digits:
            current_number = chinese_digits[char]
        elif char in chinese_units:
            total += current_number * chinese_units[char]
            current_number = 0
    total += current_number
    return total

chinese_number = input("请输入大写中文数字：")
arabic_number = chinese_to_arabic(chinese_number)
print("阿拉伯数字为:", arabic_number)
