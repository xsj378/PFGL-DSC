def max_array_sum(n, a):
    # 计算初始数组和和负数的数量
    total_sum = sum(a)
    min_abs_value = min(abs(x) for x in a)
    neg_count = sum(1 for x in a if x < 0)

    # 如果负数的数量是偶数，直接将所有元素的绝对值相加
    if neg_count % 2 == 0:
        return sum(abs(x) for x in a)
    else:
        # 如果负数的数量是奇数，选择一个最小的绝对值的元素不进行翻转
        return sum(abs(x) for x in a) - 2 * min_abs_value

# 输入处理
n = int(input())
a = list(map(int, input().split()))

# 调用函数计算最大数组和
result = max_array_sum(n, a)
print(result)