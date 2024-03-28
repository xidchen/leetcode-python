import leetcode


# 1: /problems/two-sum/
def two_sum():
    nums = [2, 7, 11, 15]
    target = 9
    res = lc.two_sum(nums, target)
    print(f"Two sums: {res}")


# 3: /problems/longest-substring-without-repeating-characters/
def length_of_longest_substring():
    s = "abcdefabc"
    res = lc.length_of_longest_substring(s)
    print(f"Length of longest substring: {res}")


# 4: /problems/median-of-two-sorted-arrays/
def find_median_sorted_arrays():
    a = [1, 2]
    b = [3, 4]
    res = lc.find_median_sorted_arrays(a, b)
    print(f"Median of sorted arrays: {res}")


# 5: /problems/longest-palindromic-substring/
def longest_palindrome():
    s = "babad"
    res = lc.longest_palindrome(s)
    print(f"Longest palindrome: {res}")


# 6: /problems/zigzag-conversion/
def convert():
    s = "PAYPALISHIRING"
    num_rows = 3
    res = lc.convert(s, num_rows)
    print(f"Zigzag conversion: {res}")


# 7: /problems/reverse-integer/
def reverse():
    x = -120
    res = lc.reverse(x)
    print(f"Reverse integer: {res}")


# 8: /problems/string-to-integer-atoi/
def my_atoi():
    s = "-273 degree"
    res = lc.my_atoi(s)
    print(f"String to integer (atoi): {res}")


# 9: /problems/palindrome-number/
def is_palindrome():
    x = 121
    res = lc.is_palindrome(x)
    print(f"Is palindrome: {res}")


# 10: /problems/regular-expression-matching/
def is_match():
    s = "aa"
    p = "a*"
    res = lc.is_match(s, p)
    print(f"Regular expression matching: {res}")


# 11: /problems/container-with-most-water/
def max_area():
    height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    res = lc.max_area(height)
    print(f"Container with most water: {res}")


# 12: /problems/integer-to-roman/
def int_to_roman():
    num = 2024
    res = lc.int_to_roman(num)
    print(f"Integer to roman: {res}")


# 13: /problems/roman-to-integer/
def roman_to_int():
    s = "MMXXIV"
    res = lc.roman_to_int(s)
    print(f"Roman to integer: {res}")


# 14: /problems/longest-common-prefix/
def longest_common_prefix():
    strs = ["flower", "flow", "flight"]
    res = lc.longest_common_prefix(strs)
    print(f"Longest common prefix: {res}")


# 15: /problems/3sum
def three_sum():
    nums = [-1, 0, 1, 2, -1, -4]
    res = lc.three_sum(nums)
    print(f"Three sum: {res}")


if __name__ == "__main__":
    lc = leetcode.Leetcode()
    two_sum()
    length_of_longest_substring()
    find_median_sorted_arrays()
    longest_palindrome()
    convert()
    reverse()
    my_atoi()
    is_palindrome()
    is_match()
    max_area()
    int_to_roman()
    roman_to_int()
    longest_common_prefix()
    three_sum()
