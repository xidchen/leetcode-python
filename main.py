import leetcode


# 1: /problems/two-sum/
def two_sum():
    nums = [2, 7, 11, 15]
    target = 9
    res = lc.two_sum(nums, target)
    print(f"Two sums: {res}")


# 2: /problems/add-two-numbers/
def add_two_numbers():
    l1 = lc.list_to_linked_list([9, 9, 9, 9, 9, 9, 9])
    l2 = lc.list_to_linked_list([9, 9, 9, 9])
    res_link_list = lc.add_two_numbers(l1, l2)
    res = lc.linked_list_to_list(res_link_list)
    print(f"Adding two linked list numbers: {res}")


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


# 16: /problems/3sum-closest/
def three_sum_closest():
    nums = [4, 0, 5, -5, 3, 3, 0, -4, -5]
    target = -2
    res = lc.three_sum_closest(nums, target)
    print(f"Three sum closest: {res}")


# 17: /problems/letter-combinations-of-a-phone-number/
def letter_combinations():
    digits = "38"
    res = lc.letter_combinations(digits)
    print(f"Letter combinations: {res}")


# 18: /problems/4sum/
def four_sum():
    nums = [0, 0, 0, 1000000000, 1000000000, 1000000000, 1000000000]
    target = 1000000000
    res = lc.four_sum(nums, target)
    print(f"Four sum: {res}")


# 19: /problems/remove-nth-node-from-end-of-list/
def remove_nth_from_end():
    head = lc.list_to_linked_list([1, 2, 3, 4, 5])
    n = 2
    res_linked_list = lc.remove_nth_from_end(head, n)
    res = lc.linked_list_to_list(res_linked_list)
    print(f"Remove nth node from end of list: {res}")


# 20: /problems/valid-parentheses/
def is_valid():
    s = "()[]{}"
    res = lc.is_valid(s)
    print(f"Valid parentheses: {res}")


# 21: /problems/merge-two-sorted-lists/
def merge_two_lists():
    list1 = lc.list_to_linked_list([1, 2, 4])
    list2 = lc.list_to_linked_list([1, 3, 4])
    res_linked_list = lc.merge_two_lists(list1, list2)
    res = lc.linked_list_to_list(res_linked_list)
    print(f"Merge two sorted lists: {res}")


# 22: /problems/generate-parentheses/
def generate_parenthesis():
    n = 3
    res = lc.generate_parenthesis(n)
    print(f"Generated parenthesis: {res}")


# 23: /problems/merge-k-sorted-lists/
def merge_k_lists():
    lists = [[1, 4, 5], [1, 3, 4], [2, 6]]
    linked_lists = [lc.list_to_linked_list(_) for _ in lists]
    res_linked_list = lc.merge_k_lists(linked_lists)
    res = lc.linked_list_to_list(res_linked_list)
    print(f"Merge k sorted lists: {res}")


# 24: /problems/swap-nodes-in-pairs/
def swap_pairs():
    head = lc.list_to_linked_list([1, 2, 3, 4])
    res_linked_list = lc.swap_pairs(head)
    res = lc.linked_list_to_list(res_linked_list)
    print(f"Swap nodes in pairs: {res}")


# 26: /problems/remove-duplicates-from-sorted-array/
def remove_duplicates():
    nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    res = lc.remove_duplicates(nums)
    print(f"Remove duplicates from sorted array: {res}")


# 27: /problems/remove-element/
def remove_element():
    nums = [0, 1, 2, 2, 3, 0, 4, 2]
    val = 2
    res = lc.remove_elements(nums, val)
    print(f"Remove element: {res}")


# 28: /problems/find-the-index-of-the-first-occurrence-in-a-string/
def str_str():
    haystack = "sadness"
    needle = "sad"
    res = lc.str_str(haystack, needle)
    print(f"Find the index of the first occurrence in a string: {res}")

# 29: /problems/divide-two-integers/
def divide():
    dividend = 7
    divisor = -3
    res = lc.divide(dividend, divisor)
    print(f"Divide two integers: {res}")

# 30: /problems/substring-with-concatenation-of-all-words/
def find_substring():
    s = "barfoothefoobarman"
    words = ["foo", "bar"]
    res = lc.find_substring(s, words)
    print(f"Substring with concatenation of all words: {res}")

# 31: /problems/next-permutation/
def next_permutation():
    nums = [4, 5, 3, 2, 1]
    lc.next_permutation(nums)
    print(f"Next permutation: {nums}")


if __name__ == "__main__":
    lc = leetcode.Leetcode()
    two_sum()
    add_two_numbers()
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
    three_sum_closest()
    letter_combinations()
    four_sum()
    remove_nth_from_end()
    is_valid()
    merge_two_lists()
    generate_parenthesis()
    merge_k_lists()
    swap_pairs()
    remove_duplicates()
    remove_element()
    str_str()
    divide()
    find_substring()
    next_permutation()
