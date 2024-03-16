from leetcode import Leetcode


if __name__ == '__main__':
    lc = Leetcode()
    print(f"Two sums: {lc.two_sum(nums=[2, 7, 11, 15], target=9)}")
    print(f"Length of longest substring: {lc.length_of_longest_substring('abcdefabc')}")
    print(f"Median of sorted arrays: {lc.find_median_sorted_arrays(a=[1, 2], b=[3, 4])}")
    print(f"Longest palindrome: {lc.longest_palindrome(s='babad')}")
    print(f"Zigzag conversion: {lc.convert(s='PAYPALISHIRING', num_rows=3)}")
    print(f"Reverse integer: {lc.reverse(x=-120)}")
    print(f"String to integer (atoi): {lc.my_atoi(s='-273 degree')}")
    print(f"Is palindrome: {lc.is_palindrome(x=121)}")
    print(f"Regular expression matching: {lc.is_match(s='aa', p='a*')}")
    print(f"Container with most water: {lc.max_area([1, 8, 6, 2, 5, 4, 8, 3, 7])}")
    print(f"Integer to roman: {lc.int_to_roman(2024)}")
