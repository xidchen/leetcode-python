from leetcode import Leetcode


if __name__ == '__main__':
    lc = Leetcode()
    print(f"Two sums: {lc.two_sum(nums=[2, 7, 11, 15], target=9)}")
    print(f"Length of longest substring: {lc.length_of_longest_substring('abcdefabc')}")
    print(f"Median of sorted arrays: {lc.find_median_sorted_arrays(a=[1, 2], b=[3, 4])}")
    print(f"Longest palindrome: {lc.longest_palindrome(s='babad')}")
    print(f"Zigzag conversion: {lc.convert(s='PAYPALISHIRING', num_rows=3)}")
