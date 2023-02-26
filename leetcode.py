from classes import TreeNode, ListNode


class Leetcode:

    def __init__(self):
        pass

    # 1: /problems/two-sum/
    @staticmethod
    def two_sum(nums: [int], target: int) -> [int]:
        dic = {}
        for i, n in enumerate(nums):
            if n in dic:
                return [dic[n], i]
            dic[target - n] = i

    # 2: /problems/add-two-numbers/
    @staticmethod
    def add_two_numbers(l1: ListNode, l2: ListNode) -> ListNode:
        carry = 0
        root = n = ListNode(0)
        while l1 or l2 or carry:
            v1 = v2 = 0
            if l1:
                v1, l1 = l1.val, l1.next
            if l2:
                v2, l2 = l2.val, l2.next
            carry, val = divmod(v1 + v2 + carry, 10)
            n.next = ListNode(val)
            n = n.next
        return root.next

    # 3: /problems/longest-substring-without-repeating-characters/
    @staticmethod
    def length_of_longest_substring(s: str) -> int:
        ss, ll, cl = '', 0, 0
        for x in s:
            if x in ss:
                ss = ss[ss.index(x) + 1:] + x
                cl = len(ss)
            else:
                ss += x
                cl += 1
                ll = cl if cl > ll else ll
        return ll

    # 4: /problems/median-of-two-sorted-arrays/
    @staticmethod
    def find_median_sorted_arrays(nums1: [int], nums2: [int]) -> float:
        nums = sorted(nums1 + nums2)
        if len(nums) % 2:
            return nums[len(nums) // 2]
        else:
            return (nums[len(nums) // 2 - 1] + nums[len(nums) // 2]) / 2

    # 5: /problems/longest-palindromic-substring/
    @staticmethod
    def longest_palindrome(s: str) -> str:
        if len(s) < 2 or s == s[::-1]:
            return s
        start, ml = -1, 0
        for i in range(len(s)):
            odd = s[i - ml - 1:i + 1]
            even = s[i - ml:i + 1]
            if i - ml - 1 >= 0 and odd == odd[::-1]:
                start = i - ml - 1
                ml += 2
                continue
            if i - ml >= 0 and even == even[::-1]:
                start = i - ml
                ml += 1
        return s[start:start + ml]

    # 6: /problems/zigzag-conversion/
    @staticmethod
    def convert(s: str, num_rows: int) -> str:
        if num_rows == 1 or len(s) < num_rows:
            return s
        zigzag = [''] * num_rows
        row, step = 0, 1
        for c in s:
            zigzag[row] += c
            if row == 0:
                step = 1
            if row == num_rows - 1:
                step = -1
            row += step
        return ''.join(zigzag)

    # 7: /problems/reverse-integer/
    @staticmethod
    def reverse(x: int) -> int:
        y = -1 * x if x < 0 else x
        y = int(str(y)[::-1])
        y = -1 * y if x < 0 else y
        return y if y in range(-2 ** 31, 2 ** 31) else 0

    # 8: /problems/string-to-integer-atoi/
    @staticmethod
    def my_atoi(s: str) -> int:
        s = s.strip()
        if not s:
            return 0
        neg_flag = 0
        if s[0] in ['+', '-']:
            if len(s) == 1:
                return 0
            if not (48 <= ord(s[1]) <= 57):
                return 0
            if s[0] == '-':
                neg_flag = 1
            s = s[1:]
        if not (48 <= ord(s[0]) <= 57):
            return 0
        for i in range(len(s)):
            if not (48 <= ord(s[i]) <= 57):
                s = s[:i]
                break
        if neg_flag:
            n = int('-' + s)
            return n if -2 ** 31 < n else -2 ** 31
        n = int(s)
        return n if n < 2 ** 31 - 1 else 2 ** 31 - 1

    # 9: /problems/palindrome-number/
    @staticmethod
    def is_palindrome(x: int) -> bool:
        return str(x) == str(x)[::-1]

    # 10: /problems/regular-expression-matching/
    @staticmethod
    def is_match(s: str, p: str) -> bool:
        memo = {}

        def dp(i, j):
            if (i, j) not in memo:
                if j == len(p):
                    ans = i == len(s)
                else:
                    first_match = i < len(s) and p[j] in {s[i], '.'}
                    if j + 1 < len(p) and p[j + 1] == '*':
                        ans = dp(i, j + 2) or first_match and dp(i + 1, j)
                    else:
                        ans = first_match and dp(i + 1, j + 1)
                memo[i, j] = ans
            return memo[i, j]

        return dp(0, 0)

    # 11: /problems/container-with-most-water/
    @staticmethod
    def max_area(height: [int]) -> int:
        max_area, i, j = 0, 0, len(height) - 1
        while i < j:
            max_area = max(max_area, min(height[i], height[j]) * (j - i))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return max_area

    # 12: /problems/integer-to-roman/
    @staticmethod
    def int_to_roman(num: int) -> str:
        m = ['', 'M', 'MM', 'MMM']
        c = ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM']
        x = ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC']
        i = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX']
        return m[num // 1000] + c[num // 100 % 10] + x[num // 10 % 10] + i[
            num % 10]

    # 13: /problems/roman-to-integer/
    @staticmethod
    def roman_to_int(s: str) -> int:
        d = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        output, prev = 0, 0
        for p in s[::-1]:
            if d[p] >= prev:
                prev = d[p]
                output += d[p]
            else:
                output -= d[p]
        return output

    # 14: /problems/longest-common-prefix/
    @staticmethod
    def longest_common_prefix(strs: [str]) -> str:
        if not strs:
            return ''
        if len(strs) == 1:
            return strs[0]
        strs.sort()
        result = ''
        for a, b in zip(strs[0], strs[-1]):
            if a == b:
                result += a
            else:
                break
        return result

    # 15: /problems/3sum
    @staticmethod
    def three_sum(nums: [int]) -> [[int]]:
        import bisect
        ref, res = {}, []
        for n in nums:
            ref[n] = ref[n] + 1 if n in ref else 1
        nums = sorted(ref)
        for i, x in enumerate(nums):
            if not x:
                if ref[x] > 2:
                    res.append([0, 0, 0])
            elif ref[x] > 1 and -2 * x in ref:
                res.append([x, x, -2 * x])
            if x < 0:
                left = bisect.bisect_left(nums, -x - nums[-1], i + 1)
                right = bisect.bisect_right(nums, -x // 2, left)
                for y in nums[left:right]:
                    z = -x - y
                    if z in ref and z != y:
                        res.append([x, y, z])
        return res

    # 16: /problems/3sum-closest/
    @staticmethod
    def three_sum_closest(nums: [int], target: int) -> int:
        n = len(nums)
        nums.sort()
        res = sum(nums[:3])
        for i in range(n - 2):
            j, k = i + 1, n - 1
            if nums[i] + nums[j] + nums[j + 1] >= target:
                k = j + 1
            if nums[i] + nums[k - 1] + nums[k] <= target:
                j = k - 1
            while j < k:
                s = nums[i] + nums[j] + nums[k]
                if abs(target - s) < abs(target - res):
                    res = s
                if s == target:
                    return res
                if s < target:
                    j += 1
                else:
                    k -= 1
        return res

    # 17: /problems/letter-combinations-of-a-phone-number/
    @staticmethod
    def letter_combination(digits: str) -> [str]:
        phone = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
                 '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        res = []

        def backtrack(combination, d):
            if d:
                for letter in phone[d[0]]:
                    backtrack(combination + letter, d[1:])
            else:
                res.append(combination)

        if digits:
            backtrack('', digits)
        return res

    # 18: /problems/4sum/
    @staticmethod
    def four_sum(nums: [int], target: int) -> [[int]]:

        def k_sum(n: [int], t: int, k: int) -> [[int]]:
            res = []
            if len(n) < k or t < n[0] * k or n[-1] * k < t:
                return res
            if k == 2:
                return two_sum(n, t)
            for i in range(len(n)):
                if i == 0 or n[i - 1] != n[i]:
                    for st in k_sum(n[i + 1:], t - n[i], k - 1):
                        res.append([n[i]] + st)
            return res

        def two_sum(n: [int], t: int) -> [[int]]:
            res = []
            lo, hi = 0, len(n) - 1
            while lo < hi:
                sm = n[lo] + n[hi]
                if sm < t or (0 < lo and n[lo] == n[lo - 1]):
                    lo += 1
                elif t < sm or (hi < len(n) - 1 and n[hi] == n[hi + 1]):
                    hi -= 1
                else:
                    res.append([n[lo], n[hi]])
                    lo += 1
                    hi -= 1
            return res

        nums.sort()
        return k_sum(nums, target, 4)

    # 20: /problems/valid-parentheses/
    @staticmethod
    def is_valid(s: str) -> bool:
        d = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for x in s:
            if x in d:
                stack.append(x)
            else:
                if not stack or x != d[stack[-1]]:
                    return False
                stack.pop()
        return stack == []

    # 22: /problems/generate-parentheses/
    @staticmethod
    def generate_parenthesis(n: int) -> [str]:
        ans = []

        def backtrack(s='', left=0, right=0):
            if len(s) == 2 * n:
                ans.append(s)
                return
            if left < n:
                backtrack(s + '(', left + 1, right)
            if right < left:
                backtrack(s + ')', left, right + 1)

        backtrack()
        return ans

    # 26: /problems/remove-duplicates-from-sorted-array/
    @staticmethod
    def remove_duplicates(nums: [int]) -> int:
        temp = set(nums)
        nums.clear()
        nums.extend(temp)
        nums.sort()
        return len(nums)

    # 27: /problems/remove-element/
    @staticmethod
    def remove_elements(nums: [int], val: int) -> int:
        while nums.count(val):
            nums.remove(val)
        return len(nums)

    # 28: /problems/implement-strstr/
    @staticmethod
    def str_str(haystack: str, needle: str) -> int:
        return haystack.index(needle) if needle in haystack else -1

    # 29: /problems/divide-two-integers/
    @staticmethod
    def divide(dividend: int, divisor: int) -> int:
        positive = (dividend < 0) is (divisor < 0)
        dividend, divisor = abs(dividend), abs(divisor)
        res = 0
        while dividend >= divisor:
            temp, i = divisor, 1
            while dividend >= temp:
                dividend -= temp
                res += i
                temp <<= 1
                i <<= 1
        res = res if positive else -res
        return min(max(-2 ** 31, res), 2 ** 31 - 1)

    # 30: /problems/substring-with-concatenation-of-all-words/
    @staticmethod
    def find_substring(s: str, words: [str]) -> [int]:
        res = []
        if not words or len(s) < len(words) * len(words[0]):
            return res
        wc, wl, sl, wd = len(words), len(words[0]), len(s), {}
        for w in words:
            wd[w] = wd.get(w, 0) + 1
        for i in range(wl):
            start, cnt, tmp_dict = i, 0, {}
            for j in range(i, sl - wl + 1, wl):
                word = s[j:j + wl]
                if wd.get(word):
                    cnt += 1
                    tmp_dict[word] = tmp_dict.get(word, 0) + 1
                    while tmp_dict[word] > wd[word]:
                        tmp_dict[s[start:start + wl]] -= 1
                        start += wl
                        cnt -= 1
                    if cnt == wc:
                        res.append(start)
                        tmp_dict[s[start:start + wl]] -= 1
                        start += wl
                        cnt -= 1
                else:
                    start, cnt, tmp_dict = j + wl, 0, {}
        return res

    # 31: problems/next-permutation/
    @staticmethod
    def next_permutation(nums: [int]) -> None:
        i = len(nums) - 2
        while i >= 0 and nums[i + 1] <= nums[i]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            (nums[i], nums[j]) = (nums[j], nums[i])
        nums[::] = nums[:i + 1] + nums[i + 1:][::-1]
        return None

    # 35: /problems/search-insert-position/
    @staticmethod
    def search_insert(nums: [str], target: int) -> int:
        for i in range(len(nums)):
            if target <= nums[i]:
                return i
        return len(nums)

    # 38: /problems/count-and-say/
    @staticmethod
    def count_and_say(n: int) -> str:
        if n == 1:
            return '1'
        s = '1'
        for i in range(n - 1):
            previous, count = s[0], 0
            new = ''
            for current in s:
                if previous != current:
                    new += str(count) + previous
                    previous, count = current, 1
                else:
                    count += 1
            new += str(count) + previous
            s = new
        return s

    # 53: /problems/maximum-subarray/
    @staticmethod
    def max_sub_array(nums: [int]) -> int:
        max_current, max_global = nums[0], nums[0]
        for i in range(1, len(nums)):
            max_current = max(nums[i], max_current + nums[i])
            if max_current > max_global:
                max_global = max_current
        return max_global

    # 58: /problems/length-of-last-word/
    @staticmethod
    def length_of_last_word(s: str) -> int:
        return len(s.strip().split(' ')[-1]) if ' ' in s else len(s)

    # 66: /problems/plus-one/
    @staticmethod
    def plus_one(digits: [int]) -> [int]:
        s = ''.join([str(d) for d in digits])
        return [int(x) for x in str(int(s) + 1)]

    # 67: /problems/add-binary/
    @staticmethod
    def add_binary(a: str, b: str) -> str:
        return str(bin(int(a, 2) + int(b, 2)))[2:]

    # 69: /problems/sqrtx/
    @staticmethod
    def my_sqrt(x: int) -> int:
        import math
        return int(math.sqrt(x))

    # 70: /problems/climbing-stairs/
    @staticmethod
    def climb_stairs(n: int) -> int:
        d = {1: 1, 2: 2}

        def sub(m):
            if m in d:
                return d[m]
            d[m] = sub(m - 1) + sub(m - 2)
            return d[m]

        return sub(n)

    # 80: /problems/remove-duplicates-from-sorted-array-ii/
    @staticmethod
    def remove_duplicates_ii(nums: [int]) -> int:
        for x in set(nums):
            if nums.count(x) > 2:
                while nums.count(x) > 2:
                    nums.remove(x)
        return len(nums)

    # 88: /problems/merge-sorted-array/
    @staticmethod
    def merge(nums1: [int], m: int, nums2: [int], n: int) -> None:
        nums1[m:] = nums2[:n]
        nums1.sort()

    # 100: /problems/same-tree/
    def is_same_tree(self, p: TreeNode, q: TreeNode) -> bool:
        if p and q:
            return (p.val == q.val and self.is_same_tree(p.left, q.left)
                    and self.is_same_tree(p.right, q.right))
        else:
            return p == q

    # 101: /problems/symmetric-tree/
    @staticmethod
    def is_symmetric(root: TreeNode) -> bool:
        def is_mirror(t1: TreeNode, t2: TreeNode) -> bool:
            if t1 and t2:
                return (t1.val == t2.val and is_mirror(t1.left, t2.left)
                        and is_mirror(t1.right, t2.right))
            else:
                return t1 is t2

        if root:
            return is_mirror(root.left, root.right)
        return True
