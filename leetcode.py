from classes import ListNode, Node, TreeNode


class Leetcode:

    def __init__(self):
        pass

    @staticmethod
    def list_to_linked_list(nums: list[int]) -> ListNode:
        current = dummy = ListNode(0)
        for num in nums:
            current.next = ListNode(num)
            current = current.next
        return dummy.next

    @staticmethod
    def linked_list_to_list(node: ListNode) -> list[int]:
        nums = []
        while node:
            nums.append(node.val)
            node = node.next
        return nums

    # 1: /problems/two-sum/
    @staticmethod
    def two_sum(nums: list[int], target: int) -> list[int] | None:
        dic = {}
        for i, n in enumerate(nums):
            if n in dic:
                return [dic[n], i]
            dic[target - n] = i
            return None
        return None

    # 2: /problems/add-two-numbers/
    @staticmethod
    def add_two_numbers(l1: ListNode, l2: ListNode) -> ListNode:
        current = dummy = ListNode(None)
        carry = 0
        while l1 or l2 or carry:
            if l1:
                carry += l1.val
                l1 = l1.next
            if l2:
                carry += l2.val
                l2 = l2.next
            current.next = ListNode(carry % 10)
            current = current.next
            carry //= 10
        return dummy.next

    # 3: /problems/longest-substring-without-repeating-characters/
    @staticmethod
    def length_of_longest_substring(s: str) -> int:
        last_seen, start, longest = {}, 0, 0
        for i, c in enumerate(s):
            if c in last_seen and last_seen[c] >= start:
                start = last_seen[c] + 1
            else:
                longest = max(longest, i - start + 1)
            last_seen[c] = i
        return longest

    # 4: /problems/median-of-two-sorted-arrays/
    @staticmethod
    def find_median_sorted_arrays(a: list[int], b: list[int]) -> float:

        def get_kth_smallest(a_start: int, b_start: int, k: int):
            if k <= 0 or k > len(a) - a_start + len(b) - b_start:
                raise ValueError('k is out of the bounds of the input lists')
            if a_start >= len(a):
                return b[b_start + k - 1]
            if b_start >= len(b):
                return a[a_start + k - 1]
            if k == 1:
                return min(a[a_start], b[b_start])
            mid_a, mid_b = float('inf'), float('inf')
            if k // 2 - 1 < len(a) - a_start:
                mid_a = a[a_start + k // 2 - 1]
            if k // 2 - 1 < len(b) - b_start:
                mid_b = b[b_start + k // 2 - 1]
            if mid_a < mid_b:
                return get_kth_smallest(a_start + k // 2, b_start, k - k // 2)
            return get_kth_smallest(a_start, b_start + k // 2, k - k // 2)

        right = get_kth_smallest(0, 0, 1 + (len(a) + len(b)) // 2)
        if (len(a) + len(b)) % 2:
            return right
        left = get_kth_smallest(0, 0, (len(a) + len(b)) // 2)
        return (left + right) / 2

    # 5: /problems/longest-palindromic-substring/
    @staticmethod
    def longest_palindrome_substring(s: str) -> str:
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
        negative = x < 0
        x = abs(x)
        y = 0
        while x != 0:
            y = y * 10 + x % 10
            x //= 10
        if y > 2 ** 31 - 1:
            return 0
        return y if not negative else -y

    # 8: /problems/string-to-integer-atoi/
    @staticmethod
    def my_atoi(s: str) -> int:
        s = s.strip()
        negative = False
        if s and s[0] == '-':
            negative = True
        if s and s[0] in {'+', '-'}:
            s = s[1:]
        if not s:
            return 0
        digits = {i for i in '0123456789'}
        n = 0
        for c in s:
            if c not in digits:
                break
            n = n * 10 + int(c)
        if negative:
            n = -n
        n = max(min(n, 2 ** 31 - 1), -2 ** 31)
        return n

    # 9: /problems/palindrome-number/
    @staticmethod
    def is_palindrome(x: int) -> bool:
        return str(x) == str(x)[::-1]

    # 10: /problems/regular-expression-matching/
    @staticmethod
    def is_match(s: str, p: str) -> bool:
        dp = [[False] * (len(p) + 1)] * (len(s) + 1)
        dp[-1][-1] = True
        for i in range(len(s), -1, -1):
            for j in range(len(p) - 1, -1, -1):
                first_match = i < len(s) and p[j] in {s[i], '.'}
                if j + 1 < len(p) and p[j + 1] == '*':
                    dp[i][j] = dp[i][j + 2] or first_match and dp[i + 1][j]
                else:
                    dp[i][j] = first_match and dp[i + 1][j + 1]
        return dp[0][0]

    # 11: /problems/container-with-most-water/
    @staticmethod
    def max_area(height: list[int]) -> int:
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
        mapping = [
            (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
            (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
            (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'),
            (1, 'I'),
        ]
        romans = []
        for i, roman in mapping:
            while i <= num:
                num -= i
                romans.append(roman)
        return ''.join(romans)

    # 13: /problems/roman-to-integer/
    @staticmethod
    def roman_to_int(s: str) -> int:
        d = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        integer, prev_int = 0, 0
        for roman in s[::-1]:
            if d[roman] >= prev_int:
                prev_int = d[roman]
                integer += d[roman]
            else:
                integer -= d[roman]
        return integer

    # 14: /problems/longest-common-prefix/
    @staticmethod
    def longest_common_prefix(strs: list[str]) -> str:
        if not strs:
            return ''
        if len(strs) == 1:
            return strs[0]
        strs.sort()
        res = ''
        for a, b in zip(strs[0], strs[-1]):
            if a == b:
                res += a
            else:
                break
        return res

    # 15: /problems/3sum
    @staticmethod
    def three_sum(nums: list[int]) -> list[list[int]]:
        import bisect
        ref: dict[int, int] = {}
        res: list[list[int]] = []
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
    def three_sum_closest(nums: list[int], target: int) -> int:
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
                if s > target:
                    k -= 1
        return res

    # 17: /problems/letter-combinations-of-a-phone-number/
    @staticmethod
    def letter_combinations(digits: str) -> list[str]:
        if not digits or '0' in digits or '1' in digits:
            return []
        results = [[]]
        mapping = {
            '2': ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }
        for digit in digits:
            temp = []
            for result in results:
                for letter in mapping[digit]:
                    temp.append(result + [letter])
            results = temp
        return [''.join(result) for result in results]

    # 18: /problems/4sum/
    @staticmethod
    def four_sum(nums: list[int], target: int) -> list[list[int]]:

        def k_sum(n: list[int], t: int, k: int) -> list[list[int]]:
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

        def two_sum(n: list[int], t: int) -> list[list[int]]:
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

    # 19: /problems/remove-nth-node-from-end-of-list/
    @staticmethod
    def remove_nth_from_end(head: ListNode, n: int) -> ListNode:
        first, second = head, head
        for _ in range(n):
            first = first.next
        if not first:
            return head.next
        while first.next:
            first = first.next
            second = second.next
        second.next = second.next.next
        return head

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

    # 21: /problems/merge-two-sorted-lists/
    @staticmethod
    def merge_two_lists(l1: ListNode, l2: ListNode) -> ListNode:
        dummy = prev = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 or l2
        return dummy.next

    # 22: /problems/generate-parentheses/
    @staticmethod
    def generate_parenthesis(n: int) -> list[str]:

        def backtrack(s='', left=0, right=0):
            if len(s) == 2 * n:
                res.append(s)
                return
            if left < n:
                backtrack(s + '(', left + 1, right)
            if right < left:
                backtrack(s + ')', left, right + 1)

        res = []
        backtrack()
        return res

    # 23: /problems/merge-k-sorted-lists/
    @staticmethod
    def merge_k_lists(lists: list[ListNode]) -> ListNode:
        import heapq
        prev = dummy = ListNode(None)
        next_nodes, heap = [], []
        for i, node in enumerate(lists):
            next_nodes.append(node)
            if node:
                heap.append((node.val, i))
        heapq.heapify(heap)
        while heap:
            value, i = heapq.heappop(heap)
            node = next_nodes[i]
            prev.next = node
            prev = prev.next
            if node.next:
                next_nodes[i] = node.next
                heapq.heappush(heap, (node.next.val, i))
        return dummy.next

    # 24: /problems/swap-nodes-in-pairs/
    @staticmethod
    def swap_pairs(head: ListNode) -> ListNode:
        prev = dummy = ListNode(None)
        while head and head.next:
            next_head = head.next.next
            prev.next = head.next
            head.next.next = head
            prev = head
            head = next_head
        prev.next = head
        return dummy.next

    # 25: /problems/reverse-nodes-in-k-group/
    def reverse_k_group(self, head: ListNode, k: int) -> ListNode:
        if k < 2:
            return head
        node = head
        for _ in range(k):
            if not node:
                return head
            node = node.next
        prev = self.reverse_k_group(node, k)
        for _ in range(k):
            temp = head.next
            head.next = prev
            prev = head
            head = temp
        return prev

    # 26: /problems/remove-duplicates-from-sorted-array/
    @staticmethod
    def remove_duplicates(nums: list[int]) -> int:
        next_new = 0
        for i in range(len(nums)):
            if i == 0 or nums[i] != nums[i - 1]:
                nums[next_new] = nums[i]
                next_new += 1
        return next_new

    # 27: /problems/remove-element/
    @staticmethod
    def remove_elements(nums: list[int], val: int) -> int:
        k = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[k] = nums[i]
                k += 1
        return k

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
    def find_substring(s: str, words: list[str]) -> list[int]:
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

    # 31: /problems/next-permutation/
    @staticmethod
    def next_permutation(nums: list[int]) -> None:
        i = len(nums) - 2
        while i >= 0 and nums[i + 1] <= nums[i]:
            i -= 1
        if i >= 0:
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        nums[::] = nums[:i + 1] + nums[i + 1:][::-1]

    # 32: /problems/longest-valid-parentheses/
    @staticmethod
    def longest_valid_parentheses(s: str) -> int:
        max_length = 0
        stack = [-1]
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    max_length = max(max_length, i - stack[-1])
        return max_length

    # 33: /problems/search-in-rotated-sorted-array/
    @staticmethod
    def search(nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1

    # 34: /problems/find-first-and-last-position-of-element-in-sorted-array/
    @staticmethod
    def search_range(nums: list[int], target: int) -> list[int]:

        def binary(tgt, left, right):
            if left > right:
                return left
            mid = (left + right) // 2
            if tgt > nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
            return binary(tgt, left, right)

        lower = binary(target - 0.5, 0, len(nums) - 1)
        upper = binary(target + 0.5, 0, len(nums) - 1)
        return [-1, -1] if lower == upper else [lower, upper - 1]

    # 35: /problems/search-insert-position/
    @staticmethod
    def search_insert(nums: list[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    # 36: /problems/valid-sudoku/
    @staticmethod
    def is_valid_sudoku(board: list[list[str]]) -> bool:
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        for r in range(9):
            for c in range(9):
                digit = board[r][c]
                if digit == '.':
                    continue
                box = r // 3 * 3 + c // 3
                if digit in rows[r] or digit in cols[c] or digit in boxes[box]:
                    return False
                rows[r].add(digit)
                cols[c].add(digit)
                boxes[box].add(digit)
        return True

    # 37: /problems/sudoku-solver/
    @staticmethod
    def solve_sudoku(board: list[list[str]]) -> None:

        def solve(sudoku_board):
            for row in range(9):
                for col in range(9):
                    if sudoku_board[row][col] == '.':
                        for num in map(str, range(1, 10)):
                            if is_valid(sudoku_board, row, col, num):
                                sudoku_board[row][col] = num
                                if solve(sudoku_board):
                                    return True
                                sudoku_board[row][col] = '.'
                        return False
            return True

        def is_valid(sudoku_board, row, col, num):
            for x in range(9):
                if sudoku_board[row][x] == num:
                    return False
            for x in range(9):
                if sudoku_board[x][col] == num:
                    return False
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            for i in range(box_row, box_row + 3):
                for j in range(box_col, box_col + 3):
                    if sudoku_board[i][j] == num:
                        return False
            return True

        if not board or len(board) != 9 or len(board[0]) != 9:
            return
        solve(board)


    # 38: /problems/count-and-say/
    @staticmethod
    def count_and_say(n: int) -> str:
        seq = [1]
        for _ in range(n - 1):
            next_seq = []
            for num in seq:
                if not next_seq or next_seq[-1] != num:
                    next_seq += [1, num]
                else:
                    next_seq[-2] += 1
            seq = next_seq
        return "".join(map(str, seq))

    # 39: /problems/combination-sum/
    @staticmethod
    def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
        res = []
        def helper(nums, nxt, t, p, r):
            if t == 0:
                r.append(p)
                return
            if nxt == len(nums):
                return
            i = 0
            while t - i * nums[nxt] >= 0:
                helper(nums, nxt + 1, t - i * nums[nxt], p + [nums[nxt]] * i, r)
                i += 1
        helper(candidates, 0, target, [], res)
        return res

    # 40: /problems/combination-sum-ii/
    @staticmethod
    def combination_sum_2(candidates: list[int], target: int) -> list[list[int]]:
        import collections
        res = []
        partials = [[]]
        freq = list(collections.Counter(candidates).items())
        for candidate, count in freq:
            new_partials = []
            for partial in partials:
                partial_sum = sum(partial)
                for i in range(count + 1):
                    if partial_sum + candidate * i == target:
                        res.append(partial + [candidate] * i)
                    elif partial_sum + candidate * i < target:
                        new_partials.append(partial + [candidate] * i)
                    else:
                        break
            partials = new_partials
        return res

    # 41: /problems/first-missing-positive/
    @staticmethod
    def first_missing_positive(nums: list[int]) -> int:
        n = len(nums)
        i = 0
        while i < n:
            if 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
            else:
                i += 1
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        return n + 1

    # 42: /problems/trapping-rain-water/
    @staticmethod
    def trap(height: list[int]) -> int:
        if not height or len(height) < 3:
            return 0
        left, right = 0, len(height) - 1
        left_max, right_max = 0, 0
        water_trapped = 0
        while left < right:
            if height[left] < height[right]:
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    water_trapped += left_max - height[left]
                left += 1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                    water_trapped += right_max - height[right]
                right -= 1
        return water_trapped

    # 43: /problems/multiply-strings/
    @staticmethod
    def multiply(num1: str, num2: str) -> str:
        if num1 == '0' or num2 == '0':
            return '0'
        m, n = len(num1), len(num2)
        res = [0] * (m + n)
        num1, num2 = num1[::-1], num2[::-1]
        for i in range(m):
            for j in range(n):
                res[i + j] += int(num1[i]) * int(num2[j])
                res[i + j + 1] += res[i + j] // 10
                res[i + j] %= 10
        start = len(res) - 1
        while start > 0 and res[start] == 0:
            start -= 1
        return ''.join(str(res[i]) for i in range(start, -1, -1))

    # 44: /problems/wildcard-matching/
    @staticmethod
    def is_match_wildcard(s: str, p: str) -> bool:
        i = j = 0
        star_idx = s_idx = -1
        while i < len(s):
            if j < len(p) and (p[j] == '?' or s[i] == p[j]):
                i += 1
                j += 1
            elif j < len(p) and p[j] == '*':
                star_idx = j
                s_idx = i
                j += 1
            elif star_idx != -1:
                j = star_idx + 1
                s_idx += 1
                i = s_idx
            else:
                return False
        while j < len(p) and p[j] == '*':
            j += 1
        return j == len(p)

    # 45: /problems/jump-game-ii/
    @staticmethod
    def jump(nums: list[int]) -> int:
        if len(nums) <= 1:
            return 0
        jumps = 0
        current_end = 0
        farthest = 0
        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])
            if i == current_end:
                jumps += 1
                current_end = farthest
                if current_end >= len(nums) - 1:
                    break
        return jumps

    # 53: /problems/maximum-subarray/
    @staticmethod
    def max_sub_array(nums: list[int]) -> int:
        max_current, max_global = nums[0], nums[0]
        for i in range(1, len(nums)):
            max_current = max(nums[i], max_current + nums[i])
            if max_current > max_global:
                max_global = max_current
        return max_global

    # 54: /problems/spiral-matrix/
    @staticmethod
    def spiral_order(matrix: list[list[int]]) -> list[int]:
        res = []
        while matrix:
            res += matrix.pop(0)
            matrix = list(zip(*matrix))[::-1]
        return res

    # 58: /problems/length-of-last-word/
    @staticmethod
    def length_of_last_word(s: str) -> int:
        return len(s.strip().split(' ')[-1]) if ' ' in s else len(s)

    # 62: /problems/unique-paths/
    @staticmethod
    def unique_paths(m: int, n: int) -> int:
        d = {}

        def dp(x: int, y: int) -> int:
            if x == 1 or y == 1:
                return 1
            if (x, y) not in d:
                d[x, y] = dp(x, y - 1) + dp(x - 1, y)
            return d[x, y]

        return dp(m, n)

    # 66: /problems/plus-one/
    @staticmethod
    def plus_one(digits: list[int]) -> list[int]:
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

        def dp(m: int) -> int:
            if m in d:
                return d[m]
            d[m] = dp(m - 1) + dp(m - 2)
            return d[m]

        return dp(n)

    # 80: /problems/remove-duplicates-from-sorted-array-ii/
    @staticmethod
    def remove_duplicates_ii(nums: list[int]) -> int:
        for x in set(nums):
            if nums.count(x) > 2:
                while nums.count(x) > 2:
                    nums.remove(x)
        return len(nums)

    # 88: /problems/merge-sorted-array/
    @staticmethod
    def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
        nums1[m:] = nums2[:n]
        nums1.sort()

    # 98: /problems/validate-binary-search-tree/
    @staticmethod
    def is_valid_bst(root: TreeNode) -> bool:
        import math
        if not root:
            return True
        stack = [(root, -math.inf, math.inf)]
        while stack:
            root, lower, upper = stack.pop()
            if not root:
                continue
            val = root.val
            if val <= lower or val >= upper:
                return False
            stack.append((root.right, val, upper))
            stack.append((root.left, lower, val))
        return True

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

    # 102: /problems/binary-tree-level-order-traversal/
    @staticmethod
    def level_order(root: TreeNode) -> list[list[int]]:
        queue = res = []
        if root:
            queue = [root]
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        return res

    # 121: /problems/best-time-to-buy-and-sell-stock/
    @staticmethod
    def max_profit(prices: list[int]) -> int:
        max_profit = 0
        price_buy = price_sell = prices[0]
        for price in prices:
            if price < price_buy:
                price_buy = price_sell = price
            if price > price_sell:
                price_sell = price
                profit = price_sell - price_buy
                if profit > max_profit:
                    max_profit = profit
        return max_profit

    # 142: /problems/linked-list-cycle-ii/
    @staticmethod
    def detect_cycle(head: ListNode):
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                fast = head
                while fast != slow:
                    fast = fast.next
                    slow = slow.next
                return fast
        return None

    # 200: /problems/number-of-islands/
    @staticmethod
    def num_islands(grid: list[list[str]]) -> int:

        def dfs(r: int, c: int) -> None:
            if (0 <= r < len(grid)
                    and 0 <= c < len(grid[0])
                    and grid[r][c] == '1'):
                grid[r][c] = '#'
                dfs(r - 1, c)
                dfs(r + 1, c)
                dfs(r, c - 1)
                dfs(r, c + 1)

        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    dfs(i, j)
                    count += 1
        return count

    # 202: /problems/happy-number/
    @staticmethod
    def is_happy(n: int) -> bool:
        s = set()
        while n != 1:
            if n in s:
                return False
            else:
                s.add(n)
                n = sum([int(c) ** 2 for c in str(n)])
        return True

    # 205: /problems/isomorphic-strings/
    @staticmethod
    def isomorphic_strings(s: str, t: str) -> bool:
        d_s, d_t = {}, {}
        for c_s, c_t in zip(s, t):
            if c_s not in d_s:
                d_s[c_s] = c_t
            elif d_s[c_s] != c_t:
                return False
            if c_t not in d_t:
                d_t[c_t] = c_s
            elif d_t[c_t] != c_s:
                return False
        return True

    # 206: /problems/reverse-linked-list/
    @staticmethod
    def reverse_list(head: ListNode) -> ListNode:
        prev = None
        curr = head
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev

    # 217: /problems/contains-duplicate/
    @staticmethod
    def contains_duplicate(nums: list[int]) -> bool:
        s = set()
        for n in nums:
            if n in s:
                return True
            else:
                s.add(n)
        return False

    # 235: /problems/lowest-common-ancestor-of-a-binary-search-tree/
    @staticmethod
    def lowest_common_ancestor(root: TreeNode,
                               p: TreeNode,
                               q: TreeNode) -> None or TreeNode:
        large = max(p.val, q.val)
        small = min(p.val, q.val)
        while root:
            if root.val > large:
                root = root.left
            elif root.val < small:
                root = root.right
            else:
                return root
        return None

    # 392: /problems/is-subsequence/
    @staticmethod
    def is_subsequence(s: str, t: str) -> bool:
        i_t = -1
        for c in s:
            try:
                i_t = t.index(c, i_t + 1, len(t))
            except ValueError:
                return False
        return True

    # 409: /problems/longest-palindrome/
    @staticmethod
    def longest_palindrome(s: str) -> int:
        d = {}
        for c in s:
            if c in d:
                d[c] += 1
            else:
                d[c] = 1
        length = 0
        odd_flag = 0
        for v in d.values():
            if v % 2:
                length += v - 1
                if not odd_flag:
                    odd_flag = 1
            else:
                length += v
        if odd_flag:
            length += 1
        return length

    # 438: /problems/find-all-anagrams-in-a-string/
    @staticmethod
    def find_anagrams(s: str, p: str) -> list[int]:
        res = []
        if len(s) < len(p):
            return res
        d_p = {}
        for c in p:
            if c in d_p:
                d_p[c] += 1
            else:
                d_p[c] = 1
        d_w = {}
        for c in s[:len(p)]:
            if c in d_w:
                d_w[c] += 1
            else:
                d_w[c] = 1
        for i in range(len(s) - len(p) + 1):
            w = s[i:i + len(p)]
            if i:
                if d_w[s[i - 1]] > 1:
                    d_w[s[i - 1]] -= 1
                else:
                    del d_w[s[i - 1]]
                if w[-1] in d_w:
                    d_w[w[-1]] += 1
                else:
                    d_w[w[-1]] = 1
            if d_p == d_w:
                res.append(i)
        return res

    # 509: /problems/fibonacci-number/
    @staticmethod
    def fib(n: int) -> int:
        d = {0: 0, 1: 1}

        def dp(m: int) -> int:
            if m in d:
                return d[m]
            d[m] = dp(m - 1) + dp(m - 2)
            return d[m]

        return dp(n)

    # 589: /problems/n-ary-tree-preorder-traversal/
    @staticmethod
    def preorder(root: Node) -> list[int]:
        if not root:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            stack += node.children[::-1]
        return res

    # 704: /problems/binary-search/
    @staticmethod
    def binary_search(nums: list[int], target: int) -> int:
        i_l, i_r = 0, len(nums)
        while True:
            i_m = (i_l + i_r) // 2
            if nums[i_m] == target:
                return i_m
            if nums[i_m] < target:
                if i_l == i_m:
                    break
                else:
                    i_l = i_m
            if nums[i_m] > target:
                if i_r == i_m:
                    break
                else:
                    i_r = i_m
        return -1

    # 724: /problems/find-pivot-index/
    @staticmethod
    def pivot_index(nums: list[int]) -> int:
        sum_l, sum_r = 0, sum(nums)
        for i, num in enumerate(nums):
            sum_r -= num
            if sum_l == sum_r:
                return i
            sum_l += num
        return -1

    # 733: /problems/flood-fill/
    @staticmethod
    def flood_fill(image: list[list[int]], sr: int, sc: int, color: int) -> list[list[int]]:
        start_color = image[sr][sc]
        filled = set()

        def dfs(r: int, c: int):
            if ((r, c) not in filled
                    and 0 <= r < len(image)
                    and 0 <= c < len(image[0])
                    and image[r][c] == start_color):
                filled.add((r, c))
                image[r][c] = color
                dfs(r - 1, c)
                dfs(r + 1, c)
                dfs(r, c - 1)
                dfs(r, c + 1)
            return

        dfs(sr, sc)
        return image

    # 746: /problems/min-cost-climbing-stairs/
    @staticmethod
    def min_cost_climbing_stairs(cost: list[int]) -> int:
        a, b = cost[0], cost[1]
        for i in range(2, len(cost)):
            a, b = b, min(a, b) + cost[i]
        return min(a, b)

    # 844: /problems/backspace-string-compare/
    @staticmethod
    def backspace_compare(s: str, t: str) -> bool:

        def backspace(s0: str) -> str:
            s1 = ''
            for c in s0:
                if c == '#' and s1:
                    s1 = s1[:-1]
                if c != '#':
                    s1 += c
            return s1

        return backspace(s) == backspace(t)

    # 876: /problems/middle-of-the-linked-list/
    @staticmethod
    def middle_node(head: ListNode) -> ListNode:
        dummy = temp = head
        i = 0
        while temp:
            temp = temp.next
            i += 1
        for _ in range(i // 2):
            dummy = dummy.next
        return dummy

    # 1046: /problems/last-stone-weight/
    @staticmethod
    def last_stone_weight(stones: list[int]) -> int:
        while stones:
            if len(stones) == 1:
                return stones[0]
            stones = sorted(stones)
            stones[-2] = stones[-1] - stones[-2]
            stones.pop()
            if not stones[-1]:
                stones.pop()
            if not stones:
                return 0
        return stones[0]

    # 1480: /problems/running-sum-of-1d-array/
    @staticmethod
    def running_sum(nums: list[int]) -> list[int]:
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        return nums

    # 1706: /problems/where-will-the-ball-fall/
    @staticmethod
    def find_ball(grid: list[list[int]]) -> list[int]:
        res = [0] * len(grid[0])
        for c in range(len(grid[0])):
            current_c = c
            for r in range(len(grid)):
                next_c = current_c + grid[r][current_c]
                if (next_c < 0 or next_c > len(grid[0]) - 1
                        or grid[r][current_c] != grid[r][next_c]):
                    res[c] = -1
                    break
                res[c] = current_c = next_c
        return res
