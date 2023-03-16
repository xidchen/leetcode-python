class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next_one=None):
        self.val = val
        self.next = next_one


class Node:
    def __iter__(self, val=None, children=None):
        self.val = val
        self.children = children
