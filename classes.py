class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Node:
    def __iter__(self, val=None, children=None):
        self.val = val
        self.children = children


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
