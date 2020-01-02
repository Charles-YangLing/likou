#单链表
class ListNode(object):
    """
    单向链表
    """
    def __init__(self, x):
        self.val = x
        self.next = None
#双向链表
class ListNodep(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        self.prev = None #指向上一级
#多级双向链表
class Node(object):
    def __init__(self, val, prev, next, child):
        self.val = val
        self.next = next
        self.prev = prev
        self.child = child #指向额外的分支