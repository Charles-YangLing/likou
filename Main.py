import Solution
import ListNode
import TreeNode

user = Solution.Solution()
# list = [[1,2],[2,3],[3,4]]
# list = [1,1]
# jinxing = user.getxiayihang(marth=list)
#jinxing = user.removeElement([0,1,2,2,3,0,4,2],2)
lianbiao = ListNode.ListNode(1)
lianbiao.next = ListNode.ListNode(2)
lianbiao.next.next = ListNode.ListNode(3)
lianbiao.next.next.next = ListNode.ListNode(4)
lianbiao.next.next.next.next = ListNode.ListNode(5)
# lianbiao.next.next.next.next.next = ListNode.ListNode(1)
# lianbiao.next.next.next.next.next.next = ListNode.ListNode(6)
# lianbiao2 = ListNode.ListNode(5)
# lianbiao2.next = ListNode.ListNode(0)
# lianbiao2.next.next = ListNode.ListNode(1)
# lianbiao2.next.next.next = lianbiao.next.next

TreeNode1 = TreeNode.TreeNode(2)
TreeNode1.left = TreeNode.TreeNode(1)
TreeNode1.right = TreeNode.TreeNode(4)
TreeNode2 = TreeNode.TreeNode(1)
TreeNode2.left = TreeNode.TreeNode(0)
TreeNode2.right = TreeNode.TreeNode(3)
jinxing = user.getAllElements(TreeNode1,TreeNode2)
print(jinxing)

