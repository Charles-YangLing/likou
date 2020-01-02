from ListNode import ListNode
from ListNode import ListNodep
from ListNode import Node
from TreeNode import TreeNode
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def __init__(self):
        pass
    def plusOne(self, digits):
        "@66加一"
        sums = 0
        for i in digits:
            sums = sums * 10 + i  # 10进制乘以10，进行累和；
        sums_str = str(sums + 1)
        return [int(j) for j in sums_str]

    def findDiagonalOrder(self, matrix):
        """
        @498对角线遍历
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if len(matrix) == 0:
            return []
        M, N, result = len(matrix), len(matrix[0]), []
        for curve_line in range(M + N - 1):
            row_begin = 0 if curve_line + 1 <= N else curve_line + 1 - N
            row_end = M - 1 if curve_line + 1 >= M else curve_line
            if curve_line % 2 == 1:
                for i in range(row_begin, row_end + 1):
                    result.append(matrix[i][curve_line - i])
            else:
                for i in range(row_end, row_begin - 1, -1):
                    result.append(matrix[i][curve_line - i])
        return result

    def spiralOrder(self, matrix):
        """
        @54螺旋矩阵
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix: return []
        R, C = len(matrix), len(matrix[0])
        seen = [[False] * C for _ in matrix]
        ans = []
        dr = [0, 1, 0, -1]
        dc = [1, 0, -1, 0]
        r = c = di = 0
        for _ in range(R * C):
            ans.append(matrix[r][c])
            seen[r][c] = True
            cr, cc = r + dr[di], c + dc[di]
            if 0 <= cr < R and 0 <= cc < C and not seen[cr][cc]:
                r, c = cr, cc
            else:
                di = (di + 1) % 4
                r, c = r + dr[di], c + dc[di]
        return ans

    def getxiayihang(self,marth):
        result=[];
        for index in range(0,len(marth)):
            if index == 0:
                result.append(1)
                diyi =False
                continue
            if index == len(marth) - 1:
                result.append(1 + marth[-2])
                result.append(1)
                break
            else:
                result.append(marth[index-1]+marth[index])
        return result

    def generate(self, numRows):
        """
        @118杨辉三角
        :type numRows: int
        :rtype: List[List[int]]
        """
        result =[[1],[1,1]]
        ceng =[1,1]
        for i in range(0,numRows):
            if i > 1:
                ceng = Solution.getxiayihang(Solution,marth=ceng)
                result.append(ceng)
        return result

    def addBinary(self, a, b):
        """
        @67二进制求和
        :type a: str
        :type b: str
        :rtype: str
        """
        # inta = int(a,2)
        # intb = int(b,2)
        # result = bin(inta + intb)
        # return  str(result).replace('0b','')
        return str(bin(int(a,2)+int(b,2))).replace('0b','')
        str.index()

    def strStr(self, haystack, needle):
        """
        @28实现 strStr() 函数。
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if needle=="" or haystack == needle:
            return 0
        if needle in haystack:
            for index in range(0,len(haystack)):
                if haystack[index] ==needle[0]:
                    if len(needle)>1:
                        for index2 in range(1,len(needle)):
                            if haystack[index+index2]!=needle[index2]:
                                break
                            if index2 == len(needle)-1:
                                return index
                    else:
                        return index
        else:
             return -1
        # return haystack.index(needle) #上面的代码实现的就是本身py这一句话的功能

    def reverseString(self, s):
        """
        @344反转字符串
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        temp=''
        for index in range(0,len(s)):
            if index < len(s)-index:
                temp = s[index]
                s[index] = s[len(s)-index-1]
                s[len(s)-index-1] = temp
        return s

    def arrayPairSum(self, nums):
        """
        @561数组拆分
        :type nums: List[int]
        :rtype: int
        """
        minx = 0
        nums.sort()
        for index in range(0,len(nums)):
            if index % 2 == 0:
                minx += min(nums[index],nums[index+1])
        return minx

    def twoSum(self, numbers, target):
        """
        @167两数之和 II - 输入有序数组
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        # for index in range(0,len(numbers)):  #时间复杂度过大
        #     for index2 in range(index,len(numbers)):
        #         if  numbers[index] + numbers[index2] == target:
        #             return [index,index2]
        minx,maxx = 0,len(numbers)-1
        sumx = 0
        while maxx > minx:
            sumx = numbers[minx] + numbers[maxx]
            if sumx == target:
                return [minx+1,maxx+1]
            elif sumx < target:
                minx += 1
                continue
            elif sumx > target:
                maxx -= 1
                continue

    def removeElement(self, nums, val):
        """
        @27移除元素
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        chang = len(nums)
        for index in range(0,chang):
            if nums[chang-1-index] ==val:
                del nums[chang-1-index]
        return nums

    def findMaxConsecutiveOnes(self, nums):
        """
        @485最大连续1的个数
        :type nums: List[int]
        :rtype: int
        """
        i,j,chang = 0,0,0
        lennum = len(nums)
        while i < lennum:
            if nums[i] == 1:
                j = i
                while lennum-i > 0:
                    if nums[i] ==1:
                        i += 1
                        if chang < i-j:
                            chang = i - j
                        continue
                    else:
                        break;
            else:i+=1
        return chang

        # cnt = 0  别人写的,思路一样但更聪明些
        # res = 0
        # for num in nums:
        #     if num == 1:
        #         cnt += 1
        #         res = max(res, cnt)
        #     else:
        #         cnt = 0
        # return res

    def minSubArrayLen(self, s, nums):
        """
        @209长度最小的子数组
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        if not nums: return 0
        left = 0
        cur = 0
        res = float("inf")
        for right in range(len(nums)):
            cur += nums[right]
            while cur >= s:
                res = min(res, right - left + 1)
                cur -= nums[left]
                left += 1
        return res if res != float("inf") else 0

    def rotate(self, nums, k):
        """
        @189 旋转数组
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        k %= n
        nums[:] = nums[-k:] + nums[:-k]

    def getRow(self,rowIndex):
        """
        @119 杨辉三角2
        :type rowIndex: int
        :rtype: List[int]
        """

        # j行的数据, 应该由j - 1行的数据计算出来.
        # 假设j - 1行为[1,3,3,1], 那么我们前面插入一个0(j行的数据会比j-1行多一个),
        # 然后执行相加[0+1,1+3,3+3,3+1,1] = [1,4,6,4,1], 最后一个1保留即可.
        r = [1]
        for i in range(1, rowIndex + 1):
            r.insert(0, 0)
            # 因为i行的数据长度为i+1, 所以j+1不会越界, 并且最后一个1不会被修改.
            for j in range(i):
                r[j] = r[j] + r[j + 1]
        return r

    def reverseWords(self, s):
        s = s.strip()
        res = ""
        i, j = len(s) - 1, len(s)
        while i > 0:
            if s[i] == ' ':
                res += s[i + 1: j] + ' '
                while s[i] == ' ': i -= 1
                j = i + 1
            i -= 1
        return res + s[:j]
        # return " ".join(s.split()[::-1]) 一句话解决的,切片操作str[::-1]可实现字符串翻转

    def reverseWords3(self, s):
        """
        @557反转字符串中的单词 III
        :type s: str
        :rtype: str
        """
        res=""
        for s in s.split():
            res += s[::-1]+" "
        return res[0:-1]

    def removeDuplicates(self, nums):
        """
        @26删除排序数组中的重复项
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) ==0:return 0
        i = 0
        for j in range(1,len(nums)):
            if nums[j] != nums[i]:
                i+=1
                nums[i] = nums[j]
        return i +1

    def moveZeroes(self, nums):
        """
        @283移动零
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        i,j =0,0
        while i < len(nums):
            if nums[i] == 0:
                del nums[i]
                j+=1
            else:i+=1
        for x in range(j):
            nums.append(0)
        return nums

    def hasCycle(self, head):
        """
        @141 环形链表
        :type head: ListNode
        :rtype: bool
        """
        #利用set求解
        # s = set()
        # while head:
        #     # 如果某个节点在set中，说明遍历到重复元素了，也就是有环
        #     if head in s:
        #         return True
        #     s.add(head)
        #     head = head.next
        # return False

        #利用字典法
        s = {}
        i=0
        while head:
            # 如果某个节点在set中，说明遍历到重复元素了，也就是有环
            if head in s.keys():
                return "tail connects to node index %s" % s[head]
            s[head] = i
            head = head.next
            i+=1
        return  "no cycle"

        #快慢指针法
        # if not (head and head.next):
        #     return False
        # # 定义两个指针i和j，i为慢指针，j为快指针
        # i, j = head, head.next
        # while j and j.next:
        #     if i == j:
        #         return True
        #     # i每次走一步，j每次走两步
        #     i, j = i.next, j.next.next
        # return False

    def getIntersectionNode(self, headA, headB):
        """
        @160. 相交链表
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        # 暴力法:时间复杂度为 m*n
        # while headA != None:
        #     list = headB
        #     while list != None:
        #         if headA.next == list.next:
        #             return 'Intersected at %s' %headA.next.val
        #         else:
        #             list = list.next
        #     headA = headA.next
        # return None

        #推荐这种双指针的，总会有差到两值相等，时间复杂度 m+n
        ha, hb = headA, headB
        while ha != hb:
            ha = ha.next if ha else headB
            hb = hb.next if hb else headA
        return ha

    def removeNthFromEnd(self, head, n):
        """
        @19. 删除链表的倒数第N个节点
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        # 增加一个特殊节点方便边界判断
        p = ListNode(-1)
        p.next, a, b = head, p, p
        # 第一个循环，b指针先往前走n步
        while n > 0 and b:
            b, n = b.next, n - 1
        # 假设整个链表长5，n是10，那么第一次遍历完后b就等用于空了
        # 于是后面的判断就不用做了，直接返回
        if not b:
            return head
        # 第二次，b指针走到链表最后，a指针也跟着走
        # 当遍历结束时，a指针就指向要删除的节点的前一个位置
        while b.next:
            a, b = a.next, b.next
        # 删除节点并返回
        a.next = a.next.next
        return p.next

    def reverseList(self, head):
        """
        @206. 反转链表
        :type head: ListNode
        :rtype: ListNode
        """
        p,dic,i = ListNode(-1),{},0
        a = p
        while head != None:
            dic[i] = head
            i+=1
            head = head.next
        b = len(dic)
        for x in range(b):
            a.next = dic[b-x-1]
            a = a.next
        return p.next

        # # 申请两个节点，pre和 cur，pre指向None
        # pre = None
        # cur = head
        # # 遍历链表，while循环里面的内容其实可以写成一行
        # while cur:
        #     # 记录当前节点的下一个节点
        #     tmp = cur.next
        #     # 然后将当前节点指向pre
        #     cur.next = pre
        #     # pre和cur节点都前进一位
        #     pre = cur
        #     cur = tmp
        # return pre

    def removeElements(self, head, val):
        """
        @203移除链表元素
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        # if head == None :
        #     return None
        # p = ListNode(0)
        # p.next = head
        # while head !=None:
        #     if head.val ==val:
        #         if head.next == None:
        #             if p.next == head:
        #                 p.next = head.next
        #                 head = head.next
        #             else:
        #                 head = None
        #         elif p.next ==head:
        #             p.next =head.next
        #             head = head.next
        #         else:
        #             head.val = head.next.val
        #             if head.next.next == None:
        #                 head.next = None
        #                 head = head.next
        #             else:
        #                 head.next = head.next.next
        #                 head = head.next
        #     elif head.next == None:
        #         return p.next
        #     elif head.next.val == val:
        #         if head.next.next != None:
        #             head.next =head.next.next
        #             head = head.next
        #         else:
        #             head.next = None
        #             head = head.next
        #     elif head.next !=None:
        #         head = head.next
        # return p.next
        temp = ListNode(0)
        temp.next = head
        prev = temp

        while (prev.next != None):
            if (prev.next.val == val):
                prev.next = prev.next.next
            else:
                prev = prev.next

        return temp.next

    def oddEvenList(self, head):
        """

        :type head: ListNode
        :rtype: ListNode
        """
        if head ==None:
            return None
        j = ListNode(-1)
        o = ListNode(-2)
        a,b = j,o
        ji = True
        while head !=None:
            if ji and head.next !=None:
                j.next = head
                head = head.next
                j = j.next
                ji =False
            elif not ji and head.next !=None:
                o.next = head
                head = head.next
                o = o.next
                ji = True
            elif ji and head.next == None:
                j.next = head
                head = head.next
                j = j.next
                o.next =None
                j.next = b.next
            elif not ji and head.next == None:
                o.next = head
                head = head.next
                o = o.next
                o.next = None
                j.next = b.next
        return a.next

    def replaceElements(self, arr):
        """
        :type arr: List[int]
        :rtype: List[int]
        """
        maxi = -1
        ans = []
        for i in reversed(arr):
            ans.append(maxi)
            maxi = max(maxi, i)
        return list(reversed(ans))

        # chang = len(arr)
        # x = 0
        # while x< chang -1:
        #     arr[x] = max(arr[x+1:chang])
        #     x+=1
        # arr[chang-1] = -1
        # return arr

    def sumZero(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        res = []
        x = n/2
        if n%2 ==0:
            for i in range(0,n//2):
                res.append(i+1)
                res.append(-(i+1))
        else:
            for i in range(0, n // 2):
                res.append(i+1)
                res.append(-(i+1))
            res.append(0)
        return res

    def getAllElements(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: List[int]
        """

        def construct_paths(root, res):
            if root:
                res.append(root.val)
                if root.left==None and root.right==None:  # 当前节点是叶子节点
                    pass  # 把路径加入到答案中
                if root.left !=None :
                     # 当前节点不是叶子节点，继续递归遍历
                    construct_paths(root.left, res)
                if root.right !=None :
                    construct_paths(root.right, res)
        res = []
        construct_paths(root1, res)
        construct_paths(root2, res)
        res.sort()
        return res

    def canReach(self, arr, start):
        """
        @1306跳跃游戏 III  (经典化dfs题)
        :type arr: List[int]
        :type start: int
        :rtype: bool
        """
        q, v, n = [start], {start}, len(arr)
        while q:
            p = []
            for i in q:
                if not arr[i]:
                    return True
                for j in i - arr[i], i + arr[i]:
                    if 0 <= j < n and j not in v:
                        p.append(j)
                        v.add(j)
            q = p
        return False

    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """

    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """

    def addTwoNumbers(self, l1, l2):
        """
        @2 两数相加
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        shu1,shu2=0,0
        i = 0
        while l1 !=None:
            shu1 += l1.val*10**i
            i+=1
            l1 = l1.next
        i =0
        while l2 !=None:
            shu2 += l2.val*10**i
            i+=1
            l2 = l2.next
        res = str(shu1+shu2)
        p = ListNode(0)
        a = p
        for j in range(len(res)):
            x= int(res[len(res)-j-1])
            p.next = ListNode(x)
            p = p.next
        return a.next

    def flatten(self, head):
        """
        @430. 扁平化多级双向链表
        :type head: Node
        :rtype: Node
        """
        if not head:
            return head

        # pseudo head to ensure the `prev` pointer is never none
        pseudoHead = Node(None, None, head, None)
        self.flatten_dfs(pseudoHead, head)

        # detach the pseudo head from the real head
        pseudoHead.next.prev = None
        return pseudoHead.next

    def flatten_dfs(self, prev, curr):
        """ return the tail of the flatten list """
        if not curr:
            return prev

        curr.prev = prev
        prev.next = curr

        # the curr.next would be tempered in the recursive function
        tempNext = curr.next
        tail = self.flatten_dfs(curr, curr.child)
        curr.child = None
        return self.flatten_dfs(tail, tempNext)























