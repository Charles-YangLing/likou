import collections

from ListNode import ListNode
from ListNode import ListNodep
from ListNode import Node
from TreeNode import TreeNode
from math import e, log
import math
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

    def maxProfit(self, prices):
        """
        @122买卖股票的最佳时机 II
        :type prices: List[int]
        :rtype: int
        """
        profit = 0
        for i in range(1, len(prices)):
            tmp = prices[i] - prices[i - 1]
            if tmp > 0: profit += tmp
        return profit

    def containsDuplicate(self, nums):
        """
        @217. 存在重复元素
        :type nums: List[int]
        :rtype: bool
        """
        s =set()
        for i in range(len(nums)):
            if nums[i] in s:
                return True
            else:s.add(nums[i])
        return False

    @staticmethod #静态方法的定义
    def singleNumber( nums):
        """
        @136. 只出现一次的数字
        :type nums: List[int]
        :rtype: int
        """
        # 超时 占性能
        # for i in nums:
        #     if nums.count(i)==1:
        #         return i
        hash_table = {}
        for i in nums:
            try:
                hash_table.pop(i)
            except:
                hash_table[i] = 1
        return hash_table.popitem()[0]

    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        nums1.sort()
        nums2.sort()
        r = []
        left, right = 0, 0
        while left < len(nums1) and right < len(nums2):
            if nums1[left] < nums2[right]:
                left += 1
            elif nums1[left] == nums2[right]:
                r.append(nums1[left])
                left += 1
                right += 1
            else:
                right += 1
        return r

    def freqAlphabets(self, s):
        """
        :type s: str
        :rtype: str
        """
        chang = len(s)
        i =0
        res=''
        while i <= chang-1:
            if i+2<=chang-1 and s[i+2]=='#':
                res+=self.convertToTitle(self,int(s[i:i+2]))
                i+=3
            else:
                res+=self.convertToTitle(self,int(s[i]))
                i+=1
        return res

    def convertToTitle(self,n):
        res = ""
        while n:
            n -= 1
            n, y = divmod(n, 26)
            res = chr(y + 97) + res
        return res

    def xorQueries(self, arr, queries):
        """
        :type arr: List[int]
        :type queries: List[List[int]]
        :rtype: List[int]
        """
        res = []
        for index in queries:
            yihuo = 0
            for i in range(index[0],index[1]+1):
                yihuo = yihuo^arr[i]
            res.append(yihuo)
        return res

    def reverse(self, x):
        """
        @7. 整数反转
        :type x: int
        :rtype: int
        """
        res,zheng = 0,True
        if x<0:
            x=-x
            zheng =False
        while x!=0:
            temp = x %10
            res = res*10+temp
            x=x//10
        if res>2147483647:
            return 0
        elif zheng:
            return res
        else:
            return -res

    def firstUniqChar(self, s):
        """
        @387. 字符串中的第一个唯一字符
        :type s: str
        :rtype: int
        """
        # 不知道哪里有错，编辑器和运行完结果不一致
        # dic,i={},0
        # while i<len(s):
        #     if dic.get(s[i]):
        #         dic[s[i]]+=1
        #     else:
        #         dic[s[i]]=1
        #     i+=1
        # for key,val in dic.items():
        #     if val==1:
        #         res = s.index(key)
        #         return res
        # return -1

        dic = {}

        # 记录字符出现次数
        for c in s:
            dic[c] = dic[c] + 1 if c in dic else 1

        # 过滤出现次数不为一的字符
        unique_chars = [k for k, v in filter(lambda kvp: kvp[1] == 1, dic.items())]
        # 遍历目标字符串，返回首个出现在unique_chars中的字符的索引
        for i, c in enumerate(s):
            if c in unique_chars:
                return i

        return -1

    def isAnagram(self, s, t):
        """
        242. 有效的字母异位词
        :type s: str
        :type t: str
        :rtype: bool
        """
        jihe ={}
        for i  in s:
            if i not in jihe:
                jihe[i]=1
            else:
                jihe[i]=jihe[i]+1
        for d in t:
            if d not in jihe:
                return False
            else:
                jihe[d]=jihe[d]-1
                if jihe[d]==0:
                    del jihe[d]
        if len(jihe)==0:
            return True
        else:
            return False
    # list(map(chr, range(ord('a'), ord('z') + 1)))  遍历字母
    # [chr(x) for x in range(ord('A'), ord('Z') + 1)]

    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        newstr = ''
        i=0
        while i < len(s):
            if ord(s[i]) in range(65,91) or ord(s[i]) in range(48,58) or ord(s[i]) in range(97,128):
                newstr+=s[i]
            i+=1
        if newstr=='':
            return True
        yi,er=0,len(newstr)-1
        newstr=newstr.lower()
        while yi!=er:
            if newstr[yi]==newstr[er]:
                if yi+1==er:
                    return True
                yi+=1
                er-=1
            else:
                return False
        return True

    def myAtoi(self, str):
        """
        @8. 字符串转换整数 (atoi)
        没啥意义  懒得写了
        :type str: str
        :rtype: int
        """
        # str = str.replace(' ','')
        # res=0
        # i=0
        # if str[0]=='-':
        #     i=1
        #     while i<len(str):
        #         if (int(str[i])>=0 and int(str[i])<=9):
        #             str
        #         i+=1
        # elif str[0]!='0':
        #     while i <len(str):
        #         if str[0]
        # else:
        #     return 0
        # return max(min(int(*re.findall('^[\+\-]?\d+', str.lstrip())), 2 ** 31 - 1), -2 ** 31) 正则一句搞定

    def countAndSay(self, n):
        """
        @38. 外观数列
        :type n: int
        :rtype: str
        """
        s='11'
        if n==1:
            return "1"
        if n==2:
            return "11"
        while n>2:
            s = Solution.waiguan(self,s)
            n-=1
        return s
    def waiguan(self,s):
        res=''
        i,j =0,1
        while j<len(s):
            if s[i]==s[j]:
                j+=1
            else:
                res+=str(j-i)+s[i]
                i=j
                j+=1
        res+=str(j-i)+s[i]
        return res

    def deleteNode(self, node):
        """
        @237 删除链表中的节点
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next

    def maxDepth(self, root):
        """
        @104. 二叉树的最大深度
        :type root: TreeNode
        :rtype: int
        """
        # 递归思想
        # if root==None:
        #     return 0
        # else:
        #     left_height = self.maxDepth(root.left)
        #     right_height = self.maxDepth(root.right)
        #     return max(left_height, right_height) + 1
        # 迭代DFS
        stack = []
        if root is not None:
            stack.append((1, root))

        depth = 0 #最大深度
        while stack != []:
            current_depth, root = stack.pop() #当前深度和节点
            if root is not None:
                depth = max(depth, current_depth) #每一步都更新深度
                stack.append((current_depth + 1, root.left)) #将后面的节点也推入栈
                stack.append((current_depth + 1, root.right))

        return depth

    def isValidBST(self, root):
        """
        @98. 验证二叉搜索树
        :type root: TreeNode
        :rtype: bool
        """

        # def helper(node, lower=float('-inf'), upper=float('inf')):
        #     if not node:
        #         return True
        #
        #     val = node.val
        #     if val <= lower or val >= upper:
        #         return False
        #
        #     if not helper(node.right, val, upper):
        #         return False
        #     if not helper(node.left, lower, val):
        #         return False
        #     return True
        #
        # return helper(root)

        stack, inorder = [], float('-inf')

        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            # If next element in inorder traversal
            # is smaller than the previous one
            # that's not BST.
            if root.val <= inorder:
                return False
            inorder = root.val
            root = root.right

        return True

    def isSymmetric(self, root):
        """
        @101. 对称二叉树
        :type root: TreeNode
        :rtype: bool
        """
        if root==None:
            return True
        else:
            left_val=[root.left.val]
            right_val=[root.right.val]
            if left_val == right_val.reverse():
                return True

    def levelOrder(self, root):
        """
        @102 二叉树的层次遍历
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        # levels = []
        # if not root:
        #     return levels
        #
        # def helper(node, level):
        #     # start the current level
        #     if len(levels) == level:
        #         levels.append([])
        #
        #     # append the current node value
        #     levels[level].append(node.val)
        #
        #     # process child nodes for the next level
        #     if node.left:
        #         helper(node.left, level + 1)
        #     if node.right:
        #         helper(node.right, level + 1)
        #
        # helper(root, 0)
        # return levels
        levels = []
        if not root:
            return levels

        level = 0
        queue = deque([root, ])
        while queue:
            # start the current level
            levels.append([])
            # number of elements in the current level
            level_length = len(queue)

            for i in range(level_length):
                node = queue.popleft()
                # fulfill the current level
                levels[level].append(node.val)

                # add child nodes of the current level
                # in the queue for the next level
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            # go to next level
            level += 1

        return levels

    def sortedArrayToBST(self, nums):
        """
        @108. 将有序数组转换为二叉搜索树
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return []

        # 找到中点作为根节点
        mid = len(nums) // 2
        node = TreeNode(nums[mid])

        # 左侧数组作为左子树
        left = nums[:mid]
        right = nums[mid + 1:]

        # 递归调用
        node.left = self.sortedArrayToBST(left)
        node.right = self.sortedArrayToBST(right)

        return node

    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        for i in nums2:
            nums1.append(i)
        nums1.remove(0)
        nums1.sort()

    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left =1
        right = n
        while left<right:
            mid = left+(right-left)//2
            if isBadVersion(mid):
                right=left
            else:
                left = mid+1
        return left

    def climbStairs(self, n):
        """
        @70 爬楼梯 ???
        :type n: int
        :rtype: int
        """
        dp = [0] * (n + 1)
        dp[1] = 1
        if (n < 2):
            return dp[n]
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    def maxProfit(self, prices):
        """
        @121 买卖股票的最佳时间 ???
        :type prices: List[int]
        :rtype: int
        """
        if len(prices)<=1:
            return 0
        diff = [0 for _ in range(len(prices)-1)]
        for i in range(len(prices)-1):
            diff[i] = prices[i+1]-prices[i]
        dp = [0 for _ in range(len(prices)-1)]
        dp[0] = max(0, diff[0])
        max_profit = dp[0]
        for i in range(1, len(prices)-1):
            dp[i] = max(0, diff[i]+dp[i-1])
            max_profit = max(max_profit, dp[i])
        return max_profit
        
    def maxSubArray(self, nums):
        """
        @53 最大子序和 ???
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        max_sum = nums[0]
        for i in range(1, n):
            if nums[i - 1] > 0:
                nums[i] += nums[i - 1]
            max_sum = max(nums[i], max_sum)

        return max_sum

    def preorderTraversal(self, root):
        """
        @144. 二叉树的前序遍历
        :type root: TreeNode
        :rtype: List[int]
        """
        res =[]
        def qianxu(root1,res1):
            """
            :type root1: TreeNode
            :rtype: List[int]
            """
            if root1.val is not None:
                res1.append(root1.val)
            if root1.left is not None:
                qianxu(root1.left,res1)
            if root1.right is not None:
                qianxu(root1.right,res1)
            return res1
        if root ==None:
            return res
        else:
            res = qianxu(root,res)
        return res

    def inorderTraversal(self, root):
        """
        @ 94.二叉树的中序遍历
        :type root: TreeNode
        :rtype: List[int]
        """
        stack=[]
        res =[]
        curr = root
        while curr is not None or not len(stack) == 0:
            while curr is not None:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            res.append(curr.val)
            curr = curr.right
        return res

    def postorderTraversal(self, root):
        """
        @145. 二叉树的后序遍历
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []

        def dfs(root):
            if root:
                dfs(root.left)
                dfs(root.right)
                result.append(root.val)

        dfs(root)
        return result

    def hasPathSum(self, root, sum):
        """
        @112. 路径总和
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if not root:
            return False

        sum -= root.val
        if not root.left and not root.right:  # if reach a leaf
            return sum == 0
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)

    def maximum69Number (self, num):
        """
        :type num: int
        :rtype: int
        """
        max = num
        res =str(num)
        if res.find('6') >=0:
            res=res.replace('6','9',1)
            max = int(res)
        return max

    def printVertically(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        num = s.split(' ')
        res=[]
        maxlen = 0
        for i in num:
            if len(i)>maxlen:
                maxlen = len(i)
        for j in range(maxlen):
            s=''
            for x in num:
                if j < len(x):
                    s += x[j]
                else:s +=' '
            res.append(s.rstrip())
        return res

    def removeLeafNodes(self, root, target):
        """
        :type root: TreeNode
        :type target: int
        :rtype: TreeNode
        """
        def dfs(root):
            if root:
                if root.left is not None:
                    dfs(root.left)
                if root.right is not None:
                    dfs(root.right)
            if root.left is not None:
                if root.left.left is None and root.left.right is None:
                    if root.left.val == target:
                        root.left = None
            if root.right is not None:
                if root.right.left is None and root.right.right is None:
                    if root.right.val == target:
                        root.right = None
        dfs(root)
        if root.left is None and root.right is None and root.val==target:
            return None
        return root

    def longestPalindrome(self, s):
        """
        @5. 最长回文子串
        :type s: str
        :rtype: str
        """
        size = len(s)
        if size < 2:
            return s

        dp = [[False for _ in range(size)] for _ in range(size)]

        max_len = 1
        start = 0

        for i in range(size):
            dp[i][i] = True

        for j in range(1, size):
            for i in range(0, j):
                if s[i] == s[j]:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = False

                if dp[i][j]:
                    cur_len = j - i + 1
                    if cur_len > max_len:
                        max_len = cur_len
                        start = i
        return s[start:start + max_len]

    def Zconvert(self,s,numRows):
        if numRows < 2: return s
        res = ["" for _ in range(numRows)]
        i, flag = 0, -1
        for c in s:
            res[i] += c
            if i == 0 or i == numRows - 1: flag = -flag
            i += flag
        return "".join(res)

    def maxArea(self, height):
        """
        @11. 盛最多水的容器
        :type height: List[int]
        :rtype: int
        """
        i, j, res = 0, len(height) - 1, 0
        while i < j:
            if height[i] < height[j]:
                res = max(res, height[i] * (j - i))
                i += 1
            else:
                res = max(res, height[j] * (j - i))
                j -= 1
        return res
            
    def searchInsert(self, nums, target):
        """
        @35. 搜索插入位置
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if target in nums:
            return nums.index(target)
        else:
            nums.append(target)
            nums.sort()
            return nums.index(target)
    def lengthOfLastWord(self, s):
        """
        @58. 最后一个单词的长度
        :type s: str
        :rtype: int
        """
        if len(s.replace(' ',''))==0:
            return 0
        if s.find(' ')>-1:
            a = s.split(' ')
            if a[-1]=='':
                return Solution.lengthOfLastWord(self,s[0:-1])
            else:
                return len(a[-1])
        else:
            return len(s)

    def mySqrt(self, x):
        """
        @69. x 的平方根
        :type x: int
        :rtype: int
        """
        if x < 2:
            return x

        left = int(e ** (0.5 * log(x)))
        right = left + 1
        return left if right * right > x else right

    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        current = head
        while current is not None and current.next is not None:
            if current.next.val == current.val:
                current.next = current.next.next
            else:
                current = current.next
        return head

    def merge(self, A, m, B, n):
        """
        :type A: List[int]
        :type m: int
        :type B: List[int]
        :type n: int
        :rtype: None Do not return anything, modify A in-place instead.
        """
        k =0
        while n-k >0:
            A[m+k] = B[k]
            k+=1
        A.sort()
        return A

    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # If the list has no node or has only one node left.
        if not head or not head.next:
            return head

        # Nodes to be swapped
        first_node = head
        second_node = head.next

        # Swapping
        first_node.next  = self.swapPairs(second_node.next)
        second_node.next = first_node

        # Now the head is the second node
        return second_node

    def orangesRotting(self, grid):
        """
        @994
        :type grid: List[List[int]]
        :rtype: int
        """
        M = len(grid)
        N = len(grid[0])
        queue = []

        count = 0  # count 表示新鲜橘子的数量
        for r in range(M):
            for c in range(N):
                if grid[r][c] == 1:
                    count += 1
                elif grid[r][c] == 2:
                    queue.append((r, c))

        round = 0  # round 表示腐烂的轮数，或者分钟数
        while count > 0 and len(queue) > 0:
            round += 1
            n = len(queue)
            for i in range(n):
                r, c = queue.pop(0)
                if r - 1 >= 0 and grid[r - 1][c] == 1:
                    grid[r - 1][c] = 2
                    count -= 1
                    queue.append((r - 1, c))
                if r + 1 < M and grid[r + 1][c] == 1:
                    grid[r + 1][c] = 2
                    count -= 1
                    queue.append((r + 1, c))
                if c - 1 >= 0 and grid[r][c - 1] == 1:
                    grid[r][c - 1] = 2
                    count -= 1
                    queue.append((r, c - 1))
                if c + 1 < N and grid[r][c + 1] == 1:
                    grid[r][c + 1] = 2
                    count -= 1
                    queue.append((r, c + 1))

        if count > 0:
            return -1
        else:
            return round

    def distributeCandies(self, candies, num_people):
        """
        @1103. 分糖果 II
        :type candies: int
        :type num_people: int
        :rtype: List[int]
        """
        ans = [0] * num_people
        i = 0
        while candies != 0:
            ans[i % num_people] += min(i + 1, candies)
            candies -= min(i + 1, candies)
            i += 1
        return ans

    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.ans = 1
        def depth(node):
            # 访问到空节点了，返回0
            if not node: return 0
            # 左儿子为根的子树的深度
            L = depth(node.left)
            # 右儿子为根的子树的深度
            R = depth(node.right)
            # 计算d_node即L+R+1 并更新ans
            self.ans = max(self.ans, L+R+1)
            # 返回该节点为根的子树的深度
            return max(L, R) + 1

        depth(root)
        return self.ans - 1

    def fib(self, N):
        """
        :type N: int
        :rtype: int
        """
        catch ={}
        def recur_fib(N):
            if N in catch:
                return catch[N]

            if N < 2:
                result = N
            else:
                result = recur_fib(N - 1) + recur_fib(N - 2)

            # put result in cache for later reference.
            catch[N] = result
            return result

        return recur_fib(N)

    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        # 超出时间限制
        # res = 1
        # if n == 0:
        #     return res
        # elif  n>0:
        #     while n >0:
        #         res= x*res
        #         n = n-1
        #     return res
        # else:
        #     n = n * -1
        #     while n >0:
        #         res= x*res
        #         n = n-1
        #     return 1 % res
        def __help(a, b):
            if b == 1:
                return a
            elif b & 1:
                return a * __help(a, b - 1)
            elif not b & 1:
                return __help(a * a, b // 2)

        if n > 0:
            return __help(x, n)
        elif n == 0:
            return 1
        else:
            return __help(1 / x, -n)

    def canThreePartsEqualSum(self, A):
        """
        @1013. 将数组分成和相等的三个部分
        :type A: List[int]
        :rtype: bool
        """
        # 超时
        # if sum(A)%3!=0:
        #     return False
        # he = sum(A)/3
        # i,j =1,len(A)-1
        # while i<j:
        #     if sum(A[:i]) != he:
        #         i = i + 1
        #     if sum(A[j:]) != he:
        #         j = j - 1
        #     if sum(A[:i]) ==sum(A[i:j]) ==sum(A[j:]) ==he:
        #         if i==j:
        #             continue
        #         return True
        # return False
        s = sum(A)
        if s % 3 != 0:
            return False
        target = s // 3
        n, i, cur = len(A), 0, 0
        while i < n:
            cur += A[i]
            if cur == target:
                break
            i += 1
        if cur != target:
            return False
        j = i + 1
        while j + 1 < n:  # 需要满足最后一个数组非空
            cur += A[j]
            if cur == target * 2:
                return True
            j += 1
        return False

    def gcdOfStrings(self, str1, str2):
        """
        :type str1: str
        :type str2: str
        :rtype: str
        """
        candidate_len = math.gcd(len(str1), len(str2))
        candidate = str1[: candidate_len]
        if str1 + str2 == str2 + str1:
            return candidate
        return ''

    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        dp = []
        for i in range(len(nums)):
            dp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def compressString(self, S):
        """
        :type S: str
        :rtype: str
        """
        res = ''
        if len(S)<1:return ''
        chang,k,d = len(S),1,0
        while chang-d-1>0:

            if d+1<chang and S[d] == S[d+1]:
                k+=1
            else:
                res+=S[d]+str(k)
                k=1
            d+=1
        res+= S[d] + str(k)
        if len(res)>=chang:
            return S
        else:
            return res

    def kthGrammar(self, N, K):
        """
        @779. 第K个语法符号
        :type N: int
        :type K: int
        :rtype: int
        """
        # 超时
        # s='0'
        # res=''
        # while N>0:
        #     for i in s:
        #         if i=='0':
        #             res+='01'
        #         elif i=='1':
        #             res+='10'
        #     s=res
        #     res=''
        #     N-=1
        # return s[K-1]
        if N == 1: return 0
        return (1 - K%2) ^ self.kthGrammar(N-1, (K+1)/2)


    def countCharacters(self, words, chars):
        """
        :type words: List[str]
        :type chars: str
        :rtype: int
        """
        chars_cnt = collections.Counter(chars)
        ans = 0
        for word in words:
            word_cnt = collections.Counter(word)
            for c in word_cnt:
                if chars_cnt[c] < word_cnt[c]:
                    break
            else:
                ans += len(word)
        return ans


    def longestPalindrome(self, s):
        """
        @409. 最长回文串
        :type s: str
        :rtype: int
        """
        ans = 0
        count = collections.Counter(s)
        for v in count.values():
            ans += v // 2 * 2
            if ans % 2 == 0 and v % 2 == 1:
                ans += 1
        return ans

    def canMeasureWater(self, x, y, z):
        """
        :type x: int
        :type y: int
        :type z: int
        :rtype: bool
        """
        if x + y < z:
            return False
        if x == 0 or y == 0:
            return z == 0 or x + y == z
        return z % math.gcd(x, y) == 0

    def numRookCaptures(self, board):
        """
        :type board: List[List[str]]
        :rtype: int
        """
        x,y,n=0,0,0
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j]=="R":
                    x=i
                    y=j
        for d in range(1,x):
            if board[x-d][y]=="p" :
                n+=1
                break
            if board[x-d][y]=="B":
               break
        for d in range(1,8-x):
            if board[x+d][y]=="p":
                n+=1
                break
            if board[x+d][y]=="B":
                break
        for d in range(1,y):
            if board[x][y-d]=="p":
                n+=1
                break
            if board[x][y-d]=="B":
                break
        for d in range(1,8-y):
            if board[x][y+d]=="p":
                n+=1
                break
            if board[x][y+d]=="B":
                break
        return n

    def coinChange(self, coins, amount):
        """
        @322. 零钱兑换
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # dp = [float('inf')] * (amount + 1)
        # dp[0] = 0
        #
        # for coin in coins:
        #     for x in range(coin, amount + 1):
        #         dp[x] = min(dp[x], dp[x - coin] + 1)
        # return dp[amount] if dp[amount] != float('inf') else -1

        # 普通动态规划解法:当前的目标金额是 n，至少需要 dp(n) 个硬币凑出该金额确定「选择」并择优，
        # 也就是对于每个状态，可以做出什么选择改变当前状态。具体到这个问题，无论当目标金额是多少，选择就是从面额列表 coins 中选择一个硬币，然后目标金额就会减少：
        # def dp(n):
        #     if n==0: return 0
        #     if n<0: return -1
        #     res = float('INF')
        #     for coin in coins:
        #         subproblem = dp(n - coin)
        #         if subproblem == -1: continue
        #         res = min(res, 1 + subproblem)
        #     return res if res != float('INF') else -1
        #
        # return dp(amount)

        # 备忘录
        memo = dict()
        def dp(n):
            # 查备忘录，避免重复计算
            if n in memo: return memo[n]

            if n == 0: return 0
            if n < 0: return -1
            res = float('INF')
            for coin in coins:
                subproblem = dp(n - coin)
                if subproblem == -1: continue
                res = min(res, 1 + subproblem)

            # 记入备忘录
            memo[n] = res if res != float('INF') else -1
            return memo[n]

        return dp(amount)

    def hasGroupsSizeX(self, deck):
        """
        @914. 卡牌分组
        :type deck: List[int]
        :rtype: bool
        """
        # dic = {}
        # for d in deck:
        #     if d in dic.keys():
        #         continue
        #     else:dic[d] = deck.count(d)
        # b = set(dic.values())
        # if len(b)==1 and dic.values()[0]>=2:
        #     return True
        # elif len(b)>1:
        #     minx = min(b)
        #     for x in b:
        #         if x%minx==0:
        #             continue
        #         else:
        #             return False
        #     return True
        # else:
        #     return False

        # 思路:寻找最大公约数 》=2的
        # from fractions import gcd
        # vals = collections.Counter(deck).values()
        # return reduce(gcd, vals) >= 2

    def createTargetArray(self, nums, index):
        """
        @1389. 按既定顺序创建目标数组
        :type nums: List[int]
        :type index: List[int]
        :rtype: List[int]
        """
        ret = []
        for i in range(len(nums)):
            ret.insert(index[i], nums[i])
        return ret

    def sumFourDivisors(self, nums):
        """
        @1390. 四因数
        :type nums: List[int]
        :rtype: int
        """
        # res =0
        # for num in nums:
        #     x= 1
        #     n=[]
        #     while num - x>=0:
        #         if num%x==0:
        #             n.append(x)
        #             if len(n)>4:
        #                 break
        #         x+=1
        #     if len(n)==4:
        #         res+=sum(n)
        # return res  超时
        ans = 0
        for num in nums:
            # factor_cnt: 因数的个数
            # factor_sum: 因数的和
            factor_cnt = factor_sum = 0
            i = 1
            while i * i <= num:
                if num % i == 0:
                    factor_cnt += 1
                    factor_sum += i
                    if i * i != num:   # 判断 i 和 num/i 是否相等，若不相等才能将 num/i 看成新的因数
                        factor_cnt += 1
                        factor_sum += num // i
                i += 1
            if factor_cnt == 4:
                ans += factor_sum
        return ans





