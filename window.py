import math
from collections import defaultdict
from itertools import accumulate
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:

    # 2461
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        sum,ans,dict = 0,0,defaultdict(int)
        for i,num in enumerate(nums):
           sum += num
           dict[num] += 1
           if i < k - 1: continue
           if len(dict) == k: ans = max(ans,sum)
           a = nums[i - k + 1]
           sum -= a
           if dict[a] == 1: del dict[a]
           else: dict[a] -= 1
        return ans

    # 1423
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        if k == len(cardPoints): return sum(cardPoints)
        ans, sum1, win = math.inf, 0, len(cardPoints) - k
        for i, num in enumerate(cardPoints):
            sum1 += cardPoints[i]
            if i < win - 1: continue
            ans = min(ans, sum1)
            sum1 -= cardPoints[i - win + 1]
        return sum(cardPoints) - ans

    # 1052
    def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
        sum = 0
        for i,num in enumerate(customers):
            if grumpy[i] == 0: sum += num
        sum1, max1 = 0, 0
        for i,num in enumerate(customers):
            if grumpy[i] == 1: sum1 += num
            if i < minutes -1 : continue
            max1 = max(max1,sum1)
            if grumpy[i - minutes + 1] == 1: sum1 -= customers[i - minutes + 1]
        return sum + max1

    # 1652
    # def decrypt(self, code: List[int], k: int) -> List[int]:
    #     ans, sum = [None] * len(code),0
    #     for i in range(len(code) + k - 1):
    #         sum += code[i % len(code)]
    #         if i < k - 1: continue
    #         j = i - k
    #         if j < 0: j += len(code)
    #         ans[j] = sum
    #         sum -= code[i - k + 1]
    #     return ans

    # 1652
    def decrypt(self,code: List[int], k: int) -> List[int]:
        if k == 0: return [0] * len(code)
        sums = list(accumulate(code + code, initial=0))
        ans = [0] * len(code)
        for i in range(len(code)):
            if k > 0: ans[i] = sums[i + k + 1] - sums[i + 1]
            else: ans[i] = sums[i + len(code)] - sums[i + len(code) + k]
        return ans

    # 3427
    def subarraySum(self, nums: List[int]) -> int:
        sums = list(accumulate(nums,initial=0))
        ans = 0
        for i,num in enumerate(nums):
            idx = max(0,i - num)
            ans += sums[i + 1] - sums[idx]
        return ans

    # 496
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        hash = defaultdict(int)
        ans = [-1] * len(nums1)
        for i,num in enumerate(nums1):
            hash[num] = i
        stack = []
        for i,num in enumerate(nums2[::-1]):
            while len(stack) > 0 and stack[-1] <= num:
                stack.pop()
            if num in hash and len(stack) > 0: ans[hash[num]] = stack[-1]
            stack.append(num)
        return ans

    # 503
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        ans = [-1] * len(nums)
        stack = []
        for i,num in enumerate(nums[::-1] + nums[::-1]):
            if len(stack) > 0 and stack[-1] <= num:
                stack.pop()
            if len(stack) > 0: ans[i % len(nums)] = stack[-1]
            stack.append(num)
        return ans

    # 965
    def isUnivalTree(self, root: Optional[TreeNode]) -> bool:
        if not root: return True
        if self.isUnivalTree(root.left) and self.isUnivalTree(root.right):
            if not root.left and not root.right:
                return True
            if (root.left and root.val == root.left.val) and (root.right and root.val == root.right.val):
                return True
            if (root.left and root.val == root.left.val) and not root.right:
                return True
            if (root.right and root.val == root.right.val) and not root.left:
                return True
        return False

    def isUnivalTre1e(self, root: Optional[TreeNode]) -> bool:
        val = -1
        def dfs(root):
            nonlocal val
            if val == -1: val = root.val
            if not root: return True
            if val != root.val: return False
            return dfs(root.left) and dfs(root.right)
        return dfs(root)

    # 100
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q: return True
        if p and not q: return False
        if not p and q: return False
        if p.val != q.val: return False
        else: return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)

    # 101
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root.left and not root.right: return True
        return self.isSameTree(root.left,root.right)

    # 951
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        if not root1 and root2 or root1 and not root2: return False
        if not root1 and not root2: return True
        if root1.val != root2.val: return False
        if root1.left.val == root2.right.val and root1.right.val == root1.left.val:
            temp = root1.left
            root1.left = root1.right
            root1.right = temp
            return self.flipEquiv(root1.left,root2.left) and self.flipEquiv(root1.right,root2.right)
        if root1.left.val == root2.left.val and root1.right.val == root2.right.val:
            return self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right)

    # 2296
    def divisorSubstrings(self, num: int, k: int) -> int:
        ans,s = 0,str(num)
        for i,ch in enumerate(s):
            if i < k - 1: continue
            a = int(s[i - k + 1:i])
            if a != 0: ans += 1 if(num % a == 0) else 0
        return ans

    # 1379
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        ans: TreeNode
        def dfs1379(root1,root2) -> None:
            nonlocal ans
            if not root1: return
            if root1.val == target.val:
                ans = root2
                return
            dfs1379(root1.left,root2.left)
            dfs1379(root1.right,root2.right)
        dfs1379(original,cloned)
        return ans

    # 110
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        flag: bool = True
        def dfs(root) -> int:
            nonlocal flag
            if not root: return 0
            a = dfs(root.left)
            b = dfs(root.right)
            if abs(a - b) > 1: flag = False
            return max(a,b) + 1
        dfs(root)
        return flag

    # 508
    def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:
        mx, ans, hash = 0, [], defaultdict(int)
        def dfs(root) -> int:
            nonlocal mx
            if not root: return 0
            val = root.val + dfs(root.left) + dfs(root.right)
            hash[val] += 1
            mx = max(mx,hash[val])
            return val
        dfs(root)
        for key in hash:
            if hash[key] == mx:
                ans.append(key)
        return ans
# 303
class NumArray:

    sums: List[int]

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.sums = list(accumulate(nums,initial=0))
    def sumRange(self, left: int, right: int) -> int:
        return self.sums[right + 1] - self.sums[left]
