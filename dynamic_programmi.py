from typing import List


class Solution:

    # 740
    def deleteAndEarn(self, nums: List[int]) -> int:
        def rob(nums1: List[int]) -> int:
            cache = [-1] * (max(nums1) + 1)
            def dp(idx) -> int:
                if idx < 0: return 0
                if cache[idx] != -1: return cache[idx]
                a = dp(idx - 1) + dp(idx - 2) + nums1[idx]
                cache[idx] = a
                return a
            return dp(nums1[len(nums) -1])
        ans = [0] * max(nums)
        for num in nums:
            ans[num] += num
        return rob(ans)