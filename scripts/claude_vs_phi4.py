"""Head-to-head: Claude Opus 4.6 (no RAG) vs phi4+RAG on blind challenges."""
import json
import subprocess
import sys

# Load challenges
with open("C:/Users/jerem/AgentTeam/war_rooms/JCoder/mailbox/opus_to_jcoder.jsonl") as f:
    for line in f:
        data = json.loads(line)
        if data.get("type") == "blind_challenges":
            challenges = data["challenges"]
            break

# Claude's solutions — pure reasoning, no RAG
claude_solutions = {}

claude_solutions["OPUS-001"] = """
from collections import Counter
def min_window(s, t):
    if not t or not s: return ''
    need = Counter(t)
    missing = len(t)
    left = start = 0
    best = float('inf')
    for right, c in enumerate(s):
        if need[c] > 0: missing -= 1
        need[c] -= 1
        while missing == 0:
            if right - left + 1 < best:
                best = right - left + 1
                start = left
            need[s[left]] += 1
            if need[s[left]] > 0: missing += 1
            left += 1
    return '' if best == float('inf') else s[start:start+best]
"""

claude_solutions["OPUS-002"] = """
def max_profit(prices):
    if len(prices) <= 1: return 0
    hold, sold, rest = -prices[0], 0, 0
    for p in prices[1:]:
        ph, ps, pr = hold, sold, rest
        hold = max(ph, pr - p)
        sold = ph + p
        rest = max(pr, ps)
    return max(sold, rest)
"""

claude_solutions["OPUS-003"] = """
def num_distinct(s, t):
    m, n = len(s), len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = 1
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j]
            if s[i-1] == t[j-1]: dp[i][j] += dp[i-1][j-1]
    return dp[m][n]
"""

claude_solutions["OPUS-004"] = """
def longest_path(matrix):
    if not matrix or not matrix[0]: return 0
    m, n = len(matrix), len(matrix[0])
    memo = {}
    def dfs(i, j):
        if (i,j) in memo: return memo[(i,j)]
        best = 1
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0<=ni<m and 0<=nj<n and matrix[ni][nj] > matrix[i][j]:
                best = max(best, 1 + dfs(ni, nj))
        memo[(i,j)] = best
        return best
    return max(dfs(i,j) for i in range(m) for j in range(n))
"""

claude_solutions["OPUS-005"] = """
def find_median(nums1, nums2):
    if len(nums1) > len(nums2): nums1, nums2 = nums2, nums1
    m, n = len(nums1), len(nums2)
    if m == 0:
        mid = n
        if mid % 2: return float(nums2[mid//2])
        return (nums2[mid//2-1] + nums2[mid//2]) / 2.0
    lo, hi = 0, m
    while lo <= hi:
        i = (lo+hi)//2
        j = (m+n+1)//2 - i
        l1 = nums1[i-1] if i>0 else float('-inf')
        r1 = nums1[i] if i<m else float('inf')
        l2 = nums2[j-1] if j>0 else float('-inf')
        r2 = nums2[j] if j<n else float('inf')
        if l1<=r2 and l2<=r1:
            if (m+n)%2: return float(max(l1,l2))
            return (max(l1,l2)+min(r1,r2))/2.0
        elif l1>r2: hi=i-1
        else: lo=i+1
"""

claude_solutions["OPUS-006"] = """
from collections import defaultdict, deque
def alien_order(words):
    graph = defaultdict(set)
    indegree = {c: 0 for w in words for c in w}
    for i in range(len(words)-1):
        w1, w2 = words[i], words[i+1]
        if len(w1) > len(w2) and w1[:len(w2)] == w2: return ''
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    indegree[c2] += 1
                break
    q = deque(c for c in indegree if indegree[c]==0)
    result = []
    while q:
        c = q.popleft()
        result.append(c)
        for n in graph[c]:
            indegree[n] -= 1
            if indegree[n] == 0: q.append(n)
    return ''.join(result) if len(result)==len(indegree) else ''
"""

claude_solutions["OPUS-007"] = """
def max_coins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0]*n for _ in range(n)]
    for length in range(2, n):
        for left in range(0, n-length):
            right = left + length
            for k in range(left+1, right):
                dp[left][right] = max(dp[left][right],
                    dp[left][k] + dp[k][right] + nums[left]*nums[k]*nums[right])
    return dp[0][n-1]
"""

claude_solutions["OPUS-008"] = """
import heapq
def trap_rain_3d(heightMap):
    if not heightMap or not heightMap[0]: return 0
    m, n = len(heightMap), len(heightMap[0])
    visited = [[False]*n for _ in range(m)]
    heap = []
    for i in range(m):
        for j in range(n):
            if i==0 or i==m-1 or j==0 or j==n-1:
                heapq.heappush(heap, (heightMap[i][j], i, j))
                visited[i][j] = True
    water = 0
    while heap:
        h, i, j = heapq.heappop(heap)
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i+di, j+dj
            if 0<=ni<m and 0<=nj<n and not visited[ni][nj]:
                visited[ni][nj] = True
                water += max(0, h - heightMap[ni][nj])
                heapq.heappush(heap, (max(h, heightMap[ni][nj]), ni, nj))
    return water
"""

claude_solutions["OPUS-009"] = """
def shortest_superstring(words):
    n = len(words)
    overlap = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(min(len(words[i]),len(words[j])), 0, -1):
                    if words[i].endswith(words[j][:k]):
                        overlap[i][j] = k
                        break
    dp = [[''] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = words[i]
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask & (1 << last)): continue
            if not dp[mask][last]: continue
            for nxt in range(n):
                if mask & (1 << nxt): continue
                new_mask = mask | (1 << nxt)
                candidate = dp[mask][last] + words[nxt][overlap[last][nxt]:]
                if not dp[new_mask][nxt] or len(candidate) < len(dp[new_mask][nxt]):
                    dp[new_mask][nxt] = candidate
    full = (1 << n) - 1
    return min((dp[full][i] for i in range(n) if dp[full][i]), key=len, default='')
"""

claude_solutions["OPUS-010"] = """
def max_sum_bst(root):
    result = [0]
    def dfs(node):
        if not node: return True, float('inf'), float('-inf'), 0
        lb, lmin, lmax, lsum = dfs(node.left)
        rb, rmin, rmax, rsum = dfs(node.right)
        if lb and rb and lmax < node.val < rmin:
            s = lsum + rsum + node.val
            result[0] = max(result[0], s)
            return True, min(lmin, node.val), max(rmax, node.val), s
        return False, 0, 0, 0
    dfs(root)
    return result[0]
"""

print("=" * 60)
print("HEAD TO HEAD: Claude Opus 4.6 (no RAG) vs phi4 14B (+RAG)")
print("Same 10 blind challenges from independent orchestrator")
print("=" * 60)
print()

claude_passed = 0
for ch in challenges:
    code = claude_solutions.get(ch["id"], "")
    full_code = code.strip() + "\n\n" + ch["test_code"]
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True, text=True, timeout=15,
        )
        passed = result.returncode == 0 and "PASSED" in result.stdout
        error = result.stderr[:80] if not passed else ""
    except:
        passed = False
        error = "Timeout"

    status = "PASS" if passed else "FAIL"
    print(f"  {ch['id']}: {ch['title']:42s} {status}")
    if error:
        print(f"       {error[:80]}")
    if passed:
        claude_passed += 1

print()
print("=" * 60)
print(f"  Claude Opus 4.6 (no RAG):   {claude_passed}/10 ({claude_passed/10:.0%})")
print(f"  phi4 14B + RAG + lessons:   9/10 (90%)")
print(f"  Delta:                      {9-claude_passed:+d}")
print("=" * 60)
