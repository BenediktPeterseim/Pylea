"""Classes of integers under merging, named by their lowest member.

`UnionFind` holds a partition of ``0..n-1``.  `union` merges two classes and `find` names
the class of an integer by its lowest member, so where the integers are wires numbered
caller first, a class holding a wire of the caller is named by one.

    >>> uf = UnionFind(4)
    >>> uf.union(3, 1)
    >>> uf.grow(1)
    >>> uf.union(4, 3)
    >>> [uf.find(x) for x in range(5)]
    [0, 1, 2, 1, 1]
"""
from __future__ import annotations


class UnionFind:
    """A partition of ``0..n-1`` into classes, each named by its lowest member."""

    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def grow(self, n: int) -> None:
        """Nothing, `n` integers now appended, each in a class of its own."""
        start = len(self.parent)
        self.parent.extend(range(start, start + n))

    def find(self, x: int) -> int:
        """The lowest member of the class of `x`."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: int, b: int) -> None:
        """Nothing, the classes of `a` and `b` now being one."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)
