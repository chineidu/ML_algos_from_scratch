"""This module is used to build Binary Tree from scratch."""
from typing import Any, NewType, Union

Tree = NewType("Tree", tp=Any)


class BinaryTree:
    """This is used to model a Binary Tree."""

    def __init__(self, *, node: Union[int, float]) -> None:
        self.node = node
        self.left = None
        self.right = None

    def __repr__(self) -> str:
        return (
            f"{__class__.__name__}(left={self.left}, " f"node={self.node}, " f"right={self.right})"
        )

    @staticmethod
    def display_tree(tree: Tree, space: str = "\t", level: int = 0):
        """This is used to visually represent the binary tree.
        (Copied!)
        """
        # If the tree has no value (empty)
        if tree is None:
            print(space * level + "Ф")
            return None

        # If the node is a leaf
        if tree.left is None and tree.right is None:
            print(space * level + str(tree.node))
            return None

        # If the node has children
        BinaryTree.display_tree(tree.right, space, level + 1)
        print(space * level + str(tree.node))
        BinaryTree.display_tree(tree.left, space, level + 1)

    def insert_values(self, *, value: Union[int, float]) -> None:
        """This is used to insert values into the Binary Tree.

        Note:
            If the value is less than the value at the node
            it's inserted to the left otherwise it's inserted
            to the right.
        """
        if value < self.node:
            # If value is still less than the node,
            # move to the left and insert value recursively
            if self.left:
                self.left.insert_values(value=value)
            else:
                self.left = BinaryTree(node=value)

        if value > self.node:
            # If value is still greater than the node,
            # move to the right and insert value recursively
            if self.right:
                self.right.insert_values(value=value)
            else:
                self.right = BinaryTree(node=value)
        return self

    def search_tree(self, *, value: Union[int, float]) -> str:
        """This searches the Binary Tree using BST."""
        # Base case
        if value == self.node:
            return f"{value} is present!"

        # If the value is less than the value at the node
        # traverse the left hand side of the tree until the
        # value is found otherwise the values is not present
        if value < self.node:
            if self.left:
                self.left.search_tree(value=value)
            else:
                return f"{value} is not found!!!"

        # If the value is greater than the value at the node
        # traverse the right hand side of the tree until the
        # value is found otherwise the values is not present
        if value > self.node:
            if self.right:
                self.right.search_tree(value=value)
            else:
                return f"{value} is not found!!!"


if __name__ == '__main__':
    tree = BinaryTree(node=10)
    tree.insert_values(value=8)
    tree.insert_values(value=12)
    print(tree)
    print(BinaryTree.display_tree(tree))
