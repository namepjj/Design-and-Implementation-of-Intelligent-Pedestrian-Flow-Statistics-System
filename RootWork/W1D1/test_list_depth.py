import os
import sys
import unittest

# 将本文件所在目录加入模块搜索路径，保证在任意工作目录下都能导入 list_depth
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from list_depth import max_nesting_depth


class TestMaxNestingDepth(unittest.TestCase):
    """max_nesting_depth 函数的单元测试"""

    def test_flat_list(self):
        """扁平列表，深度为 1"""
        self.assertEqual(max_nesting_depth([1, 2, 3]), 1)

    def test_example_nested_list(self):
        """题目示例：[[1], [2, [3]]] 的深度为 3"""
        self.assertEqual(max_nesting_depth([[1], [2, [3]]]), 3)

    def test_empty_list(self):
        """空列表本身也是一层嵌套，深度为 1"""
        self.assertEqual(max_nesting_depth([]), 1)

    def test_deeply_nested_list(self):
        """多层连续嵌套：[[[[1]]]] 的深度为 4"""
        self.assertEqual(max_nesting_depth([[[[1]]]]), 4)

    def test_mixed_depth(self):
        """不同深度的分支并存，取最大值"""
        self.assertEqual(max_nesting_depth([1, [2, [3, [4]]], 5]), 4)

    def test_nested_empty_list(self):
        """嵌套的空列表也占嵌套层级：[[], [[]]] 的深度为 3"""
        self.assertEqual(max_nesting_depth([[], [[]]]), 3)

    def test_non_list_raises_type_error(self):
        """传入非列表参数时应抛出 TypeError"""
        with self.assertRaises(TypeError):
            max_nesting_depth(123)


if __name__ == "__main__":
    unittest.main()
