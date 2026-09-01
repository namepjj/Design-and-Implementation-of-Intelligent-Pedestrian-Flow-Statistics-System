def max_nesting_depth(lst):
    """
    计算列表的最大嵌套深度。

    列表中的元素可能也是一个列表，返回该列表最大的嵌套深度。
    例如：
        [1, 2, 3]            -> 1
        [[1], [2, [3]]]      -> 3
        []                   -> 1（空列表本身也是一层嵌套）


    """
    if not isinstance(lst, list):
        raise TypeError("参数必须是一个列表")

    # 空列表也算一层嵌套
    if not lst:
        return 1

    # 当前列表占 1 层，再加上所有子列表中嵌套最深的那一个的深度
    return 1 + max(max_nesting_depth(item) if isinstance(item, list) else 0
                   for item in lst)
