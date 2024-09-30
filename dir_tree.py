import os

def generate_dir_tree(path, max_depth=None, current_depth=0, prefix='', excluded_dirs=None):
    """
    生成指定路径的目录树，最多遍历到指定深度，并忽略特定的目录。

    Args:
        path (str): 需要生成目录树的根目录路径。
        max_depth (int, optional): 最大遍历深度。如果为 None，则遍历所有层级。
        current_depth (int, optional): 当前递归深度（内部使用）。
        prefix (str, optional): 当前行的前缀（内部使用）。
        excluded_dirs (list, optional): 需要忽略的目录名称列表。

    Returns:
        str: 目录树的字符串表示。
    """
    if excluded_dirs is None:
        excluded_dirs = ['.git', '__pycache__']
        
    tree = ''

    # 达到最大深度时停止递归
    if max_depth is not None and current_depth > max_depth:
        return tree

    # 在第一层添加根目录名称
    if current_depth == 0:
        tree += os.path.basename(os.path.abspath(path)) + '\n'

    try:
        # 获取当前目录下的条目并排序
        entries = sorted(os.listdir(path))
    except PermissionError:
        # 处理没有权限访问的目录
        tree += prefix + "└── [权限被拒绝]\n"
        return tree
    except FileNotFoundError:
        # 处理目录不存在的情况
        tree += prefix + "└── [未找到]\n"
        return tree

    # 过滤掉需要忽略的目录
    entries = [entry for entry in entries if entry not in excluded_dirs]

    entries_count = len(entries)
    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        is_last = (i == entries_count - 1)
        connector = '└── ' if is_last else '├── '
        tree += prefix + connector + entry + '\n'

        if os.path.isdir(full_path):
            # 准备下一层的前缀
            extension = '    ' if is_last else '│   '
            # 递归调用以处理子目录
            tree += generate_dir_tree(
                full_path,
                max_depth=max_depth,
                current_depth=current_depth + 1,
                prefix=prefix + extension,
                excluded_dirs=excluded_dirs
            )

    return tree

# Example Usage
if __name__ == "__main__":
    # 指定你想生成目录树的路径
    directory_path = "./"

    # 指定最大深度（例如，2）。设置为 None 则遍历所有层级
    max_traversal_depth = None  # 或者设置为一个整数，如 2

    # 生成目录树
    directory_tree = generate_dir_tree(directory_path, max_depth=max_traversal_depth)

    # 打印目录树
    print(directory_tree)
