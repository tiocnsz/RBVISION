"""
启动入口 (M0 占位) - 后续 M6 填充 GUI 主循环.
"""

import sys


def main():
    print("RBVISION v0.1.0 (M0 - 项目骨架)")
    print("GUI 启动将在 M6 阶段实现")
    print()
    print("当前可用模块:")
    print("  - rbvision.core: Pose, Transform, 4x4 矩阵运算")
    print("  - rbvision.utils: config, logger")
    print()
    print("运行测试: pytest tests/ -v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
