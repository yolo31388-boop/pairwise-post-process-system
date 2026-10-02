"""
pairwise-post-process-system - PostProcessSystem

This module contains a deliberately broken implementation.
Fix all bugs so that tests pass.
"""


class PostProcessSystem:
    def __init__(self):
        self.state = {}

    def apply_bloom(self, *args, **kwargs):
        """BUG: Bloom效果不做阈值，所有亮度的像素都泛光"""
        # Intentionally broken implementation
        return None

    def apply_dof(self, *args, **kwargs):
        """BUG: 景深不计算模糊圈大小，所有不在焦距的像素模糊程度一样"""
        # Intentionally broken implementation
        return None

    def apply_motion_blur(self, *args, **kwargs):
        """BUG: 运动模糊只用速度缓冲方向，不考虑速度大小"""
        # Intentionally broken implementation
        return None

    def apply_color_grading(self, *args, **kwargs):
        """BUG: 颜色分级只做亮度对比度，不做色相和饱和度调整"""
        # Intentionally broken implementation
        return None

    def sort_post_process(self, *args, **kwargs):
        """BUG: 后处理顺序固定，不按效果类型排序"""
        # Intentionally broken implementation
        return None

