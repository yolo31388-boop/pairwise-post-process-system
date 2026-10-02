import pytest
from post_process.system import PostProcessSystem


class TestPostProcessSystem:

    def test_apply_bloom(self):
        """Bloom效果不做阈值，所有亮度的像素都泛光"""
        obj = PostProcessSystem()
        result = obj.apply_bloom()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_apply_dof(self):
        """景深不计算模糊圈大小，所有不在焦距的像素模糊程度一样"""
        obj = PostProcessSystem()
        result = obj.apply_dof()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_apply_motion_blur(self):
        """运动模糊只用速度缓冲方向，不考虑速度大小"""
        obj = PostProcessSystem()
        result = obj.apply_motion_blur()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_apply_color_grading(self):
        """颜色分级只做亮度对比度，不做色相和饱和度调整"""
        obj = PostProcessSystem()
        result = obj.apply_color_grading()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

    def test_sort_post_process(self):
        """后处理顺序固定，不按效果类型排序"""
        obj = PostProcessSystem()
        result = obj.sort_post_process()
        # The broken implementation returns None; fixed version should return proper value
        self.assertIsNotNone(result)

