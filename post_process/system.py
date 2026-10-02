"""
pairwise-post-process-system - PostProcessSystem

A pure-Python post-processing pipeline:

- Bloom: thresholded bright-pass + gaussian blur halo + clamped intensity.
- Depth of field: per-pixel circle of confusion from depth, disc (bokeh) blur.
- Motion blur: speed-scaled sample count, path accumulation (trail), static
  pixels are excluded.
- Color grading: brightness/contrast + hue/saturation + white balance,
  applied through a cached 3D LUT instead of per-pixel math.
- Pipeline: canonical effect ordering (bloom before tone mapping), skippable
  effects, and ping-pong render-target reuse.

Images are 2D lists of [r, g, b] lists with float channels in [0, 1].
"""

import colorsys
import math


def _clamp(value, lo=0.0, hi=1.0):
    return lo if value < lo else hi if value > hi else value


def _luminance(pixel):
    return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]


class PostProcessSystem:

    MAX_BLOOM_INTENSITY = 2.0
    STATIC_VELOCITY_EPSILON = 1e-3

    # Canonical pipeline order. Bloom must run before tone mapping so the
    # added glow is tone-mapped with the scene instead of blowing out after.
    EFFECT_ORDER = (
        "motion_blur",
        "dof",
        "bloom",
        "tone_mapping",
        "color_grading",
    )

    def __init__(self):
        self.state = {
            "lut_cache": {},
            "motion_history": None,
            "render_target_pool": [],
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _normalize_image(self, image, width=8, height=8):
        if image is None:
            return [
                [[x / max(width - 1, 1), y / max(height - 1, 1), 0.5]
                 for x in range(width)]
                for y in range(height)
            ]
        return [[list(pixel) for pixel in row] for row in image]

    def _normalize_field(self, field, width, height, default):
        if field is None:
            return [[list(default) for _ in range(width)] for _ in range(height)]
        return field

    def _new_render_target(self, width, height):
        pool = self.state["render_target_pool"]
        for target in pool:
            if len(target) == height and len(target[0]) == width:
                pool.remove(target)
                return target
        return [[[0.0, 0.0, 0.0] for _ in range(width)] for _ in range(height)]

    def _release_render_target(self, target):
        self.state["render_target_pool"].append(target)

    @staticmethod
    def _gaussian_kernel(radius, sigma=None):
        radius = max(1, int(radius))
        sigma = sigma or max(radius / 2.0, 0.5)
        kernel = [math.exp(-(i * i) / (2.0 * sigma * sigma))
                  for i in range(-radius, radius + 1)]
        total = sum(kernel)
        return [w / total for w in kernel]

    def _gaussian_blur(self, image, radius):
        kernel = self._gaussian_kernel(radius)
        size = len(kernel)
        offset = size // 2
        height = len(image)
        width = len(image[0])

        horizontal = [[[0.0, 0.0, 0.0] for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                acc = [0.0, 0.0, 0.0]
                for k in range(size):
                    sx = min(max(x + k - offset, 0), width - 1)
                    w = kernel[k]
                    src = image[y][sx]
                    acc[0] += src[0] * w
                    acc[1] += src[1] * w
                    acc[2] += src[2] * w
                horizontal[y][x] = acc

        blurred = [[[0.0, 0.0, 0.0] for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                acc = [0.0, 0.0, 0.0]
                for k in range(size):
                    sy = min(max(y + k - offset, 0), height - 1)
                    w = kernel[k]
                    src = horizontal[sy][x]
                    acc[0] += src[0] * w
                    acc[1] += src[1] * w
                    acc[2] += src[2] * w
                blurred[y][x] = acc
        return blurred

    # ------------------------------------------------------------------
    # bloom
    # ------------------------------------------------------------------

    def apply_bloom(self, image=None, threshold=0.8, intensity=1.0, radius=4):
        """Thresholded bright-pass, gaussian-blurred halo, clamped intensity."""
        img = self._normalize_image(image)
        height = len(img)
        width = len(img[0])
        intensity = _clamp(float(intensity), 0.0, self.MAX_BLOOM_INTENSITY)

        bright = [[[0.0, 0.0, 0.0] for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                pixel = img[y][x]
                lum = _luminance(pixel)
                if lum >= threshold:
                    bright[y][x] = list(pixel)

        halo = self._gaussian_blur(bright, radius)

        result = self._new_render_target(width, height)
        for y in range(height):
            for x in range(width):
                src = img[y][x]
                glow = halo[y][x]
                result[y][x] = [
                    _clamp(src[0] + glow[0] * intensity),
                    _clamp(src[1] + glow[1] * intensity),
                    _clamp(src[2] + glow[2] * intensity),
                ]
        return result

    # ------------------------------------------------------------------
    # depth of field
    # ------------------------------------------------------------------

    def _circle_of_confusion(self, depth_value, focal_distance, aperture, max_coc):
        depth_value = max(float(depth_value), 1e-6)
        coc = aperture * abs(depth_value - focal_distance) / depth_value
        return min(coc, max_coc)

    @staticmethod
    def _bokeh_offsets(samples=16):
        offsets = []
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(samples):
            radius = math.sqrt((i + 0.5) / samples)
            angle = i * golden_angle
            offsets.append((radius * math.cos(angle), radius * math.sin(angle)))
        return offsets

    def apply_dof(self, image=None, depth=None, focal_distance=5.0,
                  aperture=2.0, max_coc=6.0, bokeh_samples=16):
        """Per-pixel circle of confusion from the depth buffer, disc bokeh."""
        img = self._normalize_image(image)
        height = len(img)
        width = len(img[0])
        depth = self._normalize_field(depth, width, height, [focal_distance])

        coc_map = [
            [self._circle_of_confusion(depth[y][x][0] if isinstance(depth[y][x], (list, tuple)) else depth[y][x],
                                       focal_distance, aperture, max_coc)
             for x in range(width)]
            for y in range(height)
        ]

        offsets = self._bokeh_offsets(bokeh_samples)
        result = self._new_render_target(width, height)
        for y in range(height):
            for x in range(width):
                coc = coc_map[y][x]
                if coc <= 1e-3:
                    result[y][x] = list(img[y][x])
                    continue
                acc = [0.0, 0.0, 0.0]
                weight = 0.0
                for ox, oy in offsets:
                    sx = min(max(int(round(x + ox * coc)), 0), width - 1)
                    sy = min(max(int(round(y + oy * coc)), 0), height - 1)
                    src = img[sy][sx]
                    acc[0] += src[0]
                    acc[1] += src[1]
                    acc[2] += src[2]
                    weight += 1.0
                result[y][x] = [acc[0] / weight, acc[1] / weight, acc[2] / weight]
        return result

    # ------------------------------------------------------------------
    # motion blur
    # ------------------------------------------------------------------

    def apply_motion_blur(self, image=None, velocity=None, shutter=1.0,
                          max_samples=16, temporal_feedback=0.5):
        """Speed-scaled taps accumulated along the velocity vector.

        Static pixels (speed below epsilon) are passed through untouched.
        A temporal history buffer accumulates previous frames for trails.
        """
        img = self._normalize_image(image)
        height = len(img)
        width = len(img[0])
        velocity = self._normalize_field(velocity, width, height, [0.0, 0.0])

        blurred = self._new_render_target(width, height)
        for y in range(height):
            for x in range(width):
                vx, vy = velocity[y][x][0], velocity[y][x][1]
                speed = math.hypot(vx, vy)
                if speed < self.STATIC_VELOCITY_EPSILON:
                    blurred[y][x] = list(img[y][x])
                    continue
                samples = max(2, min(int(speed * shutter * 4) + 1, max_samples))
                acc = [0.0, 0.0, 0.0]
                weight_sum = 0.0
                for i in range(samples):
                    t = i / (samples - 1)
                    w = 1.0 - 0.5 * t
                    sx = min(max(int(round(x - vx * t)), 0), width - 1)
                    sy = min(max(int(round(y - vy * t)), 0), height - 1)
                    src = img[sy][sx]
                    acc[0] += src[0] * w
                    acc[1] += src[1] * w
                    acc[2] += src[2] * w
                    weight_sum += w
                blurred[y][x] = [acc[0] / weight_sum,
                                 acc[1] / weight_sum,
                                 acc[2] / weight_sum]

        history = self.state["motion_history"]
        if history is not None and len(history) == height and len(history[0]) == width:
            feedback = _clamp(temporal_feedback)
            for y in range(height):
                for x in range(width):
                    prev = history[y][x]
                    cur = blurred[y][x]
                    blurred[y][x] = [
                        cur[0] * (1.0 - feedback) + prev[0] * feedback,
                        cur[1] * (1.0 - feedback) + prev[1] * feedback,
                        cur[2] * (1.0 - feedback) + prev[2] * feedback,
                    ]
        self.state["motion_history"] = [[list(p) for p in row] for row in blurred]
        return blurred

    # ------------------------------------------------------------------
    # color grading
    # ------------------------------------------------------------------

    def _grade_pixel(self, r, g, b, brightness, contrast, saturation,
                     hue_shift, temperature, tint):
        r += temperature
        b -= temperature
        g += tint

        r = (r - 0.5) * contrast + 0.5 + brightness
        g = (g - 0.5) * contrast + 0.5 + brightness
        b = (b - 0.5) * contrast + 0.5 + brightness

        h, s, v = colorsys.rgb_to_hsv(_clamp(r), _clamp(g), _clamp(b))
        h = (h + hue_shift) % 1.0
        s = _clamp(s * saturation)
        return colorsys.hsv_to_rgb(h, s, v)

    def build_lut(self, size=16, brightness=0.0, contrast=1.0, saturation=1.0,
                  hue_shift=0.0, temperature=0.0, tint=0.0):
        """Precompute a 3D LUT for the grading transform (cached)."""
        key = (size, brightness, contrast, saturation, hue_shift, temperature, tint)
        cache = self.state["lut_cache"]
        if key in cache:
            return cache[key]
        steps = size - 1
        lut = {}
        for ri in range(size):
            for gi in range(size):
                for bi in range(size):
                    lut[(ri, gi, bi)] = self._grade_pixel(
                        ri / steps, gi / steps, bi / steps,
                        brightness, contrast, saturation,
                        hue_shift, temperature, tint)
        cache[key] = lut
        return lut

    def apply_color_grading(self, image=None, brightness=0.0, contrast=1.0,
                            saturation=1.0, hue_shift=0.0, temperature=0.0,
                            tint=0.0, lut=None, lut_size=16):
        """Grading via a cached 3D LUT: hue, saturation and white balance."""
        img = self._normalize_image(image)
        height = len(img)
        width = len(img[0])
        if lut is None:
            lut = self.build_lut(lut_size, brightness, contrast, saturation,
                                 hue_shift, temperature, tint)
        steps = lut_size - 1

        result = self._new_render_target(width, height)
        for y in range(height):
            for x in range(width):
                pixel = img[y][x]
                key = (int(round(_clamp(pixel[0]) * steps)),
                       int(round(_clamp(pixel[1]) * steps)),
                       int(round(_clamp(pixel[2]) * steps)))
                graded = lut.get(key)
                if graded is None:
                    graded = self._grade_pixel(pixel[0], pixel[1], pixel[2],
                                               brightness, contrast, saturation,
                                               hue_shift, temperature, tint)
                result[y][x] = [_clamp(graded[0]), _clamp(graded[1]), _clamp(graded[2])]
        return result

    # ------------------------------------------------------------------
    # pipeline ordering / execution
    # ------------------------------------------------------------------

    def sort_post_process(self, effects=None, skip=None):
        """Order effects by type and drop disabled/skipped ones.

        Accepts names or dicts with "name"/"enabled" keys. Returns the
        surviving entries in canonical pipeline order (bloom runs before
        tone mapping).
        """
        if effects is None:
            effects = list(self.EFFECT_ORDER)
        skip = set(skip or ())

        rank = {name: i for i, name in enumerate(self.EFFECT_ORDER)}
        survivors = []
        for effect in effects:
            if isinstance(effect, dict):
                name = effect.get("name")
                if not effect.get("enabled", True) or name in skip:
                    continue
                survivors.append(effect)
            else:
                if effect in skip:
                    continue
                survivors.append(effect)
        survivors.sort(key=lambda e: rank.get(
            e.get("name") if isinstance(e, dict) else e, len(rank)))
        return survivors

    def run_pipeline(self, image=None, effects=None, skip=None, params=None):
        """Execute the sorted pipeline, reusing pooled render targets."""
        params = params or {}
        current = self._normalize_image(image)
        pooled = False
        for effect in self.sort_post_process(effects, skip):
            name = effect.get("name") if isinstance(effect, dict) else effect
            effect_params = params.get(name, {})
            if name == "bloom":
                nxt = self.apply_bloom(current, **effect_params)
            elif name == "dof":
                nxt = self.apply_dof(current, **effect_params)
            elif name == "motion_blur":
                nxt = self.apply_motion_blur(current, **effect_params)
            elif name == "color_grading":
                nxt = self.apply_color_grading(current, **effect_params)
            elif name == "tone_mapping":
                nxt = self.apply_tone_mapping(current, **effect_params)
            else:
                continue
            if pooled:
                self._release_render_target(current)
            current = nxt
            pooled = True
        return current

    def apply_tone_mapping(self, image=None, exposure=1.0):
        img = self._normalize_image(image)
        return [
            [[_clamp(1.0 - math.exp(-channel * exposure)) for channel in pixel]
             for pixel in row]
            for row in img
        ]
