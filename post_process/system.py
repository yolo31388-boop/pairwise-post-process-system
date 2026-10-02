"""Post-process pipeline: bloom, depth of field, motion blur, color grading.

Images are ``height x width`` lists of ``(r, g, b)`` float tuples in the
0..1 range. Depth buffers are ``height x width`` lists of floats and
velocity buffers are ``height x width`` lists of ``(vx, vy)`` tuples in
pixels per frame.
"""

import math


def _clamp(value, lo=0.0, hi=1.0):
    return lo if value < lo else hi if value > hi else value


def _luminance(pixel):
    r, g, b = pixel
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


class PostProcessSystem:
    MAX_BLOOM_INTENSITY = 1.0
    BLOOM_THRESHOLD = 0.8
    MAX_COC_RADIUS = 8
    MAX_MOTION_SAMPLES = 16
    STATIC_SPEED_EPSILON = 1e-3
    LUT_SIZE = 16
    # Bloom must run before tonemapping or the frame blows out.
    EFFECT_ORDER = ("motion_blur", "dof", "bloom", "tonemapping", "color_grading")

    def __init__(self):
        self.state = {}
        self._render_target_pool = ["rt_a", "rt_b"]

    # ------------------------------------------------------------------
    # Default buffers so effects can run standalone.
    # ------------------------------------------------------------------
    @staticmethod
    def _default_image(width=8, height=8):
        return [[(x / (width - 1), y / (height - 1), 0.5)
                 for x in range(width)] for y in range(height)]

    @staticmethod
    def _default_depth(width, height):
        return [[(x + y) / (width + height - 2)
                 for x in range(width)] for y in range(height)]

    @staticmethod
    def _default_velocity(width, height):
        return [[(0.5, 0.0) for _ in range(width)] for _ in range(height)]

    # ------------------------------------------------------------------
    # Bloom: thresholded bright pass + gaussian blur + clamped intensity.
    # ------------------------------------------------------------------
    def apply_bloom(self, image=None, threshold=None, intensity=0.5, blur_radius=2):
        image = image if image is not None else self._default_image()
        threshold = self.BLOOM_THRESHOLD if threshold is None else threshold
        intensity = _clamp(intensity, 0.0, self.MAX_BLOOM_INTENSITY)
        bright_pass = [
            [pixel if _luminance(pixel) >= threshold else (0.0, 0.0, 0.0)
             for pixel in row]
            for row in image
        ]
        halo = self._gaussian_blur(bright_pass, blur_radius)
        return [[tuple(_clamp(c + intensity * h) for c, h in zip(pixel, halo[y][x]))
                 for x, pixel in enumerate(row)]
                for y, row in enumerate(image)]

    @staticmethod
    def _gaussian_kernel(radius):
        radius = max(1, int(radius))
        sigma = radius / 2.0
        kernel = [math.exp(-(i * i) / (2.0 * sigma * sigma))
                  for i in range(-radius, radius + 1)]
        total = sum(kernel)
        return [k / total for k in kernel]

    def _gaussian_blur(self, image, radius):
        kernel = self._gaussian_kernel(radius)
        half = (len(kernel) - 1) // 2
        height, width = len(image), len(image[0])

        def blur_pass(src, horizontal):
            out = []
            for y in range(height):
                row = []
                for x in range(width):
                    acc = [0.0, 0.0, 0.0]
                    for k, weight in enumerate(kernel):
                        offset = k - half
                        sx = min(max(x + offset, 0), width - 1) if horizontal else x
                        sy = y if horizontal else min(max(y + offset, 0), height - 1)
                        pixel = src[sy][sx]
                        acc[0] += pixel[0] * weight
                        acc[1] += pixel[1] * weight
                        acc[2] += pixel[2] * weight
                    row.append((acc[0], acc[1], acc[2]))
                out.append(row)
            return out

        return blur_pass(blur_pass(image, True), False)

    # ------------------------------------------------------------------
    # Depth of field: per-pixel circle of confusion + disk bokeh.
    # ------------------------------------------------------------------
    def apply_dof(self, image=None, depth=None, focus_distance=0.5, aperture=4.0):
        image = image if image is not None else self._default_image()
        height, width = len(image), len(image[0])
        depth = depth if depth is not None else self._default_depth(width, height)
        result = []
        for y in range(height):
            row = []
            for x in range(width):
                d = max(depth[y][x], 1e-6)
                # Circle of confusion grows with distance from the focal
                # plane, so near/far pixels blur by different amounts.
                coc = min(self.MAX_COC_RADIUS, aperture * abs(d - focus_distance) / d)
                radius = int(round(coc))
                if radius < 1:
                    row.append(image[y][x])
                else:
                    row.append(self._bokeh_average(image, x, y, radius))
            result.append(row)
        return result

    @staticmethod
    def _bokeh_average(image, cx, cy, radius):
        # Disk-shaped kernel gives circular bokeh discs, not gaussian blobs.
        height, width = len(image), len(image[0])
        acc = [0.0, 0.0, 0.0]
        count = 0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                x, y = cx + dx, cy + dy
                if 0 <= x < width and 0 <= y < height:
                    pixel = image[y][x]
                    acc[0] += pixel[0]
                    acc[1] += pixel[1]
                    acc[2] += pixel[2]
                    count += 1
        return (acc[0] / count, acc[1] / count, acc[2] / count)

    # ------------------------------------------------------------------
    # Motion blur: speed-scaled sampling + temporal accumulation,
    # static pixels excluded.
    # ------------------------------------------------------------------
    def apply_motion_blur(self, image=None, velocity=None, shutter=1.0, accumulation=0.5):
        image = image if image is not None else self._default_image()
        height, width = len(image), len(image[0])
        velocity = velocity if velocity is not None else self._default_velocity(width, height)
        frame = []
        for y in range(height):
            row = []
            for x in range(width):
                vx, vy = velocity[y][x]
                speed = math.hypot(vx, vy)
                if speed < self.STATIC_SPEED_EPSILON:
                    row.append(image[y][x])  # static objects stay sharp
                    continue
                # Blur length scales with speed: fast motion smears more.
                samples = 1 + min(self.MAX_MOTION_SAMPLES - 1,
                                  int(speed * shutter * self.MAX_MOTION_SAMPLES))
                acc = [0.0, 0.0, 0.0]
                for i in range(samples):
                    t = i / (samples - 1) - 0.5 if samples > 1 else 0.0
                    pixel = self._sample_bilinear(
                        image, x + vx * shutter * t, y + vy * shutter * t)
                    acc[0] += pixel[0]
                    acc[1] += pixel[1]
                    acc[2] += pixel[2]
                row.append((acc[0] / samples, acc[1] / samples, acc[2] / samples))
            frame.append(row)
        # Temporal accumulation blends in the previous frame for trails.
        history = self.state.get("motion_history")
        if (history is not None and accumulation > 0.0
                and len(history) == height and len(history[0]) == width):
            blend = _clamp(accumulation)
            frame = [[tuple(_clamp(c * (1.0 - blend) + h * blend)
                           for c, h in zip(current, previous))
                      for current, previous in zip(row_current, row_previous)]
                     for row_current, row_previous in zip(frame, history)]
        self.state["motion_history"] = frame
        return frame

    @staticmethod
    def _sample_bilinear(image, fx, fy):
        height, width = len(image), len(image[0])
        fx = _clamp(fx, 0.0, width - 1.0)
        fy = _clamp(fy, 0.0, height - 1.0)
        x0, y0 = int(fx), int(fy)
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        tx, ty = fx - x0, fy - y0
        return tuple(
            (image[y0][x0][c] * (1 - tx) + image[y0][x1][c] * tx) * (1 - ty)
            + (image[y1][x0][c] * (1 - tx) + image[y1][x1][c] * tx) * ty
            for c in range(3))

    # ------------------------------------------------------------------
    # Color grading: white balance + hue/saturation + brightness/contrast,
    # baked into a cached 3D LUT for fast per-pixel lookup.
    # ------------------------------------------------------------------
    def apply_color_grading(self, image=None, brightness=0.0, contrast=1.0,
                            saturation=1.0, hue_shift=0.0,
                            temperature=0.0, tint=0.0, lut_size=None):
        image = image if image is not None else self._default_image()
        size = lut_size or self.LUT_SIZE
        params = (brightness, contrast, saturation, hue_shift,
                  temperature, tint, size)
        if self.state.get("grading_lut_params") != params:
            self.state["grading_lut"] = self._build_grading_lut(
                size, brightness, contrast, saturation,
                hue_shift, temperature, tint)
            self.state["grading_lut_params"] = params
        lut = self.state["grading_lut"]
        scale = size - 1
        return [[lut[round(_clamp(r) * scale)]
                  [round(_clamp(g) * scale)]
                  [round(_clamp(b) * scale)]
                 for r, g, b in row]
                for row in image]

    def _build_grading_lut(self, size, brightness, contrast, saturation,
                           hue_shift, temperature, tint):
        scale = size - 1
        lut = []
        for ri in range(size):
            plane = []
            for gi in range(size):
                row = []
                for bi in range(size):
                    row.append(self._grade_pixel(
                        (ri / scale, gi / scale, bi / scale),
                        brightness, contrast, saturation,
                        hue_shift, temperature, tint))
                plane.append(row)
            lut.append(plane)
        return lut

    @staticmethod
    def _grade_pixel(pixel, brightness, contrast, saturation,
                     hue_shift, temperature, tint):
        r, g, b = pixel
        # White balance: temperature (cool<->warm), tint (green<->magenta).
        r *= 1.0 + 0.1 * temperature
        b *= 1.0 - 0.1 * temperature
        g *= 1.0 + 0.1 * tint
        # Hue rotation about the grey axis.
        angle = math.radians(hue_shift)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        r, g, b = (
            (0.213 + 0.787 * cos_a - 0.213 * sin_a) * r
            + (0.715 - 0.715 * cos_a - 0.715 * sin_a) * g
            + (0.072 - 0.072 * cos_a + 0.928 * sin_a) * b,
            (0.213 - 0.213 * cos_a + 0.143 * sin_a) * r
            + (0.715 + 0.285 * cos_a + 0.140 * sin_a) * g
            + (0.072 - 0.072 * cos_a - 0.283 * sin_a) * b,
            (0.213 - 0.213 * cos_a - 0.787 * sin_a) * r
            + (0.715 - 0.715 * cos_a + 0.715 * sin_a) * g
            + (0.072 + 0.928 * cos_a + 0.072 * sin_a) * b,
        )
        # Saturation: interpolate between the grey value and the color.
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        r = luma + (r - luma) * saturation
        g = luma + (g - luma) * saturation
        b = luma + (b - luma) * saturation
        # Brightness / contrast.
        return tuple(_clamp((c + brightness - 0.5) * contrast + 0.5)
                     for c in (r, g, b))

    # ------------------------------------------------------------------
    # Effect ordering: canonical order, skippable effects, ping-pong
    # render-target reuse.
    # ------------------------------------------------------------------
    def sort_post_process(self, effects=None):
        if effects is None:
            effects = list(self.EFFECT_ORDER)
        normalized = []
        for index, effect in enumerate(effects):
            if isinstance(effect, str):
                effect = {"name": effect}
            normalized.append((effect.get("name"),
                               bool(effect.get("enabled", True)), index))
        rank = {name: i for i, name in enumerate(self.EFFECT_ORDER)}
        active = [(rank.get(name, len(rank)), index, name)
                  for name, enabled, index in normalized if enabled]
        active.sort()
        return [{"name": name, "render_target": self._acquire_render_target(slot)}
                for slot, (_, _, name) in enumerate(active)]

    def _acquire_render_target(self, slot):
        # Ping-pong between two pooled targets instead of allocating a new
        # render target per effect.
        target = self._render_target_pool[slot % len(self._render_target_pool)]
        allocated = self.state.setdefault("render_targets_allocated", set())
        allocated.add(target)
        return target
