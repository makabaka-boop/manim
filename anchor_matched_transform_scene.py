"""
Demo for AnchorMatchedTransform.

A triangle's straight edges bend into curved ones. The curved contour starts at a
different anchor than the triangle and cuts every edge into two quadratic curves
rather than one, so an ordinary Transform lines the two lists up from index 0 and
the triangle's corners slide along the target's edges. Giving the pairs of anchors
that are meant to be the same corners keeps every corner pinned to its corner and
leaves each edge to bend only with the edge it corresponds to.
"""
import numpy as np

from manimlib import *
from manimlib.animation.transform import AnchorMatchedTransform

# A Chinese font present on macOS, for the labels
FONT = "PingFang SC"

TRI_COLOR = BLUE_D
CURVE_COLOR = GREEN_D
DOT_COLOR = YELLOW_D
BULGE = 0.38
RADIUS = 1.15


def curved_triangle(vertices, bulge=BULGE):
    """One closed contour, each edge bowed outward by one cubic bezier.

    Each cubic is stored as two quadratic curves, so every edge carries two
    segments and a smooth mid-edge anchor: corners then sit at every other
    anchor, unlike the triangle's one segment per edge.
    """
    out = VMobject()
    out.set_points(np.array([vertices[0]], dtype=float))
    for i in range(3):
        a, b = vertices[i], vertices[(i + 1) % 3]
        control = 0.5 * (a + b) + bulge * normalize(0.5 * (a + b))
        out.add_cubic_bezier_curve_to(
            a + (2 / 3) * (control - a),
            b + (2 / 3) * (control - b),
            b,
        )
    return out


def rotate_closed_anchors(mobject, anchor_shift):
    """Cyclically rotate where a single closed contour starts, same drawing."""
    points = mobject.get_points()
    n = (len(points) - 1) // 2
    shift = (2 * anchor_shift) % (2 * n)
    base = points[:-1]
    tiled = np.vstack([base, base])
    mobject.set_points(np.vstack([
        tiled[shift:shift + 2 * n],
        tiled[shift:shift + 1],
    ]))


def corner_dots(vertices):
    return VGroup(*[
        Dot(v, radius=0.045, color=DOT_COLOR, fill_opacity=1.0)
        for v in vertices
    ])


def label(text, **kwargs):
    config = dict(font=FONT, font_size=30, color=GREY_A)
    config.update(kwargs)
    return Text(text, **config)


class AnchorCorrespondenceDemo(Scene):
    def construct(self):
        # The two curves are identical; the only difference is how the morph
        # lines the contours up
        left_center = np.array([-3.4, 0.15, 0.0])
        right_center = np.array([3.4, 0.15, 0.0])
        left_vertices = Triangle(radius=RADIUS).get_vertices() + left_center
        right_vertices = Triangle(radius=RADIUS).get_vertices() + right_center

        tri_left = (
            Triangle(radius=RADIUS)
            .move_to(left_center)
            .set_stroke(TRI_COLOR, 5)
            .set_fill(opacity=0)
        )
        tri_right = (
            Triangle(radius=RADIUS)
            .move_to(right_center)
            .set_stroke(TRI_COLOR, 5)
            .set_fill(opacity=0)
        )
        dots_left = corner_dots(left_vertices)
        dots_right = corner_dots(right_vertices)

        target_left = curved_triangle(left_vertices).set_stroke(CURVE_COLOR, 5)
        target_right = curved_triangle(right_vertices).set_stroke(CURVE_COLOR, 5)
        # Start each curved contour at a mid-edge anchor rather than a corner
        rotate_closed_anchors(target_left, 1)
        rotate_closed_anchors(target_right, 1)
        # After the rotation the corner anchors sit at indices 1, 3, 5.
        # The pairs may begin anywhere and only have to run in path order.
        pairs = [(2, 3), (0, 5), (1, 1)]

        title = label(
            "三角形的直边弯成曲边：起点 / 分段不一致时，角会滑到边上",
            font_size=34, color=WHITE,
        ).to_edge(UP, buff=0.35)

        caption_left = label("普通 Transform", font_size=30, color=RED_C)
        caption_left.next_to(tri_left, DOWN, buff=0.9)
        caption_right = label("锚点配对 AnchorMatchedTransform", font_size=30, color=GREEN_C)
        caption_right.next_to(tri_right, DOWN, buff=0.9)

        pairs_text = label(
            "pairs = [(2,3), (0,5), (1,1)]，角对角、边对边",
            font_size=26, color=GREY_B,
        ).next_to(caption_right, DOWN, buff=0.25)
        slide_text = label(
            "三个角沿着目标的边滑走",
            font_size=26, color=GREY_B,
        ).next_to(caption_left, DOWN, buff=0.25)

        self.play(FadeIn(title, run_time=0.6))
        self.play(
            FadeIn(tri_left), FadeIn(tri_right),
            FadeIn(dots_left), FadeIn(dots_right),
            FadeIn(caption_left), FadeIn(caption_right),
            FadeIn(slide_text), FadeIn(pairs_text),
            run_time=0.8,
        )
        self.wait(0.6)

        # The two morphs run side by side, from and to the same shapes
        self.play(
            Transform(tri_left, target_left, run_time=3.5, rate_func=linear),
            AnchorMatchedTransform(
                tri_right, target_right,
                anchor_pairs=pairs,
                run_time=3.5, rate_func=linear,
            ),
        )
        self.wait(0.8)

        # The paired morph lands exactly on the untouched target object, so the
        # swap is invisible and the original target is what stays in the scene
        self.remove(tri_right)
        self.add(target_right)

        note = label(
            "结束后：左侧停在对齐出的形状上；右侧换成未被改动过的目标对象本身",
            font_size=24, color=GREY_C,
        ).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(note, run_time=0.6))
        self.wait(1.2)

        # ---- Invalid pairing: error before the morph starts, figure kept ----
        rest = Group(*[m for m in self.mobjects if m is not title])
        self.play(FadeOut(rest), run_time=0.6)

        tri = (
            Triangle(radius=RADIUS)
            .set_stroke(TRI_COLOR, 5)
            .set_fill(opacity=0)
        )
        vertices = Triangle(radius=RADIUS).get_vertices()
        dots = corner_dots(vertices)
        target = curved_triangle(vertices).set_stroke(CURVE_COLOR, 5)
        rotate_closed_anchors(target, 1)

        error_title = label(
            "非法配对：变形开始前报错，原图保留",
            font_size=34, color=WHITE,
        ).to_edge(UP, buff=0.35)
        self.play(
            FadeTransform(title, error_title),
            FadeIn(tri), FadeIn(dots),
            run_time=0.6,
        )

        # begin() validates and raises before touching either contour, so this
        # can be exercised straight on the animation without playing it
        bad_pairs = [(0, 1), (1, 3), (9, 5)]
        bad_anim = AnchorMatchedTransform(
            tri, target, anchor_pairs=bad_pairs, run_time=1.0,
        )
        message = None
        try:
            bad_anim.begin()
        except ValueError as error:
            message = str(error)
        assert message is not None, "expected an out-of-range error"

        error_text = label(
            f"anchor_pairs = {bad_pairs}",
            font_size=26, color=GREY_B,
        ).to_edge(DOWN, buff=0.95)
        error_msg = label(
            f"ValueError: {message}",
            font_size=26, color=RED_C,
        ).to_edge(DOWN, buff=0.45)
        kept_text = label(
            "越界 / 重复使用锚点 / 顺序不一致，都会在变形前拒绝；三角形仍是原样",
            font_size=24, color=GREY_A,
        ).next_to(tri, DOWN, buff=0.7)
        self.play(FadeIn(error_text), FadeIn(error_msg), FadeIn(kept_text), run_time=0.6)
        self.wait(2.0)

        self.play(
            FadeOut(VGroup(
                tri, dots, error_title, error_text, error_msg, kept_text,
            )),
            run_time=0.6,
        )
