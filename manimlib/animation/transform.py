from __future__ import annotations

import inspect

import numpy as np

from manimlib.animation.animation import Animation
from manimlib.constants import DEG
from manimlib.constants import OUT
from manimlib.mobject.mobject import Group
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.paths import path_along_arc
from manimlib.utils.paths import straight_path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Iterable, Sequence
    import numpy.typing as npt
    from manimlib.scene.scene import Scene
    from manimlib.typing import ManimColor


class Transform(Animation):
    replace_mobject_with_target_in_scene: bool = False

    def __init__(
        self,
        mobject: Mobject,
        target_mobject: Mobject | None = None,
        path_arc: float | Tuple[float, float] = 0.0,
        path_arc_axis: np.ndarray = OUT,
        path_func: Callable | None = None,
        **kwargs
    ):
        self.target_mobject = target_mobject
        self.path_arc = path_arc
        self.path_arc_axis = path_arc_axis
        self.path_func = path_func
        super().__init__(mobject, **kwargs)
        self.init_path_func()

    def init_path_func(self) -> None:
        if self.path_func is not None:
            return
        elif isinstance(self.path_arc, float) and self.path_arc == 0:
            self.path_func = straight_path
        else:
            self.path_func = path_along_arc(
                self.path_arc,
                self.path_arc_axis,
            )

    def begin(self) -> None:
        self.target_mobject = self.create_target()
        self.check_target_mobject_validity()

        if self.mobject.is_aligned_with(self.target_mobject):
            # Nothing to align, so nothing is done to the target and it can stand as it is
            self.target_copy = self.target_mobject
        else:
            # Use a copy of target_mobject for the align_data_and_family
            # call so that the actual target_mobject stays
            # preserved, since calling align_data will potentially
            # change the structure of both arguments
            self.target_copy = self.target_mobject.copy()
            self.mobject.align_data_and_family(self.target_copy)
        super().begin()

    def create_target(self) -> Mobject:
        # Has no meaningful effect here, but may be useful
        # in subclasses
        return self.target_mobject

    def check_target_mobject_validity(self) -> None:
        if self.target_mobject is None:
            raise Exception(
                f"{self.__class__.__name__}.create_target not properly implemented"
            )

    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)
        if self.replace_mobject_with_target_in_scene:
            scene.remove(self.mobject)
            scene.add(self.target_mobject)

    def update_config(self, **kwargs) -> None:
        Animation.update_config(self, **kwargs)
        if "path_arc" in kwargs:
            self.path_func = path_along_arc(
                kwargs["path_arc"],
                kwargs.get("path_arc_axis", OUT)
            )

    def get_all_mobjects(self) -> list[Mobject]:
        return [
            self.mobject,
            self.starting_mobject,
            self.target_mobject,
            self.target_copy,
        ]

    def get_all_families_zipped(self) -> zip[tuple[Mobject]]:
        return zip(*[
            mob.get_family()
            for mob in [
                self.mobject,
                self.starting_mobject,
                self.target_copy,
            ]
        ])

    def get_interpolation_ends(self) -> tuple[Mobject, Mobject]:
        return self.starting_mobject, self.target_copy

    def interpolate_submobject(
        self,
        submob: Mobject,
        start: Mobject,
        target_copy: Mobject,
        alpha: float
    ):
        submob.interpolate(start, target_copy, alpha, self.path_func)
        return self


class ReplacementTransform(Transform):
    replace_mobject_with_target_in_scene: bool = True


class TransformFromCopy(Transform):
    replace_mobject_with_target_in_scene: bool = True

    def __init__(self, mobject: Mobject, target_mobject: Mobject, **kwargs):
        super().__init__(mobject.copy(), target_mobject, **kwargs)


class MoveToTarget(Transform):
    def __init__(self, mobject: Mobject, **kwargs):
        self.check_validity_of_input(mobject)
        super().__init__(mobject, mobject.target, **kwargs)

    def check_validity_of_input(self, mobject: Mobject) -> None:
        if not hasattr(mobject, "target"):
            raise Exception(
                "MoveToTarget called on mobject without attribute 'target'"
            )


class _MethodAnimation(MoveToTarget):
    def __init__(self, mobject: Mobject, methods: list[Callable], **kwargs):
        self.methods = methods
        super().__init__(mobject, **kwargs)


class ApplyMethod(Transform):
    def __init__(self, method: Callable, *args, **kwargs):
        """
        method is a method of Mobject, *args are arguments for
        that method.  Key word arguments should be passed in
        as the last arg, as a dict, since **kwargs is for
        configuration of the transform itself

        Relies on the fact that mobject methods return the mobject
        """
        self.check_validity_of_input(method)
        self.method = method
        self.method_args = args
        super().__init__(method.__self__, **kwargs)

    def check_validity_of_input(self, method: Callable) -> None:
        if not inspect.ismethod(method):
            raise Exception(
                "Whoops, looks like you accidentally invoked "
                "the method you want to animate"
            )
        assert isinstance(method.__self__, Mobject)

    def create_target(self) -> Mobject:
        method = self.method
        # Make sure it's a list so that args.pop() works
        args = list(self.method_args)

        if len(args) > 0 and isinstance(args[-1], dict):
            method_kwargs = args.pop()
        else:
            method_kwargs = {}
        target = method.__self__.copy()
        method.__func__(target, *args, **method_kwargs)
        return target


class ApplyPointwiseFunction(ApplyMethod):
    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        mobject: Mobject,
        run_time: float = 3.0,
        **kwargs
    ):
        super().__init__(mobject.apply_function, function, run_time=run_time, **kwargs)


class ApplyPointwiseFunctionToCenter(Transform):
    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        mobject: Mobject,
        **kwargs
    ):
        self.function = function
        super().__init__(mobject, **kwargs)

    def create_target(self) -> Mobject:
        return self.mobject.copy().move_to(self.function(self.mobject.get_center()))


class FadeToColor(ApplyMethod):
    def __init__(
        self,
        mobject: Mobject,
        color: ManimColor,
        **kwargs
    ):
        super().__init__(mobject.set_color, color, **kwargs)


class ScaleInPlace(ApplyMethod):
    def __init__(
        self,
        mobject: Mobject,
        scale_factor: npt.ArrayLike,
        **kwargs
    ):
        super().__init__(mobject.scale, scale_factor, **kwargs)


class ShrinkToCenter(ScaleInPlace):
    def __init__(self, mobject: Mobject, **kwargs):
        super().__init__(mobject, 0, **kwargs)


class Restore(Transform):
    def __init__(self, mobject: Mobject, **kwargs):
        if not hasattr(mobject, "saved_state") or mobject.saved_state is None:
            raise Exception("Trying to restore without having saved")
        super().__init__(mobject, mobject.saved_state, **kwargs)


class ApplyFunction(Transform):
    def __init__(
        self,
        function: Callable[[Mobject], Mobject],
        mobject: Mobject,
        **kwargs
    ):
        self.function = function
        super().__init__(mobject, **kwargs)

    def create_target(self) -> Mobject:
        target = self.function(self.mobject.copy())
        if not isinstance(target, Mobject):
            raise Exception("Functions passed to ApplyFunction must return object of type Mobject")
        return target


class ApplyMatrix(ApplyPointwiseFunction):
    def __init__(
        self,
        matrix: npt.ArrayLike,
        mobject: Mobject,
        **kwargs
    ):
        matrix = self.initialize_matrix(matrix)

        def func(p):
            return np.dot(p, matrix.T)

        super().__init__(func, mobject, **kwargs)

    def initialize_matrix(self, matrix: npt.ArrayLike) -> np.ndarray:
        matrix = np.array(matrix)
        if matrix.shape == (2, 2):
            new_matrix = np.identity(3)
            new_matrix[:2, :2] = matrix
            matrix = new_matrix
        elif matrix.shape != (3, 3):
            raise Exception("Matrix has bad dimensions")
        return matrix


class ApplyComplexFunction(ApplyMethod):
    def __init__(
        self,
        function: Callable[[complex], complex],
        mobject: Mobject,
        **kwargs
    ):
        self.function = function
        method = mobject.apply_complex_function
        super().__init__(method, function, **kwargs)

    def init_path_func(self) -> None:
        func1 = self.function(complex(1))
        self.path_arc = np.log(func1).imag
        super().init_path_func()

###


class CyclicReplace(Transform):
    def __init__(self, *mobjects: Mobject, path_arc=90 * DEG, **kwargs):
        super().__init__(Group(*mobjects), path_arc=path_arc, **kwargs)

    def create_target(self) -> Mobject:
        group = self.mobject
        target = group.copy()
        cycled_targets = [target[-1], *target[:-1]]
        for m1, m2 in zip(cycled_targets, group):
            m1.move_to(m2)
        return target


class Swap(CyclicReplace):
    """Alternate name for CyclicReplace"""
    pass


class AnchorMatchedTransform(Transform):
    """Transform one closed outline into another with explicit anchor correspondences.

    An ordinary :class:`Transform` lines the two point lists up from their first
    point and pads the shorter path, so changing where either contour starts, or
    how its curves are cut into pieces, can have a corner of the starting outline
    slide along an edge of the target one on the way there. This class lets the
    caller say which anchors correspond instead.

    Both mobjects must be single planar closed :class:`VMobject` outlines with one
    uniform stroke. ``anchor_pairs`` is a list of ``(index, target_index)`` pairs,
    indexing the anchors of each contour along its own path order, starting at 0.
    The anchor closing a path back onto its first point is the same point and is
    not counted again. The pairs may start at any anchor and run around either
    contour once in its path order. Each paired anchor is then taken straight to
    its counterpart, and the curves between two consecutive pairs deform together
    as one interval even when the two sides cut that interval into different
    numbers of curves.

    At the start and end the two original outlines and styles are kept (the target
    mobject itself is never modified, alignment is done to a copy), and leaving
    ``anchor_pairs`` as ``None`` gives an ordinary :class:`Transform`.

    An out of range index, an anchor used in more than one pair, or pairs which do
    not follow path order raise a ``ValueError`` in :meth:`begin`, before either
    contour is touched, so the scene is left showing the original.
    """
    def __init__(
        self,
        mobject: VMobject,
        target_mobject: VMobject | None = None,
        anchor_pairs: Iterable[tuple[int, int]] | None = None,
        path_arc: float | Tuple[float, float] = 0.0,
        path_arc_axis: np.ndarray = OUT,
        path_func: Callable | None = None,
        **kwargs
    ):
        self.anchor_pairs = (
            None if anchor_pairs is None else [tuple(pair) for pair in anchor_pairs]
        )
        super().__init__(
            mobject, target_mobject,
            path_arc=path_arc,
            path_arc_axis=path_arc_axis,
            path_func=path_func,
            **kwargs
        )

    def begin(self) -> None:
        self.target_mobject = self.create_target()
        self.check_target_mobject_validity()
        if self.anchor_pairs is None:
            # No correspondence asked for: align in the usual way
            self.align_default()
            Animation.begin(self)
            return

        # Validate before either contour is touched, so a bad call leaves the
        # scene showing the original
        anchor_counts = [
            self.check_contour_is_eligible(mob)
            for mob in [self.mobject, self.target_mobject]
        ]
        pairs = self.check_anchor_pairs(self.anchor_pairs, anchor_counts)

        self.target_copy = self.target_mobject.copy()
        self.align_by_anchor_pairs(self.mobject, self.target_copy, pairs)
        Animation.begin(self)

    def align_default(self) -> None:
        if self.mobject.is_aligned_with(self.target_mobject):
            # Nothing to align, so nothing is done to the target and it can stand as it is
            self.target_copy = self.target_mobject
        else:
            # Work on a copy of the target so the actual target stays preserved
            self.target_copy = self.target_mobject.copy()
            self.mobject.align_data_and_family(self.target_copy)

    @staticmethod
    def check_contour_is_eligible(mobject: VMobject) -> int:
        """The number of anchors a single closed outline counts, or an error."""
        if not isinstance(mobject, VMobject):
            raise ValueError(
                "AnchorMatchedTransform only transforms VMobject outlines"
            )
        if len(mobject.submobjects) > 0:
            raise ValueError(
                "AnchorMatchedTransform only applies to a single outline with no "
                "submobjects, not a family of them"
            )
        if not mobject.has_points():
            raise ValueError(
                "AnchorMatchedTransform needs an outline holding points"
            )
        points = mobject.get_points()
        if len(points) < 7 or len(points) % 2 != 1:
            raise ValueError(
                "AnchorMatchedTransform needs a closed outline of at least three anchors"
            )
        if len(mobject.get_subpaths()) != 1:
            raise ValueError(
                "AnchorMatchedTransform needs one single closed subpath"
            )
        if not mobject.is_closed():
            raise ValueError(
                "AnchorMatchedTransform only applies to closed contours"
            )
        if not np.allclose(points[:, 2], points[0, 2], atol=1e-6):
            raise ValueError(
                "AnchorMatchedTransform only applies to planar contours"
            )
        if not mobject.has_stroke():
            raise ValueError(
                "AnchorMatchedTransform only applies to contours with a stroke"
            )
        widths = mobject.data["stroke_width"][:, 0]
        rgba = mobject.data["stroke_rgba"]
        if not np.allclose(widths, widths[0], atol=1e-6):
            raise ValueError(
                "AnchorMatchedTransform needs one uniform stroke width; found "
                "varying widths along the contour"
            )
        if not np.allclose(rgba, rgba[0], atol=1e-6):
            raise ValueError(
                "AnchorMatchedTransform needs one uniform stroke color; found "
                "varying colors along the contour"
            )
        # For a closed point list [a0, h0, ..., a_{n-1}, h_{n-1}, a0] the last
        # anchor is the first one over again, so it is not counted twice
        return (len(points) - 1) // 2

    @staticmethod
    def check_anchor_pairs(
        anchor_pairs: Sequence[tuple[int, int]],
        anchor_counts: Sequence[int],
    ) -> list[tuple[int, int]]:
        if len(anchor_pairs) == 0:
            raise ValueError("anchor_pairs is empty; pass None for an ordinary Transform")

        pairs = []
        for pair in anchor_pairs:
            if len(pair) != 2 or not all(isinstance(i, (int, np.integer)) for i in pair):
                raise ValueError(
                    f"Each entry of anchor_pairs must be a pair of int indices, "
                    f"not {pair}"
                )
            pairs.append((int(pair[0]), int(pair[1])))

        for side, count in enumerate(anchor_counts):
            for pair in pairs:
                index = pair[side]
                if not 0 <= index < count:
                    raise ValueError(
                        f"Anchor index {index} is out of range for a closed contour "
                        f"with {count} anchor{'s' if count != 1 else ''}"
                    )

        source_indices = [pair[0] for pair in pairs]
        target_indices = [pair[1] for pair in pairs]
        for name, indices in [("source", source_indices), ("target", target_indices)]:
            repeated = {i for i in indices if indices.count(i) > 1}
            if repeated:
                raise ValueError(
                    f"{name} anchor {sorted(repeated)[0]} is used more than once in "
                    f"anchor_pairs; each anchor may be paired at most once"
                )
            count = anchor_counts[0] if name == "source" else anchor_counts[1]
            # Rotate so the first listed anchor sits at 0. The offsets must then
            # strictly grow, which is exactly "around once in path order"
            offsets = [(index - indices[0]) % count for index in indices]
            if any(offsets[i + 1] <= offsets[i] for i in range(len(offsets) - 1)):
                raise ValueError(
                    f"The {name} anchor indices {tuple(indices)} do not run in path "
                    f"order around the contour starting from the first pair"
                )
        return pairs

    @staticmethod
    def cyclic_subpath(
        points: np.ndarray,
        start_anchor: int,
        end_anchor: int,
        n_anchors: int,
    ) -> np.ndarray:
        """Points of one closed contour from one anchor forward to another.

        The closing anchor repeats the first, so the base list is tiled twice and
        the wanted run is read out of it. Result length is ``2 * n_curves + 1``.
        """
        n_curves = (end_anchor - start_anchor) % n_anchors
        base = points[:-1]
        tiled = np.vstack([base, base])
        start = 2 * start_anchor
        result = tiled[start:start + 2 * n_curves + 1].copy()
        result.setflags(write=True)
        return result

    @classmethod
    def align_by_anchor_pairs(
        cls,
        source: VMobject,
        target: VMobject,
        pairs: Sequence[tuple[int, int]],
    ) -> None:
        """Re-cut the two closed paths so paired anchors share an array index.

        Each interval between consecutive pairs is aligned on its own: whichever
        side traces it with fewer curves is subdivided to match, so an interval
        only ever mixes with its matching interval, however differently the two
        sides cut their curves.
        """
        source_points = source.get_points()
        target_points = target.get_points()
        n_source = (len(source_points) - 1) // 2
        n_target = (len(target_points) - 1) // 2

        source_pieces = []
        target_pieces = []
        n_pairs = len(pairs)
        for i, (source_anchor, target_anchor) in enumerate(pairs):
            next_source, next_target = pairs[(i + 1) % n_pairs]
            source_piece = cls.cyclic_subpath(
                source_points, source_anchor, next_source, n_source
            )
            target_piece = cls.cyclic_subpath(
                target_points, target_anchor, next_target, n_target
            )
            source_curves = (len(source_piece) - 1) // 2
            target_curves = (len(target_piece) - 1) // 2
            if source_curves < target_curves:
                source_piece = source.insert_n_curves_to_point_list(
                    target_curves - source_curves, source_piece
                )
            elif target_curves < source_curves:
                target_piece = target.insert_n_curves_to_point_list(
                    source_curves - target_curves, target_piece
                )
            source_pieces.append(source_piece)
            target_pieces.append(target_piece)

        new_source_points = np.vstack([
            source_pieces[0],
            *[piece[1:] for piece in source_pieces[1:]],
        ])
        new_target_points = np.vstack([
            target_pieces[0],
            *[piece[1:] for piece in target_pieces[1:]],
        ])
        assert len(new_source_points) == len(new_target_points)

        # set_points resizes the per-point style rows with order preserved, which
        # for a uniform stroke leaves them uniform, and refreshes subpath data
        source.set_points(new_source_points)
        target.set_points(new_target_points)
