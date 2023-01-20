# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import functools
import inspect
from pathlib import Path

import symforce.symbolic as sf
from symforce import ops
from symforce import typing as T
from symforce.codegen import Codegen
from symforce.codegen import CppConfig

TYPES = (sf.Rot2, sf.Rot3, sf.V3, sf.Pose2, sf.Pose3)


def get_between_factor_docstring(between_argument_name: str) -> str:
    return """
    Residual that penalizes the difference between between(a, b) and {a_T_b}.

    In vector space terms that would be:
        (b - a) - {a_T_b}

    In lie group terms:
        local_coordinates({a_T_b}, between(a, b))
        to_tangent(compose(inverse({a_T_b}), compose(inverse(a), b)))

    Args:
        sqrt_info: Square root information matrix to whiten residual. This can be computed from
                   a covariance matrix as the cholesky decomposition of the inverse. In the case
                   of a diagonal it will contain 1/sigma values. Must match the tangent dim.
    """.format(
        a_T_b=between_argument_name
    )


def get_prior_docstring() -> str:
    return """
    Residual that penalizes the difference between a value and prior (desired / measured value).

    In vector space terms that would be:
        prior - value

    In lie group terms:
        to_tangent(compose(inverse(value), prior))

    Args:
        sqrt_info: Square root information matrix to whiten residual. This can be computed from
                   a covariance matrix as the cholesky decomposition of the inverse. In the case
                   of a diagonal it will contain 1/sigma values. Must match the tangent dim.
    """


def between_factor(
    a: T.Element, b: T.Element, a_T_b: T.Element, sqrt_info: sf.Matrix, epsilon: sf.Scalar = 0
) -> sf.Matrix:
    assert type(a) == type(b) == type(a_T_b)  # pylint: disable=unidiomatic-typecheck
    assert sqrt_info.rows == sqrt_info.cols == ops.LieGroupOps.tangent_dim(a)

    # Compute error
    tangent_error = ops.LieGroupOps.local_coordinates(
        a_T_b, ops.LieGroupOps.between(a, b), epsilon=epsilon
    )

    # Apply noise model
    residual = sqrt_info * sf.M(tangent_error)

    return residual


def prior_factor(
    value: T.Element, prior: T.Element, sqrt_info: sf.Matrix, epsilon: sf.Scalar = 0
) -> sf.Matrix:
    assert type(value) == type(prior)  # pylint: disable=unidiomatic-typecheck
    assert sqrt_info.rows == sqrt_info.cols == ops.LieGroupOps.tangent_dim(value)

    # Compute error
    tangent_error = ops.LieGroupOps.local_coordinates(prior, value, epsilon=epsilon)

    # Apply noise model
    residual = sqrt_info * sf.M(tangent_error)

    return residual


def get_arg_index(func: T.Callable, arg: str) -> int:
    func_parameters = inspect.signature(func).parameters
    try:
        return list(func_parameters).index(arg)
    except ValueError as error:
        raise ValueError(f"{arg} is not an argument of {func}") from error


def modify_argument(
    core_func: T.Callable, arg_to_modify: str, new_arg_type: T.Type, modification: T.Callable
) -> T.Callable:
    arg_index = get_arg_index(core_func, arg_to_modify)

    @functools.wraps(core_func)
    def wrapper(*args: T.Any, **kwargs: T.Any) -> T.Any:
        args_list = list(args)
        if arg_index < len(args):
            # Then arg_to_modify was passed in args
            args_list[arg_index] = modification(args[arg_index])
        else:
            # arg_to_modify should have been passed in kwargs
            try:
                kwargs[arg_to_modify] = modification(kwargs[arg_to_modify])
            except KeyError as error:
                raise TypeError(f"{wrapper} missing required argument {arg_to_modify}") from error

        return core_func(*args_list, **kwargs)

    wrapper.__annotations__ = dict(wrapper.__annotations__, **{arg_to_modify: new_arg_type})

    return wrapper


def is_not_fixed_size_square_matrix(type_t: T.Type) -> bool:
    return (
        not issubclass(type_t, sf.Matrix)
        or type_t == sf.Matrix
        or type_t.SHAPE[0] != type_t.SHAPE[1]
    )


def _get_sqrt_info_dim(func: T.Callable) -> int:
    if "sqrt_info" in func.__annotations__:
        sqrt_info_type = func.__annotations__["sqrt_info"]
        if is_not_fixed_size_square_matrix(sqrt_info_type):
            raise ValueError(
                f"""Expected sqrt_info to be annotated as a fixed size square matrix. Instead
                found {sqrt_info_type}. Either fix annotation or explicitly pass in expected number
                of dimensions of sqrt_info."""
            )
        return sqrt_info_type.SHAPE[0]
    else:
        raise ValueError(
            "sqrt_info missing annotation. Either add one or explicitly pass in expected number of dimensions"
        )


def isotropic_sqrt_info_wrapper(func: T.Callable, sqrt_info_dim: int = None) -> T.Callable:
    if sqrt_info_dim is None:
        sqrt_info_dim = _get_sqrt_info_dim(func)

    return modify_argument(
        func,
        "sqrt_info",
        T.Scalar,
        lambda sqrt_info: sqrt_info * sf.M.eye(sqrt_info_dim, sqrt_info_dim),
    )


def diagonal_sqrt_info_wrapper(func: T.Callable, sqrt_info_dim: int = None) -> T.Callable:
    if sqrt_info_dim is None:
        sqrt_info_dim = _get_sqrt_info_dim(func)

    return modify_argument(
        func,
        "sqrt_info",
        type(sf.M(sqrt_info_dim, 1)),
        sf.M.diag,
    )


def generate_with_alternate_sqrt_infos(
    output_dir: T.Openable,
    func: T.Callable,
    name: str,
    which_args: T.Sequence[str],
    input_types: T.Sequence[T.ElementOrType] = None,
    sqrt_info_dim: int = None,
    output_names: T.Sequence[str] = None,
    docstring: str = None,
) -> None:

    common_header = Path(output_dir, name + ".h")
    common_header.parent.mkdir(exist_ok=True, parents=True)

    sqrt_info_index = get_arg_index(func, "sqrt_info")

    for func_variant, variant_name in [
        (func, f"{name}_square"),
        (isotropic_sqrt_info_wrapper(func, sqrt_info_dim), f"{name}_isotropic"),
        (diagonal_sqrt_info_wrapper(func, sqrt_info_dim), f"{name}_diagonal"),
    ]:
        if input_types is None:
            variant_input_types = None
        else:
            sqrt_info_type = (
                input_types[sqrt_info_index]
                if func == func_variant
                else func_variant.__annotations__["sqrt_info"]
            )
            variant_input_types = [
                t if n != sqrt_info_index else sqrt_info_type for n, t in enumerate(input_types)
            ]
        Codegen.function(
            func=func_variant,
            input_types=variant_input_types,
            output_names=output_names,
            config=CppConfig(),
            docstring=docstring,
        ).with_linearization(name=name, which_args=which_args).generate_function(
            Path(output_dir, name),
            skip_directory_nesting=True,
            generated_file_name=variant_name + ".h",
        )

        with common_header.open("a") as f:
            f.write(f'#include "./{name}/{variant_name}.h"\n')


def generate_between_factors(types: T.Sequence[T.Type], output_dir: T.Openable) -> None:
    """
    Generates between factors for each type in types into output_dir.
    """
    for cls in types:
        tangent_dim = ops.LieGroupOps.tangent_dim(cls)
        generate_with_alternate_sqrt_infos(
            output_dir,
            func=between_factor,
            name=f"between_factor_{cls.__name__.lower()}",
            which_args=["a", "b"],
            input_types=[cls, cls, cls, sf.M(tangent_dim, tangent_dim), sf.Symbol],
            sqrt_info_dim=tangent_dim,
            output_names=["res"],
            docstring=get_between_factor_docstring("a_T_b"),
        )

        generate_with_alternate_sqrt_infos(
            output_dir,
            func=prior_factor,
            name=f"prior_factor_{cls.__name__.lower()}",
            which_args=["value"],
            input_types=[cls, cls, sf.M(tangent_dim, tangent_dim), sf.Symbol],
            sqrt_info_dim=tangent_dim,
            output_names=["res"],
            docstring=get_prior_docstring(),
        )


def generate_pose3_extra_factors(output_dir: T.Openable) -> None:
    """
    Generates factors specific to Poses which penalize individual components into output_dir.

    This includes factors for only the position or rotation components of a Pose.  This can't be
    done by wrapping the other generated functions because we need jacobians with respect to the
    full pose.
    """

    def between_factor_pose3_rotation(
        a: sf.Pose3, b: sf.Pose3, a_R_b: sf.Rot3, sqrt_info: sf.Matrix33, epsilon: sf.Scalar = 0
    ) -> sf.Matrix:
        # NOTE(aaron): This should be equivalent to between_factor(a.R, b.R, a_R_b), but we write it
        # this way for explicitness and symmetry with between_factor_pose3_position, where the two
        # are not equivalent
        tangent_error = ops.LieGroupOps.local_coordinates(
            a_R_b, ops.LieGroupOps.between(a, b).R, epsilon=epsilon
        )

        return sqrt_info * sf.M(tangent_error)

    def between_factor_pose3_position(
        a: sf.Pose3,
        b: sf.Pose3,
        a_t_b: sf.Vector3,
        sqrt_info: sf.Matrix33,
        epsilon: sf.Scalar = 0,
    ) -> sf.Matrix:
        # NOTE(aaron): This is NOT the same as between_factor(a.t, b.t, a_t_b, sqrt_info, epsilon)
        # between_factor(a.t, b.t, a_t_b) would be penalizing the difference in the global frame
        # (and expecting a_t_b to be in the global frame), we want to penalize the position
        # component of between_factor(a, b, a_T_b), which is in the `a` frame
        tangent_error = ops.LieGroupOps.local_coordinates(
            a_t_b, ops.LieGroupOps.between(a, b).t, epsilon=epsilon
        )

        return sqrt_info * sf.M(tangent_error)

    def between_factor_pose3_translation_norm(
        a: sf.Pose3,
        b: sf.Pose3,
        translation_norm: sf.Scalar,
        sqrt_info: sf.Matrix11,
        epsilon: sf.Scalar = 0,
    ) -> sf.Matrix:
        """
        Residual that penalizes the difference between translation_norm and (a.t - b.t).norm().

        Args:
            sqrt_info: Square root information matrix to whiten residual. In this one dimensional case
                    this is just 1/sigma.
        """
        error = translation_norm - (a.t - b.t).norm(epsilon)
        return sqrt_info * sf.M([error])

    def prior_factor_pose3_rotation(
        value: sf.Pose3, prior: sf.Rot3, sqrt_info: sf.Matrix33, epsilon: sf.Scalar = 0
    ) -> sf.Matrix:
        return prior_factor(value.R, prior, sqrt_info, epsilon)

    def prior_factor_pose3_position(
        value: sf.Pose3, prior: sf.Vector3, sqrt_info: sf.Matrix33, epsilon: sf.Scalar = 0
    ) -> sf.Matrix:
        return prior_factor(value.t, prior, sqrt_info, epsilon)

    generate_with_alternate_sqrt_infos(
        output_dir,
        func=between_factor_pose3_rotation,
        name="between_factor_pose3_rotation",
        which_args=["a", "b"],
        output_names=["res"],
        docstring=get_between_factor_docstring("a_R_b"),
    )

    generate_with_alternate_sqrt_infos(
        output_dir,
        func=between_factor_pose3_position,
        name="between_factor_pose3_position",
        which_args=["a", "b"],
        output_names=["res"],
        docstring=get_between_factor_docstring("a_t_b"),
    )

    generate_with_alternate_sqrt_infos(
        output_dir,
        func=between_factor_pose3_translation_norm,
        name="between_factor_pose3_translation_norm",
        which_args=["a", "b"],
        output_names=["res"],
    )

    generate_with_alternate_sqrt_infos(
        output_dir,
        func=prior_factor_pose3_rotation,
        name="prior_factor_pose3_rotation",
        output_names=["res"],
        which_args=["value"],
        docstring=get_prior_docstring(),
    )

    generate_with_alternate_sqrt_infos(
        output_dir,
        func=prior_factor_pose3_position,
        name="prior_factor_pose3_position",
        output_names=["res"],
        which_args=["value"],
        docstring=get_prior_docstring(),
    )


def generate(output_dir: Path) -> None:
    """
    Prior factors and between factors for C++.
    """
    generate_between_factors(types=TYPES, output_dir=output_dir / "factors")
    generate_pose3_extra_factors(output_dir / "factors")
