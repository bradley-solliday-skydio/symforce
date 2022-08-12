# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     ops/CLASS/group_ops.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

import math
import numpy
import typing as T

import sym  # pylint: disable=unused-import


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.pose3.Pose3'>.
    """

    @staticmethod
    def identity():
        # type: () -> sym.Pose3

        # Total ops: 0

        # Input arrays

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 7
        _res[0] = 0
        _res[1] = 0
        _res[2] = 0
        _res[3] = 1
        _res[4] = 0
        _res[5] = 0
        _res[6] = 0
        return sym.Pose3.from_storage(_res)

    @staticmethod
    def inverse(a):
        # type: (sym.Pose3) -> sym.Pose3

        # Total ops: 49

        # Input arrays
        _a = a.data

        # Intermediate terms (11)
        _tmp0 = -2 * _a[1] ** 2
        _tmp1 = 1 - 2 * _a[2] ** 2
        _tmp2 = 2 * _a[0]
        _tmp3 = _a[2] * _tmp2
        _tmp4 = 2 * _a[3]
        _tmp5 = _a[1] * _tmp4
        _tmp6 = _a[1] * _tmp2
        _tmp7 = _a[2] * _tmp4
        _tmp8 = -2 * _a[0] ** 2
        _tmp9 = 2 * _a[1] * _a[2]
        _tmp10 = _a[3] * _tmp2

        # Output terms
        _res = [0.0] * 7
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        _res[4] = -_a[4] * (_tmp0 + _tmp1) - _a[5] * (_tmp6 + _tmp7) - _a[6] * (_tmp3 - _tmp5)
        _res[5] = -_a[4] * (_tmp6 - _tmp7) - _a[5] * (_tmp1 + _tmp8) - _a[6] * (_tmp10 + _tmp9)
        _res[6] = -_a[4] * (_tmp3 + _tmp5) - _a[5] * (-_tmp10 + _tmp9) - _a[6] * (_tmp0 + _tmp8 + 1)
        return sym.Pose3.from_storage(_res)

    @staticmethod
    def compose(a, b):
        # type: (sym.Pose3, sym.Pose3) -> sym.Pose3

        # Total ops: 74

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (11)
        _tmp0 = -2 * _a[2] ** 2
        _tmp1 = -2 * _a[1] ** 2
        _tmp2 = 2 * _a[0]
        _tmp3 = _a[2] * _tmp2
        _tmp4 = 2 * _a[3]
        _tmp5 = _a[1] * _tmp4
        _tmp6 = _a[1] * _tmp2
        _tmp7 = _a[2] * _tmp4
        _tmp8 = 1 - 2 * _a[0] ** 2
        _tmp9 = 2 * _a[1] * _a[2]
        _tmp10 = _a[3] * _tmp2

        # Output terms
        _res = [0.0] * 7
        _res[0] = _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0]
        _res[1] = -_a[0] * _b[2] + _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1]
        _res[2] = _a[0] * _b[1] - _a[1] * _b[0] + _a[2] * _b[3] + _a[3] * _b[2]
        _res[3] = -_a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] + _a[3] * _b[3]
        _res[4] = (
            _a[4] + _b[4] * (_tmp0 + _tmp1 + 1) + _b[5] * (_tmp6 - _tmp7) + _b[6] * (_tmp3 + _tmp5)
        )
        _res[5] = (
            _a[5] + _b[4] * (_tmp6 + _tmp7) + _b[5] * (_tmp0 + _tmp8) + _b[6] * (-_tmp10 + _tmp9)
        )
        _res[6] = (
            _a[6] + _b[4] * (_tmp3 - _tmp5) + _b[5] * (_tmp10 + _tmp9) + _b[6] * (_tmp1 + _tmp8)
        )
        return sym.Pose3.from_storage(_res)

    @staticmethod
    def between(a, b):
        # type: (sym.Pose3, sym.Pose3) -> sym.Pose3

        # Total ops: 89

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (20)
        _tmp0 = -2 * _a[2] ** 2
        _tmp1 = 1 - 2 * _a[1] ** 2
        _tmp2 = _tmp0 + _tmp1
        _tmp3 = 2 * _a[0]
        _tmp4 = _a[2] * _tmp3
        _tmp5 = 2 * _a[3]
        _tmp6 = _a[1] * _tmp5
        _tmp7 = _tmp4 - _tmp6
        _tmp8 = _a[1] * _tmp3
        _tmp9 = _a[2] * _tmp5
        _tmp10 = _tmp8 + _tmp9
        _tmp11 = -2 * _a[0] ** 2
        _tmp12 = _tmp0 + _tmp11 + 1
        _tmp13 = 2 * _a[1] * _a[2]
        _tmp14 = _a[3] * _tmp3
        _tmp15 = _tmp13 + _tmp14
        _tmp16 = _tmp8 - _tmp9
        _tmp17 = _tmp1 + _tmp11
        _tmp18 = _tmp13 - _tmp14
        _tmp19 = _tmp4 + _tmp6

        # Output terms
        _res = [0.0] * 7
        _res[0] = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0]
        _res[1] = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1]
        _res[2] = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2]
        _res[3] = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3]
        _res[4] = (
            -_a[4] * _tmp2
            - _a[5] * _tmp10
            - _a[6] * _tmp7
            + _b[4] * _tmp2
            + _b[5] * _tmp10
            + _b[6] * _tmp7
        )
        _res[5] = (
            -_a[4] * _tmp16
            - _a[5] * _tmp12
            - _a[6] * _tmp15
            + _b[4] * _tmp16
            + _b[5] * _tmp12
            + _b[6] * _tmp15
        )
        _res[6] = (
            -_a[4] * _tmp19
            - _a[5] * _tmp18
            - _a[6] * _tmp17
            + _b[4] * _tmp19
            + _b[5] * _tmp18
            + _b[6] * _tmp17
        )
        return sym.Pose3.from_storage(_res)

    @staticmethod
    def inverse_with_jacobian(a):
        # type: (sym.Pose3) -> T.Tuple[sym.Pose3, numpy.ndarray]

        # Total ops: 121

        # Input arrays
        _a = a.data

        # Intermediate terms (47)
        _tmp0 = _a[2] ** 2
        _tmp1 = -2 * _tmp0
        _tmp2 = _a[1] ** 2
        _tmp3 = -2 * _tmp2
        _tmp4 = _a[0] * _a[2]
        _tmp5 = 2 * _tmp4
        _tmp6 = _a[1] * _a[3]
        _tmp7 = 2 * _tmp6
        _tmp8 = _a[0] * _a[1]
        _tmp9 = 2 * _tmp8
        _tmp10 = _a[2] * _a[3]
        _tmp11 = 2 * _tmp10
        _tmp12 = _a[0] ** 2
        _tmp13 = 1 - 2 * _tmp12
        _tmp14 = _a[1] * _a[2]
        _tmp15 = 2 * _tmp14
        _tmp16 = _a[0] * _a[3]
        _tmp17 = 2 * _tmp16
        _tmp18 = -_tmp12
        _tmp19 = _a[3] ** 2
        _tmp20 = _tmp0 - _tmp19
        _tmp21 = _tmp18 + _tmp2 + _tmp20
        _tmp22 = 2 * _tmp10 + 2 * _tmp8
        _tmp23 = -_tmp22
        _tmp24 = 2 * _tmp4 - 2 * _tmp6
        _tmp25 = -_tmp24
        _tmp26 = -_tmp2
        _tmp27 = _tmp0 + _tmp18 + _tmp19 + _tmp26
        _tmp28 = _a[6] * _tmp27
        _tmp29 = 2 * _tmp14 - 2 * _tmp16
        _tmp30 = _a[5] * _tmp29
        _tmp31 = 2 * _tmp4 + 2 * _tmp6
        _tmp32 = _a[4] * _tmp31
        _tmp33 = _tmp12 + _tmp20 + _tmp26
        _tmp34 = _a[5] * _tmp33
        _tmp35 = 2 * _tmp14 + 2 * _tmp16
        _tmp36 = _a[6] * _tmp35
        _tmp37 = -2 * _tmp10 + 2 * _tmp8
        _tmp38 = _a[4] * _tmp37
        _tmp39 = -_tmp37
        _tmp40 = -_tmp35
        _tmp41 = _a[4] * _tmp21
        _tmp42 = _a[6] * _tmp24
        _tmp43 = _a[5] * _tmp22
        _tmp44 = -_tmp31
        _tmp45 = -_tmp29
        _tmp46 = _tmp0 - 1.0 / 2.0

        # Output terms
        _res = [0.0] * 7
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        _res[4] = -_a[4] * (_tmp1 + _tmp3 + 1) - _a[5] * (_tmp11 + _tmp9) - _a[6] * (_tmp5 - _tmp7)
        _res[5] = -_a[4] * (-_tmp11 + _tmp9) - _a[5] * (_tmp1 + _tmp13) - _a[6] * (_tmp15 + _tmp17)
        _res[6] = -_a[4] * (_tmp5 + _tmp7) - _a[5] * (_tmp15 - _tmp17) - _a[6] * (_tmp13 + _tmp3)
        _res_D_a = numpy.zeros((6, 6))
        _res_D_a[0, 0] = _tmp21
        _res_D_a[1, 0] = _tmp23
        _res_D_a[2, 0] = _tmp25
        _res_D_a[3, 0] = 0
        _res_D_a[4, 0] = -_tmp28 - _tmp30 - _tmp32
        _res_D_a[5, 0] = -_tmp34 + _tmp36 + _tmp38
        _res_D_a[0, 1] = _tmp39
        _res_D_a[1, 1] = _tmp33
        _res_D_a[2, 1] = _tmp40
        _res_D_a[3, 1] = _tmp28 + _tmp30 + _tmp32
        _res_D_a[4, 1] = 0
        _res_D_a[5, 1] = _tmp41 - _tmp42 - _tmp43
        _res_D_a[0, 2] = _tmp44
        _res_D_a[1, 2] = _tmp45
        _res_D_a[2, 2] = -_tmp27
        _res_D_a[3, 2] = _tmp34 - _tmp36 - _tmp38
        _res_D_a[4, 2] = -_tmp41 + _tmp42 + _tmp43
        _res_D_a[5, 2] = 0
        _res_D_a[0, 3] = 0
        _res_D_a[1, 3] = 0
        _res_D_a[2, 3] = 0
        _res_D_a[3, 3] = 2 * _tmp2 + 2 * _tmp46
        _res_D_a[4, 3] = _tmp39
        _res_D_a[5, 3] = _tmp44
        _res_D_a[0, 4] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[3, 4] = _tmp23
        _res_D_a[4, 4] = 2 * _tmp12 + 2 * _tmp46
        _res_D_a[5, 4] = _tmp45
        _res_D_a[0, 5] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 5] = _tmp25
        _res_D_a[4, 5] = _tmp40
        _res_D_a[5, 5] = 2 * _tmp12 + 2 * _tmp2 - 1
        return sym.Pose3.from_storage(_res), _res_D_a

    @staticmethod
    def compose_with_jacobians(a, b):
        # type: (sym.Pose3, sym.Pose3) -> T.Tuple[sym.Pose3, numpy.ndarray, numpy.ndarray]

        # Total ops: 270

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (94)
        _tmp0 = _a[1] * _b[2]
        _tmp1 = _a[2] * _b[1]
        _tmp2 = _a[0] * _b[3]
        _tmp3 = _a[3] * _b[0]
        _tmp4 = _tmp2 + _tmp3
        _tmp5 = _a[1] * _b[3]
        _tmp6 = _a[2] * _b[0]
        _tmp7 = _a[3] * _b[1]
        _tmp8 = _a[0] * _b[2]
        _tmp9 = _tmp7 - _tmp8
        _tmp10 = _tmp5 + _tmp6 + _tmp9
        _tmp11 = _a[1] * _b[0]
        _tmp12 = -_tmp11
        _tmp13 = _a[2] * _b[3]
        _tmp14 = _a[0] * _b[1]
        _tmp15 = _a[3] * _b[2]
        _tmp16 = _tmp14 + _tmp15
        _tmp17 = _tmp12 + _tmp13 + _tmp16
        _tmp18 = _a[0] * _b[0]
        _tmp19 = _a[1] * _b[1]
        _tmp20 = _a[2] * _b[2]
        _tmp21 = _tmp19 + _tmp20
        _tmp22 = _tmp18 + _tmp21
        _tmp23 = _a[3] * _b[3]
        _tmp24 = _a[1] ** 2
        _tmp25 = -2 * _tmp24
        _tmp26 = _a[2] ** 2
        _tmp27 = 1 - 2 * _tmp26
        _tmp28 = _a[0] * _a[2]
        _tmp29 = 2 * _tmp28
        _tmp30 = _a[1] * _a[3]
        _tmp31 = 2 * _tmp30
        _tmp32 = _a[0] * _a[1]
        _tmp33 = 2 * _tmp32
        _tmp34 = _a[2] * _a[3]
        _tmp35 = 2 * _tmp34
        _tmp36 = _a[0] ** 2
        _tmp37 = -2 * _tmp36
        _tmp38 = _a[1] * _a[2]
        _tmp39 = 2 * _tmp38
        _tmp40 = _a[0] * _a[3]
        _tmp41 = 2 * _tmp40
        _tmp42 = -_tmp5
        _tmp43 = -_tmp6
        _tmp44 = _tmp42 + _tmp43 + _tmp9
        _tmp45 = -_tmp13
        _tmp46 = _tmp11 + _tmp16 + _tmp45
        _tmp47 = -_tmp2
        _tmp48 = -_tmp3
        _tmp49 = -_tmp0 + _tmp1
        _tmp50 = _tmp47 + _tmp48 + _tmp49
        _tmp51 = _tmp4 + _tmp49
        _tmp52 = -_tmp18
        _tmp53 = _tmp21 + _tmp23 + _tmp52
        _tmp54 = -_tmp23
        _tmp55 = _tmp22 + _tmp54
        _tmp56 = _tmp28 + _tmp30
        _tmp57 = _tmp32 - _tmp34
        _tmp58 = _a[3] ** 2
        _tmp59 = -_tmp58
        _tmp60 = -_tmp24 + _tmp26
        _tmp61 = _tmp36 + _tmp59 + _tmp60
        _tmp62 = _tmp38 - _tmp40
        _tmp63 = 2 * _tmp62
        _tmp64 = -_tmp36
        _tmp65 = _tmp58 + _tmp60 + _tmp64
        _tmp66 = _tmp38 + _tmp40
        _tmp67 = 2 * _tmp66
        _tmp68 = _tmp19 - _tmp20
        _tmp69 = _tmp52 + _tmp54 + _tmp68
        _tmp70 = _tmp0 + _tmp1
        _tmp71 = _tmp3 + _tmp47 + _tmp70
        _tmp72 = _tmp7 + _tmp8
        _tmp73 = _tmp43 + _tmp5 + _tmp72
        _tmp74 = _tmp14 - _tmp15
        _tmp75 = _tmp11 + _tmp13 + _tmp74
        _tmp76 = _tmp24 + _tmp26
        _tmp77 = _tmp59 + _tmp64 + _tmp76
        _tmp78 = 2 * _tmp56
        _tmp79 = _tmp32 + _tmp34
        _tmp80 = _tmp28 - _tmp30
        _tmp81 = 2 * _tmp80
        _tmp82 = _tmp18 + _tmp23 + _tmp68
        _tmp83 = _tmp12 + _tmp45 + _tmp74
        _tmp84 = _tmp2 + _tmp48 + _tmp70
        _tmp85 = _tmp42 + _tmp6 + _tmp72
        _tmp86 = 2 * _tmp57
        _tmp87 = 2 * _tmp79
        _tmp88 = _tmp22 + _tmp54
        _tmp89 = _tmp10 ** 2 + _tmp17 ** 2 + _tmp50 ** 2 + _tmp55 * _tmp88
        _tmp90 = -_tmp17 * _tmp55 + _tmp17 * _tmp88
        _tmp91 = -_tmp10 * _tmp55 + _tmp10 * _tmp88
        _tmp92 = -_tmp50 * _tmp55 + _tmp50 * _tmp88
        _tmp93 = _tmp36 - 1.0 / 2.0

        # Output terms
        _res = [0.0] * 7
        _res[0] = _tmp0 - _tmp1 + _tmp4
        _res[1] = _tmp10
        _res[2] = _tmp17
        _res[3] = -_tmp22 + _tmp23
        _res[4] = (
            _a[4]
            + _b[4] * (_tmp25 + _tmp27)
            + _b[5] * (_tmp33 - _tmp35)
            + _b[6] * (_tmp29 + _tmp31)
        )
        _res[5] = (
            _a[5]
            + _b[4] * (_tmp33 + _tmp35)
            + _b[5] * (_tmp27 + _tmp37)
            + _b[6] * (_tmp39 - _tmp41)
        )
        _res[6] = (
            _a[6]
            + _b[4] * (_tmp29 - _tmp31)
            + _b[5] * (_tmp39 + _tmp41)
            + _b[6] * (_tmp25 + _tmp37 + 1)
        )
        _res_D_a = numpy.zeros((6, 6))
        _res_D_a[0, 0] = -_tmp10 * _tmp44 - _tmp17 * _tmp46 - _tmp50 * _tmp51 - _tmp53 * _tmp55
        _res_D_a[1, 0] = _tmp10 * _tmp51 - _tmp17 * _tmp53 - _tmp44 * _tmp50 + _tmp46 * _tmp55
        _res_D_a[2, 0] = _tmp10 * _tmp53 + _tmp17 * _tmp51 - _tmp44 * _tmp55 - _tmp46 * _tmp50
        _res_D_a[3, 0] = 2 * _b[5] * _tmp56 - 2 * _b[6] * _tmp57
        _res_D_a[4, 0] = _b[5] * _tmp63 + _b[6] * _tmp61
        _res_D_a[5, 0] = _b[5] * _tmp65 - _b[6] * _tmp67
        _res_D_a[0, 1] = _tmp10 * _tmp71 - _tmp17 * _tmp69 - _tmp50 * _tmp73 + _tmp55 * _tmp75
        _res_D_a[1, 1] = _tmp10 * _tmp73 + _tmp17 * _tmp75 + _tmp50 * _tmp71 + _tmp55 * _tmp69
        _res_D_a[2, 1] = -_tmp10 * _tmp75 + _tmp17 * _tmp73 - _tmp50 * _tmp69 + _tmp55 * _tmp71
        _res_D_a[3, 1] = -_b[4] * _tmp78 - _b[6] * _tmp77
        _res_D_a[4, 1] = -2 * _b[4] * _tmp62 + 2 * _b[6] * _tmp79
        _res_D_a[5, 1] = -_b[4] * _tmp65 + _b[6] * _tmp81
        _res_D_a[0, 2] = -_tmp10 * _tmp82 - _tmp17 * _tmp84 + _tmp50 * _tmp83 + _tmp55 * _tmp85
        _res_D_a[1, 2] = -_tmp10 * _tmp83 + _tmp17 * _tmp85 - _tmp50 * _tmp82 + _tmp55 * _tmp84
        _res_D_a[2, 2] = -_tmp10 * _tmp85 - _tmp17 * _tmp83 - _tmp50 * _tmp84 - _tmp55 * _tmp82
        _res_D_a[3, 2] = _b[4] * _tmp86 + _b[5] * _tmp77
        _res_D_a[4, 2] = -_b[4] * _tmp61 - _b[5] * _tmp87
        _res_D_a[5, 2] = 2 * _b[4] * _tmp66 - 2 * _b[5] * _tmp80
        _res_D_a[0, 3] = 0
        _res_D_a[1, 3] = 0
        _res_D_a[2, 3] = 0
        _res_D_a[3, 3] = 1
        _res_D_a[4, 3] = 0
        _res_D_a[5, 3] = 0
        _res_D_a[0, 4] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[3, 4] = 0
        _res_D_a[4, 4] = 1
        _res_D_a[5, 4] = 0
        _res_D_a[0, 5] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 5] = 0
        _res_D_a[4, 5] = 0
        _res_D_a[5, 5] = 1
        _res_D_b = numpy.zeros((6, 6))
        _res_D_b[0, 0] = _tmp89
        _res_D_b[1, 0] = _tmp90
        _res_D_b[2, 0] = -_tmp91
        _res_D_b[3, 0] = 0
        _res_D_b[4, 0] = 0
        _res_D_b[5, 0] = 0
        _res_D_b[0, 1] = -_tmp90
        _res_D_b[1, 1] = _tmp89
        _res_D_b[2, 1] = -_tmp92
        _res_D_b[3, 1] = 0
        _res_D_b[4, 1] = 0
        _res_D_b[5, 1] = 0
        _res_D_b[0, 2] = _tmp91
        _res_D_b[1, 2] = _tmp92
        _res_D_b[2, 2] = _tmp89
        _res_D_b[3, 2] = 0
        _res_D_b[4, 2] = 0
        _res_D_b[5, 2] = 0
        _res_D_b[0, 3] = 0
        _res_D_b[1, 3] = 0
        _res_D_b[2, 3] = 0
        _res_D_b[3, 3] = 1 - 2 * _tmp76
        _res_D_b[4, 3] = _tmp87
        _res_D_b[5, 3] = _tmp81
        _res_D_b[0, 4] = 0
        _res_D_b[1, 4] = 0
        _res_D_b[2, 4] = 0
        _res_D_b[3, 4] = _tmp86
        _res_D_b[4, 4] = -2 * _tmp26 - 2 * _tmp93
        _res_D_b[5, 4] = _tmp67
        _res_D_b[0, 5] = 0
        _res_D_b[1, 5] = 0
        _res_D_b[2, 5] = 0
        _res_D_b[3, 5] = _tmp78
        _res_D_b[4, 5] = _tmp63
        _res_D_b[5, 5] = -2 * _tmp24 - 2 * _tmp93
        return sym.Pose3.from_storage(_res), _res_D_a, _res_D_b

    @staticmethod
    def between_with_jacobians(a, b):
        # type: (sym.Pose3, sym.Pose3) -> T.Tuple[sym.Pose3, numpy.ndarray, numpy.ndarray]

        # Total ops: 237

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (90)
        _tmp0 = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0]
        _tmp1 = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1]
        _tmp2 = _a[0] * _b[1]
        _tmp3 = _a[2] * _b[3]
        _tmp4 = _a[3] * _b[2]
        _tmp5 = _a[1] * _b[0]
        _tmp6 = _a[1] * _b[1]
        _tmp7 = _a[2] * _b[2]
        _tmp8 = _a[0] * _b[0]
        _tmp9 = _a[3] * _b[3]
        _tmp10 = _a[2] ** 2
        _tmp11 = -2 * _tmp10
        _tmp12 = _a[1] ** 2
        _tmp13 = 1 - 2 * _tmp12
        _tmp14 = _tmp11 + _tmp13
        _tmp15 = _a[0] * _a[2]
        _tmp16 = 2 * _tmp15
        _tmp17 = _a[1] * _a[3]
        _tmp18 = 2 * _tmp17
        _tmp19 = _tmp16 - _tmp18
        _tmp20 = _a[0] * _a[1]
        _tmp21 = 2 * _tmp20
        _tmp22 = _a[2] * _a[3]
        _tmp23 = 2 * _tmp22
        _tmp24 = _tmp21 + _tmp23
        _tmp25 = _a[0] ** 2
        _tmp26 = -2 * _tmp25
        _tmp27 = _tmp11 + _tmp26 + 1
        _tmp28 = _a[1] * _a[2]
        _tmp29 = 2 * _tmp28
        _tmp30 = _a[0] * _a[3]
        _tmp31 = 2 * _tmp30
        _tmp32 = _tmp29 + _tmp31
        _tmp33 = _tmp21 - _tmp23
        _tmp34 = _tmp13 + _tmp26
        _tmp35 = _tmp29 - _tmp31
        _tmp36 = _tmp16 + _tmp18
        _tmp37 = _tmp2 + _tmp3 - _tmp4 - _tmp5
        _tmp38 = _tmp37 ** 2
        _tmp39 = -_tmp38
        _tmp40 = _tmp0 ** 2
        _tmp41 = _tmp6 + _tmp7 + _tmp8 + _tmp9
        _tmp42 = _tmp41 ** 2
        _tmp43 = _tmp1 ** 2
        _tmp44 = _tmp42 - _tmp43
        _tmp45 = _tmp37 * _tmp41
        _tmp46 = _tmp0 * _tmp1
        _tmp47 = _tmp1 * _tmp41
        _tmp48 = _tmp0 * _tmp37
        _tmp49 = -_tmp25
        _tmp50 = _a[3] ** 2
        _tmp51 = _tmp10 - _tmp12
        _tmp52 = _tmp49 + _tmp50 + _tmp51
        _tmp53 = _a[6] * _tmp52
        _tmp54 = _b[6] * _tmp52
        _tmp55 = 2 * _tmp28 - 2 * _tmp30
        _tmp56 = _a[5] * _tmp55
        _tmp57 = _b[5] * _tmp55
        _tmp58 = 2 * _tmp15 + 2 * _tmp17
        _tmp59 = _a[4] * _tmp58
        _tmp60 = _b[4] * _tmp58
        _tmp61 = -_tmp50
        _tmp62 = _tmp25 + _tmp51 + _tmp61
        _tmp63 = _a[5] * _tmp62
        _tmp64 = _b[5] * _tmp62
        _tmp65 = 2 * _tmp28 + 2 * _tmp30
        _tmp66 = _a[6] * _tmp65
        _tmp67 = _b[6] * _tmp65
        _tmp68 = 2 * _tmp20 - 2 * _tmp22
        _tmp69 = _a[4] * _tmp68
        _tmp70 = _b[4] * _tmp68
        _tmp71 = -_tmp40
        _tmp72 = _tmp42 + _tmp43
        _tmp73 = _tmp1 * _tmp37
        _tmp74 = _tmp0 * _tmp41
        _tmp75 = _tmp10 + _tmp12
        _tmp76 = _tmp49 + _tmp61 + _tmp75
        _tmp77 = _a[4] * _tmp76
        _tmp78 = _b[4] * _tmp76
        _tmp79 = 2 * _tmp15 - 2 * _tmp17
        _tmp80 = _a[6] * _tmp79
        _tmp81 = _b[6] * _tmp79
        _tmp82 = 2 * _tmp20 + 2 * _tmp22
        _tmp83 = _a[5] * _tmp82
        _tmp84 = _b[5] * _tmp82
        _tmp85 = 2 * _tmp75 - 1
        _tmp86 = _tmp25 - 1.0 / 2.0
        _tmp87 = 2 * _tmp10 + 2 * _tmp86
        _tmp88 = 2 * _tmp12 + 2 * _tmp86
        _tmp89 = _tmp38 + _tmp40 + _tmp72

        # Output terms
        _res = [0.0] * 7
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res[2] = -_tmp2 - _tmp3 + _tmp4 + _tmp5
        _res[3] = _tmp6 + _tmp7 + _tmp8 + _tmp9
        _res[4] = (
            -_a[4] * _tmp14
            - _a[5] * _tmp24
            - _a[6] * _tmp19
            + _b[4] * _tmp14
            + _b[5] * _tmp24
            + _b[6] * _tmp19
        )
        _res[5] = (
            -_a[4] * _tmp33
            - _a[5] * _tmp27
            - _a[6] * _tmp32
            + _b[4] * _tmp33
            + _b[5] * _tmp27
            + _b[6] * _tmp32
        )
        _res[6] = (
            -_a[4] * _tmp36
            - _a[5] * _tmp35
            - _a[6] * _tmp34
            + _b[4] * _tmp36
            + _b[5] * _tmp35
            + _b[6] * _tmp34
        )
        _res_D_a = numpy.zeros((6, 6))
        _res_D_a[0, 0] = -_tmp39 - _tmp40 - _tmp44
        _res_D_a[1, 0] = -2 * _tmp45 - 2 * _tmp46
        _res_D_a[2, 0] = -2 * _tmp47 + 2 * _tmp48
        _res_D_a[3, 0] = 0
        _res_D_a[4, 0] = -_tmp53 + _tmp54 - _tmp56 + _tmp57 - _tmp59 + _tmp60
        _res_D_a[5, 0] = -_tmp63 + _tmp64 + _tmp66 - _tmp67 + _tmp69 - _tmp70
        _res_D_a[0, 1] = 2 * _tmp45 - 2 * _tmp46
        _res_D_a[1, 1] = -_tmp39 - _tmp71 - _tmp72
        _res_D_a[2, 1] = 2 * _tmp73 + 2 * _tmp74
        _res_D_a[3, 1] = _tmp53 - _tmp54 + _tmp56 - _tmp57 + _tmp59 - _tmp60
        _res_D_a[4, 1] = 0
        _res_D_a[5, 1] = _tmp77 - _tmp78 - _tmp80 + _tmp81 - _tmp83 + _tmp84
        _res_D_a[0, 2] = 2 * _tmp47 + 2 * _tmp48
        _res_D_a[1, 2] = 2 * _tmp73 - 2 * _tmp74
        _res_D_a[2, 2] = -_tmp38 - _tmp44 - _tmp71
        _res_D_a[3, 2] = _tmp63 - _tmp64 - _tmp66 + _tmp67 - _tmp69 + _tmp70
        _res_D_a[4, 2] = -_tmp77 + _tmp78 + _tmp80 - _tmp81 + _tmp83 - _tmp84
        _res_D_a[5, 2] = 0
        _res_D_a[0, 3] = 0
        _res_D_a[1, 3] = 0
        _res_D_a[2, 3] = 0
        _res_D_a[3, 3] = _tmp85
        _res_D_a[4, 3] = -_tmp68
        _res_D_a[5, 3] = -_tmp58
        _res_D_a[0, 4] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[3, 4] = -_tmp82
        _res_D_a[4, 4] = _tmp87
        _res_D_a[5, 4] = -_tmp55
        _res_D_a[0, 5] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 5] = -_tmp79
        _res_D_a[4, 5] = -_tmp65
        _res_D_a[5, 5] = _tmp88
        _res_D_b = numpy.zeros((6, 6))
        _res_D_b[0, 0] = _tmp89
        _res_D_b[1, 0] = 0
        _res_D_b[2, 0] = 0
        _res_D_b[3, 0] = 0
        _res_D_b[4, 0] = 0
        _res_D_b[5, 0] = 0
        _res_D_b[0, 1] = 0
        _res_D_b[1, 1] = _tmp89
        _res_D_b[2, 1] = 0
        _res_D_b[3, 1] = 0
        _res_D_b[4, 1] = 0
        _res_D_b[5, 1] = 0
        _res_D_b[0, 2] = 0
        _res_D_b[1, 2] = 0
        _res_D_b[2, 2] = _tmp89
        _res_D_b[3, 2] = 0
        _res_D_b[4, 2] = 0
        _res_D_b[5, 2] = 0
        _res_D_b[0, 3] = 0
        _res_D_b[1, 3] = 0
        _res_D_b[2, 3] = 0
        _res_D_b[3, 3] = -_tmp85
        _res_D_b[4, 3] = _tmp68
        _res_D_b[5, 3] = _tmp58
        _res_D_b[0, 4] = 0
        _res_D_b[1, 4] = 0
        _res_D_b[2, 4] = 0
        _res_D_b[3, 4] = _tmp82
        _res_D_b[4, 4] = -_tmp87
        _res_D_b[5, 4] = _tmp55
        _res_D_b[0, 5] = 0
        _res_D_b[1, 5] = 0
        _res_D_b[2, 5] = 0
        _res_D_b[3, 5] = _tmp79
        _res_D_b[4, 5] = _tmp65
        _res_D_b[5, 5] = -_tmp88
        return sym.Pose3.from_storage(_res), _res_D_a, _res_D_b
