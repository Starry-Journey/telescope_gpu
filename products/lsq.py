"""真正有界最小二乘的公共入口（scipy.optimize.lsq_linear）。

仓库里原先有三处"假·有界最小二乘"：先 ``np.linalg.lstsq`` 求无界解，再用
``np.clip`` 把越界分量硬裁到界内。clip 只压制越界分量、不重新分配其余分量，
触界时结果既不是无界最优也不是有界最优，残差偏大；三处调用点还各自重复
这段样板。这里统一为一个入口，用 :func:`scipy.optimize.lsq_linear`
（TRF 算法）在界约束内直接求解。

符号约定（与三处调用点的原始语义一致，docstring 即契约）:

* 拟合约定 ``S u ≈ target`` —— :mod:`products.control.service` 与
  :mod:`products.sensitivity.matrix_service`（``residual = S u - target``），
  直接传 ``target``；
* 补偿约定 ``S u ≈ -target`` —— :mod:`products.pipeline.orchestrator`
  （机构命令 ``u`` 的波前增量 ``S u`` 要抵消 ``target``），调用点传
  ``-target`` 即可。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import lsq_linear

__all__ = ["bounded_least_squares"]


def bounded_least_squares(S: Any, target: Any, lo: Any, hi: Any,
                          lam: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve the box-constrained linear least-squares problem.

    求解::

        min_u  ||S u - target||^2  (+ lam * ||u||^2, 当 lam > 0)
        s.t.   lo <= u <= hi       （逐分量）

    内部用 :func:`scipy.optimize.lsq_linear`（``method="trf"``）在界约束内
    直接求解，不是"lstsq 后 clip"。``lam > 0`` 时与原调用点的 Tikhonov
    增广完全一致：对 ``[S; sqrt(lam) * I] u ≈ [target; 0]`` 求解，正则项
    本身不加界约束。宽界且无约束解落在界内时，结果与
    ``np.linalg.lstsq`` 的最小范数解一致（scipy 在该情形直接返回无界解）。

    参数:
        S: 形状 ``(m, n)`` 的灵敏度/响应矩阵；必须全部有限。
        target: 形状 ``(m,)`` 或 ``(m, k)``。二维表示 k 个独立目标，
            逐列独立求解（"2D 多目标"）。
        lo, hi: 形状 ``(n,)`` 的逐分量下/上界；可为 ``±np.inf`` 但不得含
            NaN，且要求 ``lo < hi``（逐分量严格，scipy lsq_linear 不接受
            相等界；要固定某个 DOF 请在本函数之外处理）。会转成 float ndarray。
        lam: Tikhonov 正则强度，``>= 0``；默认 0（纯正则最小二乘）。

    返回:
        ``(u, info)`` 二元组:

        * ``u``: float ndarray。``target`` 一维时形状 ``(n,)``；二维时
          形状 ``(n, k)``。恒满足 ``lo <= u <= hi``（对 ~1ulp 的数值
          噪声做了一次卫生裁剪）。
        * ``info``: dict，键:

          - ``"success"``: bool，是否收敛（lsq_linear ``status > 0``；
            多目标时为"所有列收敛"）；
          - ``"status"`` / ``"message"``: 求解器状态码/说明，多目标时
            为每列一个的列表；
          - ``"active_bounds"``: bool ndarray，形状 ``(n,)``（或二维时
            ``(n, k)``），分量是否落在下界或上界（``np.isclose`` 判定，
            与调用点既有的触界判定一致）；
          - ``"n_active"``: int，落在界上的分量总数。

    异常:
        ValueError: ``S`` 不是非空二维矩阵、``S``/``target``/界含 NaN/Inf、
        界含 NaN、``lo >= hi``、形状不匹配或 ``lam < 0``。输入非有限时
        直接快速失败（与 :func:`products.sensitivity.matrix_service.analyze`
        等调用点的既有前置校验一致），调用点如需容错应自行清洗输入。

    使用示例（补偿约定，见 orchestrator）::

        u, info = bounded_least_squares(S, -target, lo, hi)  # S u ≈ -target
        residual = target + S @ u
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.size == 0:
        raise ValueError("S must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(S)):
        raise ValueError("S contains NaN/Inf")
    m, n = S.shape

    t = np.asarray(target, dtype=float)
    if t.ndim == 1:
        t = t.reshape(-1, 1)          # 统一按列处理，返回前再压回一维
        single = True
    elif t.ndim == 2:
        single = False
    else:
        raise ValueError("target must be 1-D or 2-D")
    if t.shape[0] != m or t.shape[1] == 0:
        raise ValueError(
            f"target has shape {np.asarray(target).shape} but S is {S.shape}")
    if not np.all(np.isfinite(t)):
        raise ValueError("target contains NaN/Inf")
    if lam < 0:
        raise ValueError("lam must be >= 0")

    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if lo.size != n or hi.size != n:
        raise ValueError(
            f"bounds must have one value per control column ({n}), "
            f"got lo={lo.size}, hi={hi.size}")
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError("bounds must not contain NaN")
    if np.any(lo >= hi):
        raise ValueError(
            "bounds must satisfy lo < hi per component "
            "(scipy lsq_linear requires strict bounds)")

    # lam > 0 时复现原调用点的 Tikhonov 增广：[S; sqrt(lam) I] u ≈ [target; 0]
    if lam > 0:
        A_aug = np.vstack([S, math.sqrt(lam) * np.eye(n)])
    else:
        A_aug = S

    k = t.shape[1]
    u = np.empty((n, k), dtype=float)
    statuses, messages, successes = [], [], []
    for j in range(k):
        b = t[:, j]
        if lam > 0:
            b = np.concatenate([b, np.zeros(n)])
        res = lsq_linear(A_aug, b, bounds=(lo, hi))
        # 清掉 ~1ulp 的越界数值噪声，保证契约 lo <= u <= hi 严格成立
        u[:, j] = np.clip(res.x, lo, hi)
        successes.append(bool(res.success))
        statuses.append(int(res.status))
        messages.append(str(res.message))

    # 触界判定与调用点既有逻辑一致（np.isclose 到 lo/hi）
    active = (np.isclose(u, lo[:, None]) | np.isclose(u, hi[:, None]))

    if single:
        u = u[:, 0]
        active = active[:, 0]
        info: Dict[str, Any] = {
            "success": successes[0],
            "status": statuses[0],
            "message": messages[0],
            "active_bounds": active,
            "n_active": int(np.count_nonzero(active)),
        }
    else:
        info = {
            "success": all(successes),
            "status": statuses,
            "message": messages,
            "active_bounds": active,
            "n_active": int(np.count_nonzero(active)),
        }
    return u, info
