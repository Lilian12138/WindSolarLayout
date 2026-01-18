import numpy as np
import geopandas as gpd

# ===========================
# 1. Weibull 风速分布
# ===========================
def weibull_pdf(v, A, k):
    v = np.asarray(v, dtype=float)
    A = float(A); k = float(k)
    return (k / A) * (v / A) ** (k - 1) * np.exp(-(v / A) ** k)

# ===========================
# 2. 8MW 风机功率曲线（向量化）
# ===========================
def power_curve_8MW_vec(v, v_ci=3.0, v_r=11.0, v_co=25.0, P_r=8.0):
    v = np.asarray(v, dtype=float)
    p = np.zeros_like(v)

    # v_ci <= v < v_r
    m1 = (v >= v_ci) & (v < v_r)
    p[m1] = P_r * (v[m1]**3 - v_ci**3) / (v_r**3 - v_ci**3)

    # v_r <= v < v_co
    m2 = (v >= v_r) & (v < v_co)
    p[m2] = P_r

    return p  # MW

# ===========================
# 3. 年发电量计算（AEP）
# ===========================
def calc_AEP_from_AK(A, k, v_max=50.0, dv=0.05, P_r=8.0):
    """
    返回：年发电量 (MWh/yr) 和 容量因子 CF
    关键：对 [0, v_max] 的截断 pdf 做归一化，避免系统性低估
    """
    # 基本校验
    if A is None or k is None:
        return np.nan, np.nan
    if (A <= 0) or (k <= 0):
        return np.nan, np.nan

    v = np.arange(0.0, v_max + dv, dv)

    pdf = weibull_pdf(v, A, k)

    # 截断归一化：确保积分区间概率为 1
    mass = np.trapz(pdf, v)
    if mass <= 0 or not np.isfinite(mass):
        return np.nan, np.nan
    pdf = pdf / mass

    power = power_curve_8MW_vec(v, P_r=P_r)  # MW

    # 期望功率 (MW)
    P_mean = np.trapz(power * pdf, v)

    # 年发电量 (MWh)
    AEP = 8760.0 * P_mean

    # 容量因子
    CF = P_mean / P_r

    return AEP, CF

if __name__ == '__main__':
    in_shp = r"E:\Tibet\processing\Output\windturbine_merge_WGS84.shp"
    out_shp = r"E:\Tibet\processing\Output\windturbine_merge_WGS84_AEP.shp"

    gdf = gpd.read_file(in_shp)

    # 计算
    res = gdf.apply(lambda r: calc_AEP_from_AK(r.get('A'), r.get('k')), axis=1)
    gdf['AEP'] = [x[0] for x in res]  # MWh/yr
    gdf['CF']  = [x[1] for x in res]  # 0-1

    # 保存
    gdf.to_file(out_shp, encoding='utf-8')

    # 简单输出前几条检查
    print(gdf[['A', 'k', 'AEP', 'CF']].head())
