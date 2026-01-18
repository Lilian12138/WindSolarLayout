import math
import json
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from shapely.geometry import Polygon, box, mapping
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt


# -----------------------------
# 1) 太阳几何：给定经纬度与时间（UTC+8），计算太阳高度角（近似足够工程排距）
#    若你要更高精度，可替换为 pvlib.solarposition
# -----------------------------
def solar_elevation_deg(lat_deg: float, lon_deg: float,
                        year: int, month: int, day: int,
                        hour_local: int, minute_local: int,
                        tz_offset_hours: int = 8) -> float:
    """
    计算太阳高度角（degrees）。使用简化 NOAA 近似。
    输入为当地时间（默认中国 UTC+8）。
    """
    # --- Julian day / fractional year
    # Convert local time to UTC
    hour_utc = hour_local - tz_offset_hours + minute_local / 60.0
    # Day of year
    import datetime as dt
    d = dt.date(year, month, day)
    doy = d.timetuple().tm_yday

    gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (hour_utc - 12) / 24.0)

    # Equation of time (minutes)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))

    # Solar declination (radians)
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma) + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))

    # Time offset (minutes)
    time_offset = eqtime + 4.0 * lon_deg - 60.0 * tz_offset_hours

    # True solar time (minutes)
    tst = (hour_local * 60.0 + minute_local + time_offset) % 1440.0

    # Hour angle (degrees)
    ha_deg = tst / 4.0 - 180.0
    ha = math.radians(ha_deg)

    lat = math.radians(lat_deg)

    # Solar zenith
    cos_zenith = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    cos_zenith = min(1.0, max(-1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    elev = math.degrees(math.pi / 2 - zenith)
    return elev


# -----------------------------
# 2) 投影：经纬度 → UTM（米），便于做几何排布与间距
# -----------------------------
def utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    epsg = 32600 + zone if is_northern else 32700 + zone
    return CRS.from_epsg(epsg)


def lonlat_polygon_to_utm(poly_lonlat: List[Tuple[float, float]]):
    """输入 [(lon,lat), ...] 返回 (poly_utm, transformer_fwd, transformer_inv, utm_crs)."""
    lon0, lat0 = poly_lonlat[0]
    utm_crs = utm_crs_from_lonlat(lon0, lat0)
    wgs84 = CRS.from_epsg(4326)
    fwd = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
    inv = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

    xs, ys = zip(*[fwd.transform(lon, lat) for lon, lat in poly_lonlat])
    return Polygon(list(zip(xs, ys))), fwd, inv, utm_crs


# -----------------------------
# 3) 排布参数
# -----------------------------
@dataclass
class FixedPVLayoutParams:
    # 单块组件尺寸（米）
    module_length_m: float = 1.956   # 竖向长度（沿组件长边）
    module_width_m: float = 0.992    # 竖向宽度（沿组件短边）

    # Table 组成：沿行方向的组件数（东西向），沿深度方向的组件数（南北向）
    modules_along_row: int = 22      # 一排 table 中组件数量（可改）
    modules_in_depth: int = 4        # 例如 2P（两排竖装）

    # 组件摆放方向：True=竖装(长边沿深度)，False=横装(长边沿行向)
    portrait: bool = True

    # 固定倾角与朝向
    tilt_deg: float = 25.0
    # 地面电站固定式通常面向正南；这里 row 方向假设东西向，法向朝南
    # 若你希望顺地块走向，可把 row_azimuth_deg 设为你的行方向方位角
    row_azimuth_deg: float = 90.0  # 行方向方位角：90 表示东西向（x轴）

    # 退距/间隙（米）
    boundary_buffer_m: float = 1  # 与边界退距
    obstacle_buffer_m: float = 2.0  # 与障碍退距
    gap_between_tables_x_m: float = 0.5  # 同行 table 间隙
    maintenance_extra_y_m: float = 0.8   # 行间额外运维间隙（叠加在阴影排距上）

    # 排距控制：选择冬至某时刻的太阳高度角（可按项目标准调整）
    control_date: Tuple[int, int, int] = (2025, 12, 21)  # 年月日：冬至附近
    control_time_local: Tuple[int, int] = (10, 0)        # 当地时刻（10:00 常用）
    tz_offset: int = 8

    # 是否输出为 WGS84（经纬度）
    export_lonlat: bool = True

    module_power_w = 300


# -----------------------------
# 4) 计算 table 尺寸、行间距（row pitch）
# -----------------------------
def table_dimensions(params: FixedPVLayoutParams) -> Tuple[float, float]:
    """
    返回 (table_width_x, table_depth_y) in meters，
    其中 x 为行方向（东西向），y 为深度方向（南北向）。
    """
    if params.portrait:
        # 竖装：深度用 length，行向用 width
        depth_per_module = params.module_length_m
        width_per_module = params.module_width_m
    else:
        # 横装：深度用 width，行向用 length
        depth_per_module = params.module_width_m
        width_per_module = params.module_length_m

    table_width = params.modules_along_row * width_per_module
    table_depth = params.modules_in_depth * depth_per_module
    return table_width, table_depth


def row_pitch_m(lat: float, lon: float, params: FixedPVLayoutParams) -> float:
    """
    简化行间距：row_pitch = 投影深度 + (高度差)/tan(太阳高度角) + 运维间隙
    投影深度 = table_depth * cos(tilt)
    高度差 = table_depth * sin(tilt)
    """
    table_w, table_d = table_dimensions(params)
    tilt = math.radians(params.tilt_deg)

    y_proj = table_d * math.cos(tilt)
    height_diff = table_d * math.sin(tilt)

    y, m, d = params.control_date
    hh, mm = params.control_time_local
    elev = solar_elevation_deg(lat, lon, y, m, d, hh, mm, tz_offset_hours=params.tz_offset)

    # 防止极端导致 tan 很小
    elev = max(3.0, elev)
    pitch = y_proj + height_diff / math.tan(math.radians(elev)) + params.maintenance_extra_y_m
    return pitch


# -----------------------------
# 5) 核心排布：旋转到“行方向坐标系” → 网格放置矩形 → 反旋转回原坐标
# -----------------------------
def generate_fixed_layout(buildable_lonlat: List[Tuple[float, float]],
                          obstacles_lonlat: Optional[List[List[Tuple[float, float]]]],
                          params: FixedPVLayoutParams):
    # 投影到 UTM
    site_poly_utm, fwd, inv, utm_crs = lonlat_polygon_to_utm(buildable_lonlat)

    # 障碍物（可选）
    obstacles_utm = []
    if obstacles_lonlat:
        for obs in obstacles_lonlat:
            obs_poly, _, _, _ = lonlat_polygon_to_utm(obs)
            obstacles_utm.append(obs_poly)

    # 退距：可用区域 = 场地 - 障碍（均做 buffer）
    work_area = site_poly_utm.buffer(-params.boundary_buffer_m)
    if work_area.is_empty:
        raise ValueError("边界退距后可用区域为空，请减小 boundary_buffer_m 或检查边界多边形。")

    if obstacles_utm:
        obs_union = unary_union([o.buffer(params.obstacle_buffer_m) for o in obstacles_utm])
        work_area = work_area.difference(obs_union)
        if work_area.is_empty:
            raise ValueError("扣除障碍退距后可用区域为空，请检查 obstacle_buffer_m 与障碍范围。")

    # 旋转到排布坐标系：让行方向与 x 轴对齐
    # row_azimuth_deg=90 表示行方向东西向；旋转角度 = -(row_azimuth_deg - 90)
    # 这样 row_azimuth=90 时不旋转；row_azimuth=0(南北向) 会旋转 -(-90)=+90
    theta = -(params.row_azimuth_deg - 90.0)
    work_rot = rotate(work_area, theta, origin='centroid', use_radians=False)

    # table 尺寸与 pitch
    # 用场地中心点经纬度估算太阳高度角（也可改成用站址代表点）
    lon0, lat0 = buildable_lonlat[0]
    pitch = row_pitch_m(lat0, lon0, params)
    table_w, table_d = table_dimensions(params)

    step_x = table_w + params.gap_between_tables_x_m
    step_y = pitch

    minx, miny, maxx, maxy = work_rot.bounds

    placements = []
    idx = 0

    # 从左下角开始铺
    y = miny
    while y + table_d <= maxy:
        x = minx
        while x + table_w <= maxx:
            rect = box(x, y, x + table_w, y + table_d)
            # 仅保留完全在可用区域内的 rect（你也可以改成 area_ratio 门槛）
            if work_rot.contains(rect):
                # 旋转回原坐标系
                rect_back = rotate(rect, -theta, origin=work_area.centroid, use_radians=False)
                idx += 1
                placements.append((idx, rect_back))
            x += step_x
        y += step_y

    return {
        "utm_crs": utm_crs.to_string(),
        "work_area_utm": work_area,
        "placements_utm": placements,
        "inv_transformer": inv,
        "params": params,
    }


# -----------------------------
# 6) 导出：GeoJSON / CSV（中心点）+ 简单可视化
# -----------------------------
def export_geojson(result, out_geojson_path: str, out_csv_path: str):
    inv = result["inv_transformer"]
    params = result["params"]

    features = []
    rows = []

    for idx, poly_utm in result["placements_utm"]:
        if params.export_lonlat:
            # 将 polygon 顶点转回经纬度
            coords_utm = list(poly_utm.exterior.coords)
            coords_ll = [inv.transform(x, y) for x, y in coords_utm]  # returns (lon, lat)
            poly = Polygon(coords_ll)
            cx, cy = poly.centroid.x, poly.centroid.y
        else:
            poly = poly_utm
            cx, cy = poly.centroid.x, poly.centroid.y

        features.append({
            "type": "Feature",
            "properties": {"id": idx},
            "geometry": mapping(poly)
        })
        rows.append([idx, cx, cy])

    gj = {"type": "FeatureCollection", "features": features}
    with open(out_geojson_path, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "center_x_lon_or_m", "center_y_lat_or_m"])
        w.writerows(rows)


def plot_layout(result, buildable_lonlat: List[Tuple[float, float]], obstacles_lonlat=None):
    # 绘制在 UTM 下（米），直观
    site_poly_utm, _, _, _ = lonlat_polygon_to_utm(buildable_lonlat)
    work_area = result["work_area_utm"]

    fig, ax = plt.subplots()

    x, y = site_poly_utm.exterior.xy
    ax.plot(x, y, linewidth=2)

    x, y = work_area.exterior.xy
    ax.plot(x, y, linewidth=2)

    if obstacles_lonlat:
        for obs in obstacles_lonlat:
            obs_poly_utm, _, _, _ = lonlat_polygon_to_utm(obs)
            ox, oy = obs_poly_utm.exterior.xy
            ax.plot(ox, oy, linewidth=2)

    for _, poly in result["placements_utm"]:
        px, py = poly.exterior.xy
        ax.plot(px, py, linewidth=0.8)

    ax.set_aspect("equal", "box")
    ax.set_title("Fixed PV Layout (UTM meters): site / work_area / tables")
    plt.show()


# -----------------------------
# 7) 示例用法：把你的边界经纬度点填进去即可
# -----------------------------
if __name__ == "__main__":
    # 你的可开发范围（经纬度，多边形闭合点可不重复，shapely 会自动闭合）
    buildable = [
        (112.0000, 33.0000),
        (112.0100, 33.0005),
        (112.0120, 32.9950),
        (112.0020, 32.9940),
        (111.9990, 32.9970),
    ]

    # 可选：障碍物（多个多边形）
    obstacles = None
    # obstacles = [
    #     [(112.005, 32.998), (112.006, 32.998), (112.006, 32.997), (112.005, 32.997)]
    # ]

    # 参数：中部地区固定式常用“朝南+倾角接近纬度或略小”
    p = FixedPVLayoutParams(
        tilt_deg=25.0,
        row_azimuth_deg=90.0,          # 行方向东西向
        boundary_buffer_m=2.0,
        modules_along_row=28,
        modules_in_depth=2,
        portrait=True,
        control_date=(2025, 12, 21),
        control_time_local=(10, 0),
        maintenance_extra_y_m=0.8,
        export_lonlat=True
    )

    res = generate_fixed_layout(buildable, obstacles, p)

    # 输出
    export_geojson(res, "pv_tables.geojson", "pv_tables_centers.csv")

    # 画图
    plot_layout(res, buildable, obstacles)

    print(f"Generated tables: {len(res['placements_utm'])}")
    print("GeoJSON: pv_tables.geojson")
    print("CSV: pv_tables_centers.csv")
