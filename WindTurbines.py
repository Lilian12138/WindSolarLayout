import os
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep

# -----------------------------
# 参数区（按需修改）
# -----------------------------
IN_SHP = r"E:\Tibet\processing\Scratch\gisfiles\fishnet_0p22_0p22EuqalArea.shp"
DIR_FIELD = "Dir"

D = 200.0  # meters

# 你说“dir方向3D×8D间距”：这里默认解释为
#   - 沿 dir 方向：8D
#   - 垂直 dir 方向：3D
# 如需相反，把两行交换即可
SPACING_ALONG_DIR = 5.0 * D      # along dir axis
SPACING_CROSS_DIR = 2.5 * D      # perpendicular to dir axis

OUT_SHP = r"E:\Tibet\processing\Scratch\gisfiles\fishnet_dir_grid_points_D200_terrain.shp"
# OUT_GEOJSON = r"E:\Tibet\processing\Scratch\gisfiles\fishnet_dir_grid_points_D200.geojson"

# 数值容差，避免边界点丢失
EPS = 1e-9

def unit_vectors_from_azimuth_deg(az_deg: float):
    """
    GIS 方位角：0°北，90°东，顺时针。
    坐标系：x=东向，y=北向（投影平面）。
    返回：
      u: 沿 dir 的单位向量
      v: 垂直 dir（右侧）单位向量
    """
    a = math.radians(az_deg)
    ux = math.sin(a)
    uy = math.cos(a)
    # v 为 u 逆时针旋转 90°（也可以顺/逆，网格等价）
    vx = -uy
    vy = ux
    return (ux, uy), (vx, vy)

def dot(x, y, vx, vy):
    return x * vx + y * vy

def generate_points_in_polygon(poly, az_deg, ds, dt):
    """
    在单个 polygon 内生成以 az_deg 定向的网格点。
    ds: 沿 dir 方向间距
    dt: 垂直 dir 方向间距
    """
    (ux, uy), (vx, vy) = unit_vectors_from_azimuth_deg(az_deg)

    minx, miny, maxx, maxy = poly.bounds

    # 用 bbox 四角估计在 (s,t) 坐标系下的范围
    corners = [
        (minx, miny),
        (minx, maxy),
        (maxx, miny),
        (maxx, maxy),
    ]

    s_vals = [dot(x, y, ux, uy) for x, y in corners]
    t_vals = [dot(x, y, vx, vy) for x, y in corners]

    smin, smax = min(s_vals), max(s_vals)
    tmin, tmax = min(t_vals), max(t_vals)

    # 对齐到网格（向下/向上取整到间距倍数）
    s0 = math.floor((smin - EPS) / ds) * ds
    s1 = math.ceil((smax + EPS) / ds) * ds
    t0 = math.floor((tmin - EPS) / dt) * dt
    t1 = math.ceil((tmax + EPS) / dt) * dt

    # 生成网格
    s_arr = np.arange(s0, s1 + ds * 0.5, ds)
    t_arr = np.arange(t0, t1 + dt * 0.5, dt)

    ppoly = prep(poly)
    pts = []

    # 将 (s,t) 反变换回 (x,y)：
    # [x,y] = s*u + t*v
    for s in s_arr:
        for t in t_arr:
            x = s * ux + t * vx
            y = s * uy + t * vy
            pt = Point(x, y)
            if ppoly.contains(pt) or poly.touches(pt):
                pts.append(pt)

    return pts

def main():
    gdf = gpd.read_file(IN_SHP)
    if gdf.empty:
        raise ValueError("输入 shp 为空。")

    if DIR_FIELD not in gdf.columns:
        raise ValueError(f"找不到字段 {DIR_FIELD}，当前字段：{list(gdf.columns)}")

    # 仅处理面
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise ValueError("输入图层没有 Polygon/MultiPolygon 要素。")

    out_records = []
    out_geoms = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        az = row[DIR_FIELD]

        if az is None or (isinstance(az, float) and math.isnan(az)):
            continue

        try:
            az = float(az)
        except Exception:
            continue

        # MultiPolygon 逐个处理
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]

        cell_points = []
        for poly in polys:
            cell_points.extend(
                generate_points_in_polygon(poly, az, SPACING_ALONG_DIR, SPACING_CROSS_DIR)
            )

        # 输出点（保留网格ID、dir）
        for k, pt in enumerate(cell_points):
            out_geoms.append(pt)
            out_records.append({
                "src_fid": int(idx),
                "dir": az,
                "pidx": int(k),
                "ds_m": float(SPACING_ALONG_DIR),
                "dt_m": float(SPACING_CROSS_DIR),
            })

    out_gdf = gpd.GeoDataFrame(out_records, geometry=out_geoms, crs=gdf.crs)

    # 写出
    os.makedirs(os.path.dirname(OUT_SHP), exist_ok=True)
    out_gdf.to_file(OUT_SHP, driver="ESRI Shapefile", encoding="UTF-8")
    print(f"Done. points={len(out_gdf)}")
    print(f"Output: {OUT_SHP}")

    # 如需 geojson：
    # out_gdf.to_file(OUT_GEOJSON, driver="GeoJSON")
    # print(f"Output: {OUT_GEOJSON}")

if __name__ == "__main__":
    main()
