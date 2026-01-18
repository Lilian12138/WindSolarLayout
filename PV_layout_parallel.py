import os
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely import wkb

from solar import *


def _worker(task):
    """
    task: (geom_wkb, name, IDD, out_dir)
    """
    geom_wkb, name, IDD, out_dir, OPTA = task

    # 反序列化几何
    geom = wkb.loads(geom_wkb)

    # 兼容 Polygon / MultiPolygon：MultiPolygon 取最大面（可按你业务改成遍历）
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda p: p.area)

    # 提取外环坐标
    coords = list(geom.exterior.coords)
    buildable = [(round(x, 6), round(y, 6)) for x, y in coords]

    obstacles = None

    p = FixedPVLayoutParams(
        module_length_m=1.956,
        module_width_m=0.992,
        modules_in_depth=4,
        modules_along_row=22,
        portrait=True,

        tilt_deg=OPTA,
        row_azimuth_deg=90.0,
        boundary_buffer_m=2.0,

        control_date=(2025, 12, 21),
        control_time_local=(10, 0),
        maintenance_extra_y_m=0.8,
        export_lonlat=True
    )

    # 生成与导出
    res = generate_fixed_layout(buildable, obstacles, p)

    geojson_out = os.path.join(out_dir, f"pv_tables_{IDD}_{name}.geojson")
    csv_out = os.path.join(out_dir, f"pv_tables_centers_{IDD}_{name}.csv")
    export_geojson(res, geojson_out, csv_out)

    return (IDD, name, len(res.get("placements_utm", [])), geojson_out, csv_out)


def main():
    path = r"E:\Tibet\processing\Scratch\gisfiles\windValidArea_Plain_intersect.shp"
    out_dir = r"E:\Tibet\processing\Scratch\gisfiles\PV"
    os.makedirs(out_dir, exist_ok=True)

    gdf = gpd.read_file(path)

    # 组装任务：用 WKB 传几何，避免多进程序列化问题
    tasks = []
    for row in gdf.itertuples(index=False):
        # 注意：itertuples 下字段名是列名；geometry 列一般叫 geometry
        geom = getattr(row, "geometry")
        name = getattr(row, "name")
        IDD = int(getattr(row, "IDD"))
        OPTA = round(getattr(row, "OPTA"),1)
        tasks.append((geom.wkb, name, IDD, out_dir, OPTA))

    total_n = len(tasks)
    print("Total features:", total_n)

    # 并发数建议：CPU 核心数或略小（避免内存爆）
    max_workers = max(1, os.cpu_count() - 1)

    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, t) for t in tasks]

        for fut in as_completed(futures):
            try:
                IDD, name, n_tbl, geojson_out, csv_out = fut.result()
                done += 1
                print(f"[OK] {done}/{total_n} IDD={IDD}, name={name}, tables={n_tbl}")
            except ValueError as e:
                # 你原逻辑里对 ValueError 是跳过
                failed += 1
                done += 1
                print(f"[SKIP(ValueError)] {done}/{total_n} {e}")
            except Exception as e:
                failed += 1
                done += 1
                print(f"[FAIL] {done}/{total_n} {type(e).__name__}: {e}")

    print(f"Finished. total={total_n}, failed={failed}")


if __name__ == "__main__":
    main()
