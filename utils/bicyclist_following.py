from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import polars as pl

# Reuse your class IDs
PERSON_CLASS = 0
BICYCLE_CLASS = 1


@dataclass
class FollowingParams:
    # Motion / geometry
    speed_min: float = 0.8          # pixels per frame (tune)
    dir_cos_thresh: float = 0.75    # direction alignment between follower and leader
    rel_speed_thresh: float = 0.6   # |vL - vF| / vF

    # Distance gates in units of follower "size" (height)
    long_min: float = 0.8           # leader must be at least 0.8*h ahead
    long_max: float = 10.0          # and at most 10*h ahead
    lat_max: float = 0.9            # lateral offset at most 0.9*h

    # Persistence
    min_follow_frames: int = 10     # minimum frames to qualify as following episode
    gap_allow: int = 1              # allow up to this many missing frames inside an episode

    eps: float = 1e-9


class CyclistFollowing:
    """
    Pipeline:
      1) identify_bicyclists: person_id -> bicycle_id (using Algorithm.classify_rider_type)
      2) build_cyclist_states: per-frame cyclist state (x,y, speed, direction)
      3) detect_following_episodes: leader/follower episodes with metrics
    """

    def __init__(self, algorithm) -> None:
        self.algorithm = algorithm  # your Algorithm instance/class

    @staticmethod
    def _dedup_per_frame(df: pl.DataFrame) -> pl.DataFrame:
        # Use your dedup logic if present; fall back to simple unique
        if "confidence" not in df.columns:
            return df.unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")

        return (
            df.sort(
                ["yolo-id", "unique-id", "frame-count", "confidence"],
                descending=[False, False, False, True],
            )
            .unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
        )

    def identify_bicyclists(
        self,
        df: pl.DataFrame,
        *,
        min_shared_frames: int = 4,
        score_thresh: float = 0.0,
        person_class: int = PERSON_CLASS,
        bicycle_class: int = BICYCLE_CLASS,
    ) -> pl.DataFrame:
        """
        Returns mapping table:
          person_id (cyclist_id), bicycle_id, score, shared_frames
        """
        df = self._dedup_per_frame(df)

        person_ids = (
            df.filter(pl.col("yolo-id") == person_class)
              .select("unique-id")
              .unique()
              .to_series()
              .to_list()
        )

        rows = []
        for pid in person_ids:
            res = self.algorithm.classify_rider_type(
                df,
                pid,
                min_shared_frames=min_shared_frames,
                person_class=person_class,
                bicycle_class=bicycle_class,
            )
            if (
                bool(res.get("is_rider"))
                and res.get("rider_type") == "bicycle"
                and res.get("role") == "rider"
                and res.get("vehicle_id") is not None
                and float(res.get("score", 0.0)) >= score_thresh
            ):
                rows.append(
                    {
                        "cyclist_id": int(pid),
                        "bicycle_id": int(res["vehicle_id"]),
                        "score": float(res["score"]),
                        "shared_frames": int(res.get("shared_frames", 0)),
                    }
                )

        if not rows:
            return pl.DataFrame(
                {
                    "cyclist_id": pl.Series([], dtype=pl.Int64),
                    "bicycle_id": pl.Series([], dtype=pl.Int64),
                    "score": pl.Series([], dtype=pl.Float64),
                    "shared_frames": pl.Series([], dtype=pl.Int64),
                }
            )

        return pl.DataFrame(rows)

    @staticmethod
    def build_cyclist_states(
        df: pl.DataFrame,
        cyclist_map: pl.DataFrame,
        *,
        prefer_vehicle_center: bool = True,
        person_class: int = PERSON_CLASS,
        bicycle_class: int = BICYCLE_CLASS,
        eps: float = 1e-9,
    ) -> pl.DataFrame:
        """
        Produces per-frame cyclist states with velocity and unit direction.

        Output columns:
          frame-count, cyclist_id, bicycle_id, x, y, w, h, speed, dirx, diry
        """
        df = CyclistFollowing._dedup_per_frame(df)

        if cyclist_map.height == 0:
            return pl.DataFrame()

        # Person detections for cyclist_ids
        p = (
            df.filter(pl.col("yolo-id") == person_class)
              .join(cyclist_map.select(["cyclist_id"]), left_on="unique-id", right_on="cyclist_id", how="inner")
              .select(
                  [
                      pl.col("frame-count"),
                      pl.col("unique-id").alias("cyclist_id"),
                      pl.col("x-center").alias("px"),
                      pl.col("y-center").alias("py"),
                      pl.col("width").alias("pw"),
                      pl.col("height").alias("ph"),
                  ]
              )
        )

        # Bicycle detections for bicycle_ids
        b = (
            df.filter(pl.col("yolo-id") == bicycle_class)
              .join(cyclist_map.select(["cyclist_id", "bicycle_id"]), left_on="unique-id", right_on="bicycle_id", how="inner")  # noqa: E501
              .select(
                  [
                      pl.col("frame-count"),
                      pl.col("cyclist_id"),
                      pl.col("unique-id").alias("bicycle_id"),
                      pl.col("x-center").alias("bx"),
                      pl.col("y-center").alias("by"),
                      pl.col("width").alias("bw"),
                      pl.col("height").alias("bh"),
                  ]
              )
        )

        # Join per frame. If prefer_vehicle_center, use bicycle center when present else person.
        j = (
            p.join(b, on=["frame-count", "cyclist_id"], how="left")
             .join(cyclist_map.select(["cyclist_id", "bicycle_id"]), on="cyclist_id", how="left")
        )

        if prefer_vehicle_center:
            x = pl.when(pl.col("bx").is_not_null()).then(pl.col("bx")).otherwise(pl.col("px")).alias("x")
            y = pl.when(pl.col("by").is_not_null()).then(pl.col("by")).otherwise(pl.col("py")).alias("y")
            w = pl.when(pl.col("bw").is_not_null()).then(pl.col("bw")).otherwise(pl.col("pw")).alias("w")
            h = pl.when(pl.col("bh").is_not_null()).then(pl.col("bh")).otherwise(pl.col("ph")).alias("h")
        else:
            x, y, w, h = pl.col("px").alias("x"), pl.col("py").alias("y"), pl.col("pw").alias("w"), pl.col("ph").alias("h")  # noqa: E501

        states = (
            j.select(
                [
                    pl.col("frame-count"),
                    pl.col("cyclist_id"),
                    pl.col("bicycle_id"),
                    x, y, w, h,
                ]
            )
            .sort(["cyclist_id", "frame-count"])
            .with_columns(
                [
                    (pl.col("x") - pl.col("x").shift(1)).over("cyclist_id").alias("dx"),
                    (pl.col("y") - pl.col("y").shift(1)).over("cyclist_id").alias("dy"),
                ]
            )
            .with_columns(
                [
                    (pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt().alias("speed"),
                ]
            )
            .with_columns(
                [
                    (pl.col("dx") / (pl.col("speed") + eps)).alias("dirx"),
                    (pl.col("dy") / (pl.col("speed") + eps)).alias("diry"),
                ]
            )
            .drop(["dx", "dy"])
        )

        return states

    @staticmethod
    def detect_following_episodes(
        states: pl.DataFrame,
        *,
        params: FollowingParams = FollowingParams(),
        fps: Optional[float] = None,
    ) -> pl.DataFrame:
        """
        Returns following episodes with leader/follower labels.

        Output columns (episodes):
          follower_id, leader_id, start_frame, end_frame, n_frames,
          mean_long, mean_lat, mean_dist, mean_dir_cos, mean_rel_speed,
          mean_time_headway_frames, mean_time_headway_s (if fps provided)
        """
        if states.height == 0:
            return pl.DataFrame()

        # Keep only frames where direction is meaningful (speed >= speed_min)
        s = states.filter(pl.col("speed") >= params.speed_min)

        if s.height == 0:
            return pl.DataFrame()

        frames = (
            s.select("frame-count")
             .unique()
             .sort("frame-count")
             .to_series()
             .to_list()
        )

        # Frame-wise assignments: follower -> leader
        assign_rows: List[dict] = []

        for f in frames:
            sf = s.filter(pl.col("frame-count") == f)
            if sf.height < 2:
                continue

            # Extract arrays
            follower_ids = sf.get_column("cyclist_id").to_numpy()
            x = sf.get_column("x").to_numpy()
            y = sf.get_column("y").to_numpy()
            h = sf.get_column("h").to_numpy()
            sp = sf.get_column("speed").to_numpy()
            dirx = sf.get_column("dirx").to_numpy()
            diry = sf.get_column("diry").to_numpy()

            # For direction similarity between i and j: cos = di · dj
            # Precompute dir matrix components
            for i in range(sf.height):
                di = np.array([dirx[i], diry[i]], dtype=float)
                if not np.isfinite(di).all():
                    continue

                # relative vectors to all others
                rx = x - x[i]
                ry = y - y[i]
                dist = np.sqrt(rx * rx + ry * ry)

                # follower i forward axis
                longi = rx * di[0] + ry * di[1]
                lati = np.abs(rx * di[1] - ry * di[0])  # |cross(rel, dir)| in 2D

                # direction cosine between i and each j
                cos_dir = dirx * di[0] + diry * di[1]

                # relative speed
                rel_sp = np.abs(sp - sp[i]) / max(sp[i], params.eps)

                size_i = max(float(h[i]), params.eps)

                cand = (
                    (follower_ids != follower_ids[i]) &
                    (cos_dir >= params.dir_cos_thresh) &
                    (longi >= params.long_min * size_i) &
                    (longi <= params.long_max * size_i) &
                    (lati <= params.lat_max * size_i) &
                    (rel_sp <= params.rel_speed_thresh)
                )

                if not np.any(cand):
                    continue

                # pick nearest leader ahead (smallest longitudinal gap)
                cand_idx = np.where(cand)[0]
                j_best = cand_idx[np.argmin(longi[cand_idx])]

                # time headway in frames: longitudinal gap / follower speed
                thw_frames = float(longi[j_best] / max(sp[i], params.eps))

                assign_rows.append(
                    {
                        "frame-count": int(f),
                        "follower_id": int(follower_ids[i]),
                        "leader_id": int(follower_ids[j_best]),
                        "long_gap": float(longi[j_best]),
                        "lat_gap": float(lati[j_best]),
                        "dist": float(dist[j_best]),
                        "dir_cos": float(cos_dir[j_best]),
                        "rel_speed": float(rel_sp[j_best]),
                        "thw_frames": thw_frames,
                    }
                )

        if not assign_rows:
            return pl.DataFrame()

        assigns = pl.DataFrame(assign_rows).sort(["follower_id", "frame-count"])

        # Merge into episodes per follower where leader stays constant and frames are consecutive (allowing small gaps)
        episodes: List[dict] = []
        for follower_id, g in assigns.group_by("follower_id", maintain_order=True):
            g = g.sort("frame-count")
            frames_g = g.get_column("frame-count").to_numpy()
            leaders_g = g.get_column("leader_id").to_numpy()

            long_g = g.get_column("long_gap").to_numpy()
            lat_g = g.get_column("lat_gap").to_numpy()
            dist_g = g.get_column("dist").to_numpy()
            cos_g = g.get_column("dir_cos").to_numpy()
            relsp_g = g.get_column("rel_speed").to_numpy()
            thw_g = g.get_column("thw_frames").to_numpy()

            start = 0
            for k in range(1, len(frames_g) + 1):
                end_of_run = False
                if k == len(frames_g):
                    end_of_run = True
                else:
                    gap = frames_g[k] - frames_g[k - 1]
                    if (leaders_g[k] != leaders_g[k - 1]) or (gap > (params.gap_allow + 1)):
                        end_of_run = True

                if end_of_run:
                    seg_slice = slice(start, k)
                    n = k - start
                    if n >= params.min_follow_frames:
                        leader = int(leaders_g[start])
                        seg_frames = frames_g[seg_slice]
                        episodes.append(
                            {
                                "follower_id": int(follower_id),  # type: ignore
                                "leader_id": leader,
                                "start_frame": int(seg_frames[0]),
                                "end_frame": int(seg_frames[-1]),
                                "n_frames": int(n),
                                "mean_long": float(np.mean(long_g[seg_slice])),
                                "mean_lat": float(np.mean(lat_g[seg_slice])),
                                "mean_dist": float(np.mean(dist_g[seg_slice])),
                                "mean_dir_cos": float(np.mean(cos_g[seg_slice])),
                                "mean_rel_speed": float(np.mean(relsp_g[seg_slice])),
                                "mean_time_headway_frames": float(np.mean(thw_g[seg_slice])),
                            }
                        )
                    start = k

        ep = pl.DataFrame(episodes).sort(["start_frame", "follower_id", "leader_id"])

        if ep.height == 0:
            return ep

        if fps is not None and fps > 0:
            ep = ep.with_columns(
                (pl.col("mean_time_headway_frames") / float(fps)).alias("mean_time_headway_s")
            )

        return ep
