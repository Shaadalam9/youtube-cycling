# TODO: Algorithm is:
# 1. Detect of traffic light.
# 2. Detect 2+ cycle objects with speed=0.
# 4. One of them cyclist moves and then after t=threshold 1+ cyclists move.

import os
import numpy as np
import polars as pl
import common
from custom_logger import CustomLogger
from logmod import logs
from utils.tools import Tools
from utils.values import Values
import warnings
from itertools import combinations
from collections import defaultdict


# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# set template for plotly output
template = common.get_configs('plotly_template')

# File to store the city coordinates
file_results = 'results.pickle'

tools_class = Tools()
values_class = Values()


class Analysis():

    def filter_csv_files(self, file, df_mapping):
        """
        Filters and processes CSV files based on predefined criteria.

        This function checks if the given file is a CSV, verifies its mapping and value requirements,
        and further processes the file by loading it into a DataFrame and optionally applying geometry corrections.
        Files are only accepted if their mapping indicates sufficient footage and if required columns are present.

        Args:
            file (str): The filename to check and process.

        Returns:
            str or None: The original filename if all checks pass and the file is valid for processing;
                         otherwise, None to indicate the file should be skipped.

        Notes:
            - This method depends on several external classes and variables:
                - `values_class`: For value lookup and calculations.
                - `df_mapping`: DataFrame with mapping data for video IDs.
                - `common`: Configuration utility for various thresholds and flags.
                - `geometry_class`: Utility for geometry correction.
                - `logger`: Logging utility.
                - `folder_path`: Path to search for CSV files.
        """
        # Only process files ending with ".csv"
        file = tools_class.clean_csv_filename(file)
        if file.endswith(".csv"):
            filename = os.path.splitext(file)[0]

            # Lookup values *before* reading CSV
            values = values_class.find_values_with_video_id(df_mapping, filename)
            if values is None:
                return None  # Skip if mapping or required value is None

            vehicle_type = values[18]
            vehicle_list = common.get_configs("vehicles_analyse")

            # Only check if the list is NOT empty
            if vehicle_list:  # This is True if the list is not empty
                if vehicle_type not in vehicle_list:
                    return None

        return file

    def is_bicyclist_id(self, df, id, avg_height, min_shared_frames=5,
                        dist_thresh=80, similarity_thresh=0.8, overlap_ratio=0.7):
        """
        Determines if a person (unique-id) is riding a **bicycle only** during their trajectory
        in the YOLO detection DataFrame. Motorcycles are ignored.

        Args:
            df (pd.DataFrame): YOLO detections DataFrame containing columns:
                'yolo-id' (class, 0=person, 1=bicycle, 3=motorcycle), 'unique-id',
                'frame-count', 'x-center', 'y-center', 'width', 'height'.
            id (int or str): The unique-id of the person to analyse.
            avg_height (float): Average real-world height of the person (cm).
            min_shared_frames (int, optional): Minimum number of frames with both the person and bicycle
                present for comparison. Defaults to 5.
            dist_thresh (float, optional): Maximum real-world distance (same units as derived from avg_height,
                typically centimeters) between person and bicycle centers to be considered "moving together".
                Defaults to 80.
            similarity_thresh (float, optional): Min cosine similarity of movement direction per step.
                Range -1..1. Defaults to 0.8.
            overlap_ratio (float, optional): Fraction of overlapping frames/steps that must satisfy
                proximity/direction similarity. Defaults to 0.7 (70%).

        Returns:
            bool: True if the person is moving together with a bicycle (i.e., is a bicyclist); False otherwise.
        """
        # Extract all rows corresponding to the person id
        person_track = df[df['unique-id'] == id]
        if person_track.empty:
            return False  # No data for this id

        frames = person_track['frame-count'].values
        if len(frames) < min_shared_frames:
            return False  # Not enough frames to perform check

        first_frame, last_frame = frames.min(), frames.max()

        # --- Only consider BICYCLES (yolo-id == 1). Motorcycles (3) are ignored. ---
        mask = (
            (df['frame-count'] >= first_frame)
            & (df['frame-count'] <= last_frame)
            & (df['yolo-id'].isin([1]))  # <-- bicycle only
        )
        bikes_in_frames = df[mask]

        for vehicle_id in bikes_in_frames['unique-id'].unique():
            # Get trajectory for this bicycle
            bike_track = bikes_in_frames[bikes_in_frames['unique-id'] == vehicle_id]

            # Find shared frames between person and bicycle
            shared_frames = np.intersect1d(person_track['frame-count'], bike_track['frame-count'])
            if len(shared_frames) < min_shared_frames:
                continue  # Not enough overlapping frames to check movement together

            # Align positions for person and bicycle on shared frames, sorted by frame-count
            person_aligned = (
                person_track[person_track['frame-count'].isin(shared_frames)]
                .sort_values('frame-count')
            )
            bike_aligned = (
                bike_track[bike_track['frame-count'].isin(shared_frames)]
                .sort_values('frame-count')
            )

            person_pos = person_aligned[['x-center', 'y-center']].values
            bike_pos = bike_aligned[['x-center', 'y-center']].values

            # Person bbox heights (pixels) for shared frames
            person_heights_px = person_aligned['height'].values  # pixels per frame

            # Convert pixel distances to real-world distance using avg_height (cm)
            pixels_per_cm = person_heights_px / avg_height  # px / cm
            pixel_dists = np.linalg.norm(person_pos - bike_pos, axis=1)
            # Avoid division by zero if any pixels_per_cm is 0
            valid_scale = pixels_per_cm > 0
            if not valid_scale.all():
                # If any invalid, skip this pairing (not enough info to scale)
                continue
            distances_cm = pixel_dists / pixels_per_cm  # cm per frame

            # Proximity check over frames
            proximity = (distances_cm < dist_thresh)
            if (proximity.sum() / len(distances_cm)) < overlap_ratio:
                continue  # Not close enough for sufficient frames

            # Movement similarity (per step between consecutive frames)
            person_mov = np.diff(person_pos, axis=0)
            bike_mov = np.diff(bike_pos, axis=0)

            # If there are no steps (shouldn't happen with min_shared_frames >= 2), skip
            if len(person_mov) == 0 or len(bike_mov) == 0:
                continue

            similarities = []
            for a, b in zip(person_mov, bike_mov):
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na == 0 or nb == 0:
                    similarities.append(0.0)
                else:
                    similarities.append(np.dot(a, b) / (na * nb))
            similarities = np.array(similarities)

            # Only count steps that were also "proximate" (align step indices to frame indices - 1)
            prox_steps = proximity[1:]  # step i corresponds to frames i and i+1
            if prox_steps.size != similarities.size:
                # Safety alignment (should be equal)
                min_len = min(prox_steps.size, similarities.size)
                prox_steps = prox_steps[:min_len]
                similarities = similarities[:min_len]

            similarity_mask = (similarities > similarity_thresh) & prox_steps

            if (similarity_mask.sum() / len(similarities)) >= overlap_ratio:
                return True  # bicyclist found

        # No bicycle moving together with this person => not a bicyclist
        return False

    def detect_bicycle_following_pl(self, df: pl.DataFrame, id_to_height_cm: dict | None = None,
                                    default_avg_height_cm: float = 170.0, min_shared_frames: int = 12,
                                    dir_sim_thresh: float = 0.88, lat_offset_cm: float = 60.0,
                                    min_long_cm: float = 120.0, max_long_cm: float = 900.0,
                                    speed_ratio_tol: float = 0.45, overlap_ratio: float = 0.65,
                                    min_consecutive_steps: int = 8) -> list[dict]:
        """
        Detect pairs of *bicyclists* where one cyclist is following another.

        The function uses YOLO detection data to identify pairs of riders moving
        in the same direction, maintaining consistent longitudinal gaps
        and small lateral offsets over multiple frames.

        Parameters
        ----------
        df : pl.DataFrame
            The full YOLO detection dataframe containing columns:
            ['yolo-id', 'unique-id', 'frame-count', 'x-center', 'y-center', 'width', 'height'].
            - yolo-id: class label (0 = person, 1 = bicycle, 3 = motorcycle)
            - unique-id: tracker ID assigned per object
        id_to_height_cm : dict, optional
            Mapping of unique person IDs → their estimated real height (cm).
            Used for converting pixel distances to real-world distances.
            If None, uses `default_avg_height_cm`.
        default_avg_height_cm : float, default=170.0
            Fallback person height used for scale conversion if actual height unknown.
        min_shared_frames : int, default=12
            Minimum number of frames both riders must appear together to be considered.
        dir_sim_thresh : float, default=0.88
            Minimum cosine similarity between motion vectors to be considered same direction.
        lat_offset_cm : float, default=60.0
            Maximum allowed *lateral* (sideways) separation in cm for following.
        min_long_cm : float, default=120.0
            Minimum allowed *longitudinal* gap between bikes in cm.
        max_long_cm : float, default=900.0
            Maximum allowed longitudinal gap in cm (beyond this they’re not following).
        speed_ratio_tol : float, default=0.45
            Maximum relative speed difference allowed between riders.
        overlap_ratio : float, default=0.65
            Minimum fraction of steps that must satisfy the geometric/directional rules.
        min_consecutive_steps : int, default=8
            Minimum number of *consecutive* steps that meet all criteria for a valid event.

        Returns
        -------
        list[dict]
            Each dict represents one following event with keys:
                - 'leader_id': ID of the leading cyclist
                - 'follower_id': ID of the following cyclist
                - 'start_frame', 'end_frame': frame range of the event
                - 'support_ratio': fraction of valid steps meeting all conditions
                - 'avg_long_gap_cm': mean longitudinal gap (cm)
                - 'avg_lat_offset_cm': mean lateral offset (cm)

        Notes
        -----
        - Requires you to already have a function `is_bicyclist_id_pl` to classify persons as bicyclists.
        - The function is *microscopic*: it works on per-frame motion and heading.
        - For smoother motion, you can apply a small rolling mean on x/y before running this.
        """

        eps = 1e-6  # small constant for numerical stability

        # 1. Identify all PERSON tracks and filter those who are bicyclists
        person_ids = (
            df.filter(pl.col('yolo-id') == 0)
              .select('unique-id').unique()
              .to_series().to_list()
        )

        bicyclist_ids = []
        for pid in person_ids:
            height_cm = (id_to_height_cm.get(pid, default_avg_height_cm)
                         if id_to_height_cm else default_avg_height_cm)
            try:
                # Use your existing helper (assumed already defined)
                if self.is_bicyclist_id(df, pid, avg_height=height_cm):
                    bicyclist_ids.append(pid)
            except Exception:
                continue

        # No valid bicyclists → nothing to check
        if len(bicyclist_ids) < 2:
            return []

        # Extract per-cyclist trajectories for faster reuse
        def get_track(pid: int) -> pl.DataFrame:
            """Return sorted trajectory (frame, x, y, height_px) for a given cyclist."""
            return (
                df.filter((pl.col('unique-id') == pid) & (pl.col('yolo-id') == 0))
                  .sort('frame-count')
                  .select([
                      'frame-count',
                      'x-center',
                      'y-center',
                      pl.col('height').alias('height_px'),
                  ])
            )

        tracks = {pid: get_track(pid) for pid in bicyclist_ids}
        detections: list[dict] = []

        # Helper to compute average pixel→cm conversion per frame
        def pixels_per_cm_pair(pa_heights_px, pb_heights_px, ha_cm, hb_cm):
            """Compute average pixel/cm scale for two riders given their heights."""
            scale_a = pa_heights_px / max(ha_cm, eps)
            scale_b = pb_heights_px / max(hb_cm, eps)
            scale = 0.5 * (scale_a + scale_b)
            valid = np.isfinite(scale) & (scale > 0)
            return scale, valid

        # Pairwise comparison between all bicyclists
        for a_id, b_id in combinations(bicyclist_ids, 2):
            A = tracks[a_id]
            B = tracks[b_id]

            # Shared frames between A and B
            shared_frames = np.intersect1d(
                A['frame-count'].to_numpy(),
                B['frame-count'].to_numpy()
            )
            if shared_frames.size < min_shared_frames:
                continue

            # Align trajectories on shared frames
            A_al = A.filter(pl.col('frame-count').is_in(shared_frames)).sort('frame-count')
            B_al = B.filter(pl.col('frame-count').is_in(shared_frames)).sort('frame-count')

            A_pos = A_al.select(['x-center', 'y-center']).to_numpy().astype(float)
            B_pos = B_al.select(['x-center', 'y-center']).to_numpy().astype(float)

            # Frame-to-frame motion vectors (velocity approximations)
            A_mov = np.diff(A_pos, axis=0)
            B_mov = np.diff(B_pos, axis=0)
            if A_mov.size == 0 or B_mov.size == 0:
                continue

            # Shared average heading per step
            avg_mov = (A_mov + B_mov) / 2.0
            norms = np.linalg.norm(avg_mov, axis=1)
            heading = avg_mov / (norms[:, None] + eps)

            # Determine stable leader/follower relationship:
            # whoever is usually ahead along the direction of motion.
            proj_ahead_A = np.sum((A_pos[:-1] - B_pos[:-1]) * heading, axis=1)
            A_ahead_ratio = (proj_ahead_A > 0).mean()

            if A_ahead_ratio >= 0.6:
                leader_id, follower_id = a_id, b_id
                L_pos, F_pos = A_pos, B_pos
                L_mov, F_mov = A_mov, B_mov
                L_heights = A_al['height_px'].to_numpy()
                F_heights = B_al['height_px'].to_numpy()
                hL = id_to_height_cm.get(a_id, default_avg_height_cm) if id_to_height_cm else default_avg_height_cm
                hF = id_to_height_cm.get(b_id, default_avg_height_cm) if id_to_height_cm else default_avg_height_cm
            else:
                leader_id, follower_id = b_id, a_id
                L_pos, F_pos = B_pos, A_pos
                L_mov, F_mov = B_mov, A_mov
                L_heights = B_al['height_px'].to_numpy()
                F_heights = A_al['height_px'].to_numpy()
                hL = id_to_height_cm.get(b_id, default_avg_height_cm) if id_to_height_cm else default_avg_height_cm
                hF = id_to_height_cm.get(a_id, default_avg_height_cm) if id_to_height_cm else default_avg_height_cm

            # Compute pixel-to-cm conversion for each frame
            px_per_cm_pair, valid_scale = pixels_per_cm_pair(L_heights, F_heights, hL, hF)
            px_per_cm_steps = px_per_cm_pair[:-1]        # align to "steps" (frame→frame)
            valid_scale_steps = valid_scale[:-1]
            if valid_scale_steps.sum() < (min_shared_frames - 1):
                continue

            # Compute motion and geometry similarity between leader & follower

            # (a) Direction (heading) similarity — cosine between motion vectors
            nL = np.linalg.norm(L_mov, axis=1)
            nF = np.linalg.norm(F_mov, axis=1)
            cos_sim = np.zeros_like(nL)
            nz = (nL > eps) & (nF > eps)
            cos_sim[nz] = np.einsum('ij,ij->i', L_mov[nz], F_mov[nz]) / (nL[nz] * nF[nz])

            # (b) Relative geometry per step (at frame i)
            r = (L_pos[:-1] - F_pos[:-1])                   # vector follower→leader
            long_px = np.einsum('ij,ij->i', r, heading)     # projection on heading
            perp = r - long_px[:, None] * heading           # lateral component
            lat_px = np.linalg.norm(perp, axis=1)

            # Convert to centimeters
            long_cm = long_px / (px_per_cm_steps + eps)
            lat_cm = lat_px / (px_per_cm_steps + eps)

            # (c) Apply geometric, directional, and speed consistency checks
            geom_ok = (long_cm >= min_long_cm) & (long_cm <= max_long_cm) & (np.abs(lat_cm) <= lat_offset_cm)
            dir_ok = (cos_sim >= dir_sim_thresh)
            speed_ok = (np.abs(nL - nF) / (nL + nF + eps) <= speed_ratio_tol)
            valid_ok = valid_scale_steps & np.isfinite(long_cm) & np.isfinite(lat_cm)

            all_ok = geom_ok & dir_ok & speed_ok & valid_ok

            # Require a minimum fraction of steps to satisfy all criteria
            if all_ok.mean() < overlap_ratio:
                continue

            # Require a continuous sequence of valid following behavior
            idx = np.flatnonzero(all_ok)
            if idx.size == 0:
                continue

            # Identify consecutive runs of True
            breaks = np.where(np.diff(idx) > 1)[0] + 1
            segments = np.split(idx, breaks)
            longest = max(segments, key=len)

            if len(longest) < min_consecutive_steps:
                continue

            # Convert step indices → frame indices
            start_frame = int(shared_frames[longest[0]])
            end_frame = int(shared_frames[longest[-1] + 1])

            detections.append({
                'leader_id': leader_id,
                'follower_id': follower_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'support_ratio': float(all_ok.mean()),
                'avg_long_gap_cm': float(np.mean(long_cm[all_ok])),
                'avg_lat_offset_cm': float(np.mean(np.abs(lat_cm[all_ok]))),
            })

        # Merge overlapping detections for same leader–follower pair
        return self.merge_follow_spans(detections)

    def merge_follow_spans(self, detections: list[dict]) -> list[dict]:
        """
        Merge overlapping or adjacent following intervals belonging to the same pair.

        Parameters
        ----------
        detections : list[dict]
            List of raw detected following intervals produced by
            `detect_bicycle_following_pl()`.

        Returns
        -------
        list[dict]
            Cleaned list where overlapping or consecutive spans for the same
            (leader_id, follower_id) are merged into single continuous intervals.

        Notes
        -----
        - When merging, the function averages the gap/offset metrics
          weighted by the duration of each segment.
        - Useful if multiple short consecutive detections occur
          due to temporary occlusions or minor detection noise.
        """
        if not detections:
            return detections

        grouped = defaultdict(list)

        # Group detections by (leader, follower)
        for d in detections:
            grouped[(d['leader_id'], d['follower_id'])].append(d)

        merged = []
        for (leader, follower), group in grouped.items():
            # Sort by start frame
            group.sort(key=lambda x: x['start_frame'])
            current = None

            for det in group:
                if current is None:
                    current = det.copy()
                    continue

                # If the next detection overlaps or directly follows current → merge
                if det['start_frame'] <= current['end_frame'] + 1:
                    prev_len = current['end_frame'] - current['start_frame'] + 1
                    new_len = det['end_frame'] - det['start_frame'] + 1
                    total = max(prev_len + new_len, 1)

                    # Extend interval
                    current['end_frame'] = max(current['end_frame'], det['end_frame'])

                    # Weighted averages for smooth stats
                    current['avg_long_gap_cm'] = (
                        current['avg_long_gap_cm'] * prev_len +
                        det['avg_long_gap_cm'] * new_len
                    ) / total
                    current['avg_lat_offset_cm'] = (
                        current['avg_lat_offset_cm'] * prev_len +
                        det['avg_lat_offset_cm'] * new_len
                    ) / total
                    current['support_ratio'] = (current['support_ratio'] + det['support_ratio']) / 2.0
                else:
                    # Non-overlapping: push current and start new
                    merged.append(current)
                    current = det.copy()

            if current is not None:
                merged.append(current)

        return merged
