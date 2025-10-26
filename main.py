# TODO: Algorithm is:
# 1. Detect of traffic light.
# 2. Detect 2+ cycle objects with speed=0.
# 4. One of them cyclist moves and then after t=threshold 1+ cyclists move.

import os
import numpy as np
import common
from custom_logger import CustomLogger
from logmod import logs
from utils.tools import Tools
from utils.values import Values
import warnings

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

    def cycle_following(self):
        pass
