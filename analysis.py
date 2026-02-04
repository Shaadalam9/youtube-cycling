from tqdm import tqdm
from typing import Optional, Set
import math
from custom_logger import CustomLogger
from logmod import logs
from urllib.parse import urljoin, urlparse
from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
import requests
import os
import common
import pathlib
from bs4 import BeautifulSoup
import polars as pl
import cv2
from types import SimpleNamespace                # lightweight config container
from utils.bicyclist_detect import Algorithm
from utils.bicyclist_following import CyclistFollowing, FollowingParams
from utils.analytics.io import IO
from utils.core.tools import Tools
from utils.analytics.geo import Geo


tools = Tools()
algo = Algorithm()
cf = CyclistFollowing(algo)
analytics_IO = IO()
geo = Geo()

# Common junk files/folders to ignore.
MISC_FILES: Set[str] = {"DS_Store"}


logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger


# -----------------------------------------------------------------------------
# USER PARAMETERS (edit these variables; no argparse / no CLI parser is used)
# -----------------------------------------------------------------------------
def _cfg_bool(key: str, default: bool) -> bool:
    """Best-effort boolean config lookup with a safe fallback."""
    try:
        return bool(common.get_configs(key))
    except Exception:
        return default


def _cfg_float(key: str, default: float) -> float:
    """Best-effort float config lookup with a safe fallback."""
    try:
        return float(common.get_configs(key))
    except Exception:
        return default


# When True: if cyclist-following episodes are detected in a CSV, the script will
# download the corresponding source video and generate an annotated video showing
# the involved cyclists (and their bicycles) across the entire CSV segment.
DOWNLOAD_AND_ANNOTATE: bool = _cfg_bool("DOWNLOAD_AND_ANNOTATE", True)

# Output folders
videos_dir: str = common.get_configs("videos")
DOWNLOADED_VIDEOS_DIR = os.path.join(videos_dir, "downloaded_video")
ANNOTATED_VIDEOS_DIR = os.path.join(videos_dir, "annotated_video")
TRIMMED_CLIPS_DIR = os.path.join(videos_dir, "trimmed_video")

# If True, keep the intermediate trimmed clip on disk. If False, it will be deleted
# after the annotated video is written successfully.
KEEP_TRIMMED_CLIP: bool = _cfg_bool("KEEP_TRIMMED_CLIP", False)

# Annotation controls
# If True, always draw IDs for ALL detected bicyclists (not only those involved in following episodes).
ANNOTATE_ALL_BICYCLISTS: bool = _cfg_bool("ANNOTATE_ALL_BICYCLISTS", True)
# If True, render a text overlay each frame showing active follower->leader pairs.
ANNOTATE_PAIR_OVERLAY: bool = _cfg_bool("ANNOTATE_PAIR_OVERLAY", True)
# If True, always write a full-segment annotated video for each CSV segment that has bicyclists,
# regardless of whether a following episode is detected. Cropped clips (if enabled) are still
# produced only when following exists.
ANNOTATE_WHOLE_SEGMENT: bool = _cfg_bool("ANNOTATE_WHOLE_SEGMENT", False)


# Cropping controls (optional)
# If True: when following episodes exist, write cropped annotated clips around each follower->leader pair.
# Crop window definition:
#   - start: CROP_PRE_SECONDS before the first encounter (episode start_frame)
#   - end  : CROP_POST_GONE_SECONDS after BOTH leader and follower have disappeared from the frame
#           (within the CSV segment; clamped to segment bounds).
CROP_AROUND_FOLLOWING: bool = _cfg_bool("CROP_AROUND_FOLLOWING", False)
CROP_PRE_SECONDS: float = _cfg_float("CROP_PRE_SECONDS", 5.0)
CROP_POST_GONE_SECONDS: float = _cfg_float("CROP_POST_GONE_SECONDS", 6.0)
# If True and CROP_AROUND_FOLLOWING is enabled, also write the full-segment annotated video.
ALSO_WRITE_FULL_SEGMENT_WHEN_CROPPING: bool = _cfg_bool("ALSO_WRITE_FULL_SEGMENT_WHEN_CROPPING", False)


# -----------------------------------------------------------------------------
# ANNOTATION COLORS (BGR)
# -----------------------------------------------------------------------------
# follower  : cyclist that is following another cyclist
# leader    : cyclist that is being followed (leader in episodes table)
# normal    : cyclist that is not in a following episode
COLOR_CYCLIST_FOLLOWER = (0, 0, 255)   # red
COLOR_CYCLIST_FOLLOWING = (0, 255, 0)  # green
COLOR_CYCLIST_LEADER = COLOR_CYCLIST_FOLLOWING  # alias for clarity
COLOR_CYCLIST_NORMAL = (0, 215, 255)   # orange/yellow-ish (visible on most backgrounds)
COLOR_BICYCLE = (255, 0, 0)            # blue


class Analysis():

    def download_videos_from_ftp(self, filename: str, base_url: Optional[str] = None, out_dir: str = ".",
                                 username: Optional[str] = None, password: Optional[str] = None,
                                 token: Optional[str] = None, timeout: int = 20, debug: bool = True,
                                 max_pages: int = 500) -> Optional[tuple[str, str, str, float]]:
        """
        Search and download a specific .mp4 file from a multi-directory FastAPI-based
        HTTP file server (e.g., files.mobility-squad.com). This function attempts direct
        download from known /files/ paths (tue1/tue2/tue3), and if not found, recursively
        crawls the /browse pages to locate the video file. Progress is shown with tqdm.
        Args: filename (str): Target file name (with or without .mp4 extension).
        base_url (str, optional): Base URL of the file server.
        Must include protocol, e.g. "https://files.mobility-squad.com/".
        out_dir (str, optional): Local output directory to save the video.
        Defaults to current directory ".".
        username (str, optional): Username for HTTP Basic Auth.
        password (str, optional): Password for HTTP Basic Auth.
        token (str, optional): Token string for token-based authentication.
        Sent as a query parameter ?token=.... timeout (int, optional):
        Request timeout in seconds. Default is 20. max_pages (int, optional):
        Safety limit for crawl depth/pages. Default is 500.
        Returns: Optional[Tuple[str, str, str, float]]: Returns a tuple
        (local_path, filename, resolution_label, fps) if the download succeeds,
        or None if the file is not found or download fails.
        Logging: - logger.info: start, success summaries.
        - logger.debug: HTTP requests, crawl steps, file matches.
        - logger.warning: non-fatal issues (metadata failures, skipped pages).
        - logger.error: fatal errors (network/IO exceptions). Example:

            result = self.download_videos_from_http_fileserver(
                filename="3ai7SUaPoHM",
                base_url="https://files.mobility-squad.com/",
                out_dir="./downloads",
                username="mobility",
                password="your_password"
            )
            if result:
                path, name, res, fps = result
                print(f"Downloaded {name} ({res}, {fps} fps) to {path}")
            else:
                print("File not found or failed.")
        """
        # -------------------- Input Preparation --------------------
        if not base_url:
            logger.error("Base URL is missing.")
            return None

        base = base_url if base_url.endswith("/") else base_url + "/"

        if username == "":
            username = None
        if password == "":
            password = None

        filename_with_ext = filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"
        filename_lower = filename_with_ext.lower()

        # Local cache: if the file is already present on disk, reuse it.
        os.makedirs(out_dir, exist_ok=True)
        cached_path = os.path.join(out_dir, filename_with_ext)
        if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
            resolution, fps_meta = "unknown", 0.0
            try:
                fps_meta = float(Analysis.get_video_fps(cached_path))
                resolution = Analysis.get_video_resolution_label(cached_path)
            except Exception:
                pass
            logger.info(f"Using cached video: {cached_path}")
            return cached_path, filename_with_ext, resolution, fps_meta
        aliases = ["tue1", "tue2", "tue3", "tue4"]

        req_params = {"token": token} if token else None

        logger.info(f"Starting download for '{filename_with_ext}'")
        logger.debug(
            f"Base URL: {base} | Auth: {'Basic' if username and password else 'None'} | Token: {'Yes' if token else 'No'}"  # noqa:E501
        )  # noqa: E501

        # ---------- Session ----------
        with requests.Session() as session:
            if username and password:
                session.auth = (username, password)
            session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

            def fetch(url: str, stream: bool = False) -> Optional[requests.Response]:
                """GET with logging and safe error handling."""
                try:
                    r = session.get(url, timeout=timeout, params=req_params, stream=stream)
                    logger.debug(f"GET {url} -> {r.status_code}")
                    if r.status_code == 401:
                        logger.error(f"Authentication failed for {url}")
                    r.raise_for_status()
                    return r
                except requests.RequestException as e:
                    logger.warning(f"Request failed [{url}]: {e}")
                    return None

            # ---------- 1. Try direct /files paths ----------
            for alias in aliases:
                direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
                logger.debug(f"Trying direct URL: {direct_url}")

                r = fetch(direct_url, stream=True)
                if r is None:
                    continue

                logger.info(f"Found file via direct URL: {direct_url}")
                content_len = int(r.headers.get("content-length", 0))
                logger.debug(f"Content-Length: {content_len or 'unknown'} bytes")

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                # Avoid overwriting
                if os.path.exists(local_path):
                    stem, suf = os.path.splitext(local_path)
                    i = 1
                    while os.path.exists(f"{stem} ({i}){suf}"):
                        i += 1
                    local_path = f"{stem} ({i}){suf}"
                    logger.warning(f"File exists, saving as: {local_path}")

                # ---------- Download ----------
                try:
                    total = content_len or None
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading from ftp: {filename_with_ext}",
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if total:
                                    bar.update(len(chunk))
                    logger.info(f"Download complete: {local_path} ({written} bytes)")
                except Exception as e:
                    logger.error(f"Download failed for {filename_with_ext}: {e}")
                    return None

                # ---------- Metadata ----------
                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path))  # type: ignore
                    resolution = Analysis.get_video_resolution_label(local_path)
                    logger.debug(f"Metadata extracted: fps={fps}, resolution={resolution}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                logger.info(f"✅ Saved '{filename_with_ext}' (res={resolution}, fps={fps})")
                return local_path, filename_with_ext, resolution, fps

            # ---------- 2. Crawl /browse fallback ----------
            visited: Set[str] = set()

            def is_dir_link(href: str) -> bool:
                return href.startswith("/v/") and "/browse" in href

            def is_file_link(href: str) -> bool:
                return "/files/" in href

            def crawl(start_url: str) -> Optional[str]:
                """Recursively traverse /browse pages."""
                stack = [start_url]
                pages_seen = 0

                while stack:
                    url = stack.pop()

                    if url in visited:
                        continue

                    visited.add(url)
                    pages_seen += 1
                    if pages_seen > max_pages:
                        logger.warning(f"Crawl aborted after {max_pages} pages.")
                        return None

                    resp = fetch(url)
                    if resp is None:
                        continue

                    try:
                        soup = BeautifulSoup(resp.text, "html.parser")
                    except Exception as e:
                        logger.warning(f"HTML parse failed at {url}: {e}")
                        continue

                    for a in soup.find_all("a"):
                        href = (a.get("href") or "").strip()  # type: ignore
                        if not href:
                            continue

                        full = urljoin(url, href)

                        if is_file_link(href):
                            anchor_text = (a.text or "").strip().lower()
                            tail = pathlib.PurePosixPath(urlparse(full).path).name.lower()
                            if anchor_text == filename_lower or tail == filename_lower:
                                logger.info(f"File located via crawl: {full}")
                                return full

                        if is_dir_link(href):
                            stack.append(full)

                logger.debug("Crawl finished — no file found.")
                return None

            for alias in aliases:
                start_url = urljoin(base, f"v/{alias}/browse")
                logger.debug(f"Crawling alias: {alias} -> {start_url}")

                found = crawl(start_url)
                if not found:
                    continue

                r = fetch(found, stream=True)
                if not r:
                    continue

                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                try:
                    total = int(r.headers.get("content-length", 0)) or None
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {filename_with_ext}",
                    ) as bar:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if total:
                                    bar.update(len(chunk))
                    logger.info(f"Downloaded via crawl: {local_path} ({written} bytes)")
                except Exception as e:
                    logger.error(f"Download during crawl failed: {e}")
                    return None

                resolution, fps = "unknown", 0.0
                try:
                    fps = float(self.get_video_fps(local_path))  # type: ignore
                    resolution = Analysis.get_video_resolution_label(local_path)
                    logger.debug(f"Metadata: fps={fps}, resolution={resolution}")
                except Exception as e:
                    logger.warning(f"Metadata extraction failed: {e}")

                return local_path, filename_with_ext, resolution, fps

            logger.warning(f"File '{filename_with_ext}' not found in any alias.")
            return None

    @staticmethod
    def get_video_fps(video_path: str) -> float:
        """Return FPS for a local video file (OpenCV)."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return fps

    def annotate_following_segment(
        self,
        *,
        input_video_path: str,
        output_video_path: str,
        df: pl.DataFrame,
        cyclist_map: pl.DataFrame,
        episodes: pl.DataFrame,
        involved_cyclist_ids: set[int],
        frame_offset: int = 0,
        fps_override: Optional[float] = None,
        draw_all_bicyclists: bool = True,
        draw_pair_overlay: bool = True,
        draw_labels: bool = True,
    ) -> None:
        """
        Render an annotated video for a CSV segment.

        - Reads frames from `input_video_path`.
        - Draws bboxes for cyclists (yolo-id==0) and bicycles (yolo-id!=0, typically 1)
          for the IDs selected from `cyclist_map` and/or `involved_cyclist_ids`.
        - If `episodes` contains intervals, cyclists are labelled as follower/leader when active.
        - Writes output WITHOUT audio via OpenCV VideoWriter.
        - `frame_offset` aligns video frame index -> CSV frame-count.
        """
        out_dir = os.path.dirname(output_video_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        # -----------------------------
        # Defensive column checks
        # -----------------------------
        required_df_cols = {"frame-count", "yolo-id", "unique-id", "x-center", "y-center", "width", "height"}
        missing_df_cols = required_df_cols - set(df.columns)
        if missing_df_cols:
            raise ValueError(f"df is missing required columns: {sorted(missing_df_cols)}")

        # cyclist_map is optional-ish, but if present we expect these cols
        has_cyclist_map = cyclist_map is not None and cyclist_map.height > 0
        cyclist_map_has_cols = has_cyclist_map and {"cyclist_id", "bicycle_id"}.issubset(set(cyclist_map.columns))

        # -----------------------------
        # Select which IDs to draw
        # -----------------------------
        cyclist_ids: set[int] = set()

        if draw_all_bicyclists and cyclist_map_has_cols:
            cyclist_ids |= set(map(int, cyclist_map.get_column("cyclist_id").to_list()))

        if involved_cyclist_ids:
            cyclist_ids |= set(map(int, involved_cyclist_ids))

        # If we ended up with nothing but cyclist_map exists, fall back to cyclist_map cyclists.
        if not cyclist_ids and cyclist_map_has_cols:
            cyclist_ids = set(map(int, cyclist_map.get_column("cyclist_id").to_list()))

        bicycle_ids: set[int] = set()
        if cyclist_ids and cyclist_map_has_cols:
            bicycle_ids = set(
                map(
                    int,
                    cyclist_map.filter(pl.col("cyclist_id").is_in(list(cyclist_ids)))
                    .get_column("bicycle_id")
                    .to_list(),
                )
            )

        object_ids_to_draw: set[int] = cyclist_ids | bicycle_ids

        # Map bicycle_id -> cyclist_id for nicer labels
        bike_to_cyclist: dict[int, int] = {}
        if cyclist_map_has_cols:
            for r in cyclist_map.select(["bicycle_id", "cyclist_id"]).iter_rows(named=True):
                try:
                    bike_to_cyclist[int(r["bicycle_id"])] = int(r["cyclist_id"])
                except Exception:
                    continue

        # -----------------------------
        # Build episode intervals (roles)
        # -----------------------------
        leader_col: Optional[str] = None
        if episodes is not None and episodes.height > 0:
            if "leader_id" in episodes.columns:
                leader_col = "leader_id"
            elif "following_id" in episodes.columns:
                leader_col = "following_id"

        intervals: list[tuple[int, int, int, int]] = []  # (start, end, follower_id, leader_id)
        if leader_col is not None:
            needed = {"start_frame", "end_frame", "follower_id", leader_col}
            missing = needed - set(episodes.columns)
            if missing:
                raise ValueError(f"episodes is missing required columns for roles: {sorted(missing)}")

            for r in episodes.select(["start_frame", "end_frame", "follower_id", leader_col]).iter_rows(named=True):
                intervals.append(
                    (
                        int(r["start_frame"]),
                        int(r["end_frame"]),
                        int(r["follower_id"]),
                        int(r[leader_col]),
                    )
                )

        def roles_for_frame(frame_count: int) -> tuple[dict[int, str], dict[int, int]]:
            """roles: cyclist_id -> role; active_pairs: follower_id -> leader_id"""
            roles: dict[int, str] = {}
            active_pairs: dict[int, int] = {}
            for s, e, fid, lid in intervals:
                if s <= frame_count <= e:
                    roles[fid] = "follower"
                    roles[lid] = "leader"
                    active_pairs[fid] = lid
            return roles, active_pairs

        # -----------------------------
        # Pre-index bboxes by frame
        # -----------------------------
        frame_to_rows: dict[int, list[dict]] = {}
        if df.height > 0 and object_ids_to_draw:
            wanted = (
                df.filter(pl.col("unique-id").is_in(list(object_ids_to_draw)))
                .select(["frame-count", "yolo-id", "unique-id", "x-center", "y-center", "width", "height"])
            )
            for row in wanted.iter_rows(named=True):
                fc = int(row["frame-count"])
                frame_to_rows.setdefault(fc, []).append(row)

        # Coordinates are YOLO-normalized [0,1] in this project
        coords_normalised = True

        # -----------------------------
        # Open input + output video
        # -----------------------------
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError(f"Invalid video dimensions for: {input_video_path} (w={width}, h={height})")

        if fps_override is not None and float(fps_override) > 0:
            fps = float(fps_override)
        else:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open VideoWriter: {output_video_path}")

        # -----------------------------
        # Main loop
        # -----------------------------
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                csv_frame = frame_idx + int(frame_offset)
                roles, active_pairs = roles_for_frame(csv_frame)

                # Draw all bboxes for this CSV frame
                for row in frame_to_rows.get(csv_frame, []):
                    yolo_id = int(row["yolo-id"])
                    obj_id = int(row["unique-id"])

                    xc = float(row["x-center"])
                    yc = float(row["y-center"])
                    w = float(row["width"])
                    h = float(row["height"])

                    if coords_normalised:
                        xc *= width
                        yc *= height
                        w *= width
                        h *= height

                    x1 = int(round(xc - w / 2.0))
                    y1 = int(round(yc - h / 2.0))
                    x2 = int(round(xc + w / 2.0))
                    y2 = int(round(yc + h / 2.0))

                    # clip to bounds
                    x1 = max(0, min(width - 1, x1))
                    x2 = max(0, min(width - 1, x2))
                    y1 = max(0, min(height - 1, y1))
                    y2 = max(0, min(height - 1, y2))

                    if yolo_id == 0:  # cyclist/person
                        role = roles.get(obj_id, "normal")
                        if role == "follower":
                            color = COLOR_CYCLIST_FOLLOWER
                        elif role == "leader":
                            color = COLOR_CYCLIST_LEADER
                        else:
                            color = COLOR_CYCLIST_NORMAL
                        label = f"{role}:{obj_id}"
                    else:  # bicycle (or other)
                        color = COLOR_BICYCLE
                        rider_id = bike_to_cyclist.get(obj_id)
                        label = f"bicycle:{obj_id}" if rider_id is None else f"bicycle:{obj_id} rider:{rider_id}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    if draw_labels:
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

                # Optional overlay: active pairs
                if draw_pair_overlay and active_pairs:
                    x0, y0 = 12, 28
                    for i, (fid, lid) in enumerate(sorted(active_pairs.items())):
                        txt = f"Follower {fid} -> Leader {lid}"
                        cv2.putText(
                            frame,
                            txt,
                            (x0, y0 + 18 * i),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                out.write(frame)
                frame_idx += 1

        finally:
            cap.release()
            out.release()

    @staticmethod
    def get_video_resolution_label(video_path: str) -> str:
        """
        Return a resolution label for a local video file using an "exact, truthful" policy.

        Policy
        -----------------
        - Read the frame height (pixels) from the file via OpenCV.
        - If the height matches a known standard, return its label (e.g., "720p", "1080p").
        - If the height is close to a known standard within a small tolerance (to account for
          encoder/container padding such as 1088 instead of 1080), return the nearest
          standard label.
        - Otherwise, return the exact height in the form "<height>p" (e.g., "540p", "768p").

        This approach remains compatible with the updated download selection logic, which
        may select non-standard heights when they are the best available option.

        Parameters
        ----------
        video_path : str
            Path to the video file on disk.

        Returns
        -------
        str
            A resolution label (e.g., "144p", "360p", "720p", "1080p") or "<height>p" for
            non-standard heights.

        Raises
        ------
        FileNotFoundError
            If `video_path` does not exist.
        RuntimeError
            If the video cannot be opened or the frame height cannot be determined.

        Notes
        -----
        - The label is derived from frame height only (not bitrate, codec, aspect ratio, etc.).
        - Some videos may report padded heights (e.g., 544, 736, 1088). These are mapped to
          the nearest standard label only when within the configured tolerance.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

        if height <= 0:
            raise RuntimeError(f"Could not determine frame height for video: {video_path}")

        # Canonical/common heights (add more if we want "named" labels, but non-matches still fall back to "<height>p")
        labels = {
            144: "144p",
            240: "240p",
            360: "360p",
            480: "480p",
            540: "540p",
            576: "576p",
            720: "720p",
            900: "900p",
            1080: "1080p",
            1440: "1440p",
            2160: "2160p",  # 4K UHD
            4320: "4320p",  # 8K UHD
        }

        # Small tolerance to normalize padded encodes (e.g., 1088 -> 1080).
        tolerance_px = 16

        # Exact match
        if height in labels:
            return labels[height]

        # Nearest label within tolerance (padding normalization only)
        closest_h = min(labels.keys(), key=lambda h: abs(height - h))
        if abs(height - closest_h) <= tolerance_px:
            return labels[closest_h]

        # Truthful fallback for truly non-standard heights
        return f"{height}p"

    def trim_video(self, input_path, output_path, start_time, end_time):
        """
        Trims a segment from a video and saves the result to a specified file.
        Parameters: input_path (str): The file path to the original video.
        output_path (str): The destination file path where the trimmed video will be saved.
        start_time (float or str): The start time for the trimmed segment.
        This can be specified in seconds or in a time format recognised by MoviePy.
        nd_time (float or str): The end time for the trimmed segment.
        Similar to start_time, it can be in seconds or another supported time format.
        Returns: None The function performs the following steps:
        1. Loads the original video using MoviePy's VideoFileClip.
        2. Creates a subclip from the original video based on the provided start_time and end_time.
        3. Writes the subclip to the output_path using the H.264 video codec and AAC audio codec.
        4. Closes the video file to free up resources.
        """
        # Load the video and create a subclip using the provided start and end times.
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)  # type: ignore

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the subclip to the specified output file using the 'libx264' codec for video and 'aac' for audio.
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Close the video clip to release any resources used.
        video_clip.close()


if __name__ == "__main__":
    logger.info("Analysis started.")

    # ---------------------------------------------------------------------
    # Load secrets (email + FTP credentials)
    # ---------------------------------------------------------------------
    secret = SimpleNamespace(
        email_smtp=common.get_secrets("email_smtp"),
        email_account=common.get_secrets("email_account"),
        email_password=common.get_secrets("email_password"),
        ftp_username=common.get_secrets("ftp_username"),
        ftp_password=common.get_secrets("ftp_password"),
    )

    mapping_path = common.get_configs("mapping")
    df_mapping = pl.read_csv(
        mapping_path,
        schema_overrides={"literacy_rate": pl.Float64},
    )

    countries_analyse: list[str] = common.get_configs("countries_analyse")

    if countries_analyse:  # non-empty -> filter
        df_mapping = df_mapping.filter(pl.col("iso3").is_in(countries_analyse))
    # else: empty -> do nothing (keep all rows)

    min_conf = common.get_configs("min_confidence")

    # Precompute fast lookup once (place this immediately before the loops)
    id_to_place: dict[int, tuple[str, str, str]] = {
        int(row_id): (city, state, country)
        for row_id, city, state, country in df_mapping.select(["id", "city", "state", "country"]).iter_rows()
    }

    analysis = Analysis()

    for folder_path in common.get_configs("data"):  # Iterable[str]
        if not os.path.exists(folder_path):
            logger.warning(f"Folder does not exist: {folder_path}.")
            continue

        for file_name in tqdm(os.listdir(folder_path), desc=f"Processing files in {folder_path}"):
            filtered: Optional[str] = analytics_IO.filter_csv_files(
                file=file_name, df_mapping=df_mapping
            )
            if filtered is None:
                continue

            file_str: str = os.fspath(filtered)

            if file_str in MISC_FILES:
                continue

            filename_no_ext = os.path.splitext(file_str)[0]
            logger.debug(f"{filename_no_ext}: fetching values.")

            file_path = os.path.join(folder_path, file_str)

            # Polars read + filter
            df = pl.read_csv(file_path)

            df = (
                df
                # 1) drop invalid IDs early
                .filter(pl.col("unique-id") != -1)
                # 2) enforce join-key dtype compatibility (avoid crashes if some rows look like 15.0)
                .with_columns(
                    pl.col("unique-id").cast(pl.Int64, strict=False)
                )
                # 3) drop rows that couldn't be cast to int
                .filter(pl.col("unique-id").is_not_null())
                # 4) your existing confidence filter
                .filter(pl.col("confidence") >= min_conf)
            )

            # After reading the file, clean up the filename
            base_name = tools.clean_csv_filename(file_str)
            filename_no_ext = os.path.splitext(base_name)[0]

            try:
                video_id, start_index, fps = filename_no_ext.rsplit("_", 2)
            except ValueError:
                logger.warning(f"Unexpected filename format: {filename_no_ext}")
                continue

            video_city_id = geo.find_city_id(df_mapping, video_id, int(start_index))

            place = id_to_place.get(int(video_city_id)) if video_city_id is not None else None
            if place is None:
                logger.warning(f"{file_str}: no mapping row found for id={video_city_id}.")
                continue

            video_city, video_state, video_country = place
            logger.info(f"{file_str}: found values {video_city}, {video_state}, {video_country}.")

            cyclist_map = cf.identify_bicyclists(df, min_shared_frames=30, score_thresh=0.0)
            states = cf.build_cyclist_states(df, cyclist_map, prefer_vehicle_center=True)
            fps_csv = float(fps)  # from filename suffix
            episodes = cf.detect_following_episodes(
                states,
                params=FollowingParams(
                    speed_min=8e-4,
                    dir_cos_thresh=0.2,
                    rel_speed_thresh=0.003,
                    long_min=0.03,
                    long_max=0.3,
                    lat_max=0.1,
                    min_follow_frames=10,
                    gap_allow=10,
                ),
                fps=None,
            )
            logger.info(str(cyclist_map))

            logger.info(str(episodes))

            # Human-readable summary of unique pairs
            try:
                pair_summary = cf.summarize_following_pairs(episodes, fps=fps_csv)
                logger.info(str(pair_summary))
            except Exception:
                pass

            # If enabled, download the source video and generate an annotated video for this CSV segment
            logger.info(f"{file_str}: annotation pipeline triggered (DOWNLOAD_AND_ANNOTATE={DOWNLOAD_AND_ANNOTATE},bicyclists={cyclist_map.height})")  # noqa: E501
            should_annotate = bool(DOWNLOAD_AND_ANNOTATE) and (episodes.height > 0)

            # If cropping is enabled and we're NOT writing full segments, only process segments where following exists.

            if CROP_AROUND_FOLLOWING and (not ANNOTATE_WHOLE_SEGMENT):

                should_annotate = should_annotate and (episodes.height > 0)

            if should_annotate:
                try:
                    involved_cyclists: set[int] = set()
                    if "follower_id" in episodes.columns:
                        involved_cyclists |= set(episodes.get_column("follower_id").to_list())
                    if "leader_id" in episodes.columns:
                        involved_cyclists |= set(episodes.get_column("leader_id").to_list())

                    # Download base video (cached if already present on disk)
                    dl = analysis.download_videos_from_ftp(
                        filename=video_id,
                        base_url=common.get_configs("ftp_server"),
                        out_dir=DOWNLOADED_VIDEOS_DIR,
                        username=getattr(secret, "ftp_username", None),
                        password=getattr(secret, "ftp_password", None),
                        token=getattr(secret, "ftp_token", None),
                    )
                    if dl is None:
                        logger.warning(f"{file_str}: could not download video for video_id='{video_id}'.")
                        continue

                    local_video_path, downloaded_name, resolution, downloaded_fps = dl
                    logger.info(f"{file_str}: downloaded video '{downloaded_name}' ({resolution}, fps={downloaded_fps}).")  # noqa: E501

                    # Compute segment [start_seconds, end_seconds] from CSV filename + frame-count
                    try:
                        start_seconds = float(start_index)
                    except Exception:
                        logger.warning(f"{file_str}: could not parse start time from '{start_index}'. Skipping video.")
                        continue

                    # CSV fps (from filename) is the best alignment hint; fall back to downloaded fps
                    fps_value = 0.0
                    try:
                        fps_value = float(fps)
                    except Exception:
                        fps_value = float(downloaded_fps or 0.0)

                    if fps_value <= 0:
                        fps_value = 25.0

                    min_frame = int(df.select(pl.min("frame-count")).item()) if df.height > 0 else 0
                    max_frame = int(df.select(pl.max("frame-count")).item()) if df.height > 0 else 0
                    # IMPORTANT: frame-count is absolute within the segment; min_frame can be >0 simply
                    # because early frames had no detections. Clip timing must be based on the actual
                    # frame indices, not (max-min+1) starting at t=0.
                    end_seconds = float(start_seconds) + (float(max_frame + 1) / float(fps_value)) if fps_value > 0 else float(start_seconds)

                    # Decide which window(s) to annotate
                    os.makedirs(TRIMMED_CLIPS_DIR, exist_ok=True)
                    os.makedirs(ANNOTATED_VIDEOS_DIR, exist_ok=True)

                    jobs: list[dict] = []

                    def _add_job(*, clip_start_s: float, clip_end_s: float, frame_offset: int,
                                 clip_name: str, annotated_name: str) -> None:
                        jobs.append({
                            'clip_start_s': float(clip_start_s),
                            'clip_end_s': float(clip_end_s),
                            'frame_offset': int(frame_offset),
                            'clip_name': str(clip_name),
                            'annotated_name': str(annotated_name),
                        })

                    # Default (full CSV segment) job definition
                    full_annotated_suffix = 'following_annotated' if episodes.height > 0 else 'annotated'
                    full_clip_name = f"{filename_no_ext}.mp4"
                    full_annotated_name = f"{filename_no_ext}_{full_annotated_suffix}.mp4"

                    if CROP_AROUND_FOLLOWING and episodes.height > 0:
                        # Produce a cropped annotated clip per detected follower->leader pair.
                        pre_frames = int(math.ceil(CROP_PRE_SECONDS * fps_value))
                        post_frames = int(math.ceil(CROP_POST_GONE_SECONDS * fps_value))

                        try:
                            pairs = episodes.select(['follower_id', 'leader_id']).unique()
                        except Exception:
                            pairs = pl.DataFrame(schema={'follower_id': pl.Int64, 'leader_id': pl.Int64})

                        for fid, lid in pairs.select(['follower_id', 'leader_id']).iter_rows():
                            fid_i = int(fid)
                            lid_i = int(lid)

                            pair_eps = episodes.filter((pl.col('follower_id') == fid_i
                                                        ) & (pl.col('leader_id') == lid_i))
                            if pair_eps.height == 0:
                                continue

                            # 1) First encounter = first start_frame for this pair.
                            try:
                                first_encounter = int(pair_eps.select(pl.min('start_frame')).item())
                            except Exception:
                                continue

                            # 2) 'Gone' = first frame AFTER the last time either cyclist appears (based on states).
                            try:
                                last_f = states.filter(pl.col('cyclist_id') == fid_i
                                                       ).select(pl.max('frame-count')).item()
                            except Exception:
                                last_f = None
                            try:
                                last_l = states.filter(pl.col('cyclist_id') == lid_i
                                                       ).select(pl.max('frame-count')).item()
                            except Exception:
                                last_l = None

                            try:
                                fallback_last = int(pair_eps.select(pl.max('end_frame')).item())
                            except Exception:
                                fallback_last = first_encounter

                            last_frames: list[int] = []
                            for v in (last_f, last_l):
                                if v is not None:
                                    try:
                                        last_frames.append(int(v))
                                    except Exception:
                                        pass
                            last_any = max(last_frames + [int(fallback_last)]) if last_frames else int(fallback_last)
                            gone_frame = min(max_frame + 1, int(last_any) + 1)

                            crop_start_frame = max(min_frame, int(first_encounter) - pre_frames)
                            crop_end_frame = min(max_frame, int(gone_frame) + post_frames)

                            # Ensure at least ~1 second of output (VideoFileClip can error on empty subclips).
                            if crop_end_frame <= crop_start_frame:
                                crop_end_frame = min(max_frame, crop_start_frame + max(1, int(math.ceil(fps_value))))

                            clip_start_s = float(start_seconds) + float(crop_start_frame) / float(fps_value)

                            # +1 frame so we include crop_end_frame
                            clip_end_s = float(start_seconds) + float(crop_end_frame + 1) / float(fps_value)

                            clip_name = f"{filename_no_ext}_f{fid_i}_l{lid_i}_crop.mp4"
                            annotated_name = f"{filename_no_ext}_f{fid_i}_l{lid_i}_following_crop_annotated.mp4"
                            _add_job(
                                clip_start_s=clip_start_s,
                                clip_end_s=clip_end_s,
                                frame_offset=crop_start_frame,
                                clip_name=clip_name,
                                annotated_name=annotated_name,
                            )
                            logger.info(
                                f"{file_str}: crop window for follower {fid_i} -> leader {lid_i}: "
                                f"frames [{crop_start_frame}, {crop_end_frame}] "
                                f"(t=[{clip_start_s:.2f}s, {clip_end_s:.2f}s])"
                            )

                        if ANNOTATE_WHOLE_SEGMENT or ALSO_WRITE_FULL_SEGMENT_WHEN_CROPPING or (not jobs):
                            _add_job(
                                clip_start_s=float(start_seconds) + (float(min_frame) / float(fps_value) if fps_value > 0 else 0.0),
                                clip_end_s=float(end_seconds),
                                frame_offset=int(min_frame),
                                clip_name=full_clip_name,
                                annotated_name=full_annotated_name,
                            )
                    else:
                        # Default behavior: annotate the full CSV segment.
                        _add_job(
                            clip_start_s=float(start_seconds) + (float(min_frame) / float(fps_value) if fps_value > 0 else 0.0),
                                clip_end_s=float(end_seconds),
                                frame_offset=int(min_frame),
                            clip_name=full_clip_name,
                            annotated_name=full_annotated_name,
                        )

                    # Execute jobs
                    for job in jobs:
                        clip_path = os.path.join(TRIMMED_CLIPS_DIR, job['clip_name'])
                        annotated_path = os.path.join(ANNOTATED_VIDEOS_DIR, job['annotated_name'])

                        analysis.trim_video(local_video_path, clip_path, job['clip_start_s'], job['clip_end_s'])
                        analysis.annotate_following_segment(
                            input_video_path=clip_path,
                            output_video_path=annotated_path,
                            df=df,
                            cyclist_map=cyclist_map,
                            episodes=episodes,
                            involved_cyclist_ids=involved_cyclists,
                            frame_offset=int(job['frame_offset']),
                            fps_override=fps_value,
                            draw_all_bicyclists=ANNOTATE_ALL_BICYCLISTS,
                            draw_pair_overlay=ANNOTATE_PAIR_OVERLAY,
                            draw_labels=True,
                        )
                        logger.info(f"{file_str}: annotated video written to {annotated_path}")

                        if not KEEP_TRIMMED_CLIP:
                            try:
                                os.remove(clip_path)
                            except Exception:
                                pass

                except Exception as e:
                    logger.error(f"{file_str}: download/annotate failed: {e}")
