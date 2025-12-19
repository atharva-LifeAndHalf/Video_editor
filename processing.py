import cv2
import numpy as np
from clip_blip import cosine
from video_analysis import segment_scenes
from config import FINAL_FPS, LOGO_PATH
from video_analysis import deduplicate_frames
from filters import apply_filter
from transitions import crossfade, dissolve, fade_to_black
from advance import apply_stabilization, remove_background, remove_object, add_logo_overlay

# ---------------- PROCESSING ----------------
def process_clip_to_file(clip_data, style, output_path):
    """Process clip and save to file"""
    frames = clip_data["frames"]

    print(f"   🎬 Segmenting scenes...")
    scenes, all_embeddings = segment_scenes(frames, style["scene_threshold"])
    print(f"   🎞️ Found {len(scenes)} scenes")

    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), FINAL_FPS, (w, h))

    total_written = 0
    for scene in scenes:
        clean_frames = deduplicate_frames(scene, all_embeddings, style["frame_similarity"])
        for frame in clean_frames:
            filtered = apply_filter(frame, style["filter"])
            out.write(cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))
            total_written += 1

    out.release()
    print(f"   ✓ Wrote {total_written} frames")
    return total_written


def combine_videos_with_transitions(temp_files, style, output_path):
    """Combine videos with transitions"""
    if not temp_files:
        return

    print("\n🎬 Combining with transitions...")

    cap = cv2.VideoCapture(temp_files[0])
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), FINAL_FPS, (w, h))
    last_frame = None

    for idx, temp_file in enumerate(temp_files):
        print(f"   Adding clip {idx + 1}/{len(temp_files)}")

        cap = cv2.VideoCapture(temp_file)
        first_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if idx > 0 and first_frame is None and last_frame is not None:
                first_frame = frame_rgb

                if style["transition"] == "crossfade":
                    trans = crossfade(last_frame, first_frame, style["transition_frames"])
                elif style["transition"] == "dissolve":
                    trans = dissolve(last_frame, first_frame, style["transition_frames"])
                elif style["transition"] == "fade_to_black":
                    trans = fade_to_black(last_frame, first_frame, style["transition_frames"])
                else:
                    trans = []

                for t in trans:
                    out.write(cv2.cvtColor(t, cv2.COLOR_RGB2BGR))

            out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            last_frame = frame_rgb

        cap.release()

    out.release()


def apply_ai_edits(input_path, edits, output_path, roi=None):
    """Apply AI-suggested edits"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    new_fps = fps * edits.get("speed", 1.0)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), new_fps, (w, h))

    start_frame = int(edits.get("trim_start", 0) * fps)
    end_frame = total_frames - int(edits.get("trim_end", 0) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_gray = None
    idx = start_frame

    print(f"🎬 Applying AI edits...")

    while idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if edits.get("stabilize"):
            if prev_gray is None:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame, prev_gray = apply_stabilization(prev_gray, frame)

        if edits.get("remove_bg"):
            frame = remove_background(frame)

        if edits.get("erase_object") and roi:
            frame = remove_object(frame, roi)

        frame = apply_filter(frame, edits.get("filter", "none"))

        if edits.get("add_logo"):
            frame = add_logo_overlay(frame, LOGO_PATH)

        out.write(frame)
        idx += 1

        if idx % 30 == 0:
            progress = ((idx - start_frame) / (end_frame - start_frame)) * 100
            print(f"   Progress: {progress:.1f}%", end="\r")

    print(f"\n   ✓ Processed {idx - start_frame} frames")

    cap.release()
    out.release()


def semantic_order(clips):
    """Order clips by visual similarity"""
    if len(clips) <= 1:
        return clips

    ordered = [clips.pop(0)]

    while clips:
        last_emb = np.array(ordered[-1]["embedding"])
        best_idx = 0
        best_sim = -1

        for i, c in enumerate(clips):
            emb = np.array(c["embedding"])
            sim = cosine(last_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        ordered.append(clips.pop(best_idx))

    return ordered