import streamlit as st
import os
import json
import cv2
import gc
import torch
from config import VIDEO_DIR, ANALYSIS_JSON, TEMP_DIR, FINAL_VIDEO, FINAL_FPS, OUTPUT_DIR
from processing import semantic_order, process_clip_to_file, combine_videos_with_transitions, apply_ai_edits
from clip_blip import STYLES
from video_analysis import extract_and_analyze_video
from gemini_ai import API_KEY, get_gemini_instructions

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="L&H AI Studio", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stButton > button:first-child {
        background-color: #6366f1;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stTextArea textarea {
        background-color: #1a1d2e;
        color: white;
        border: 1px solid #334155;
    }
    .video-card {
        border: 2px solid #334155;
        border-radius: 12px;
        padding: 5px;
        background: #000;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    # Logo Integration
    if os.path.exists("l&h_logo.png"):
        st.sidebar.image("l&h_logo.png")

    st.title("🎥AI Video Editor")

    # --- STEP 1: SIDEBAR ASSETS ---
    with st.sidebar:
        st.header("1. Project Assets")
        uploaded_files = st.file_uploader("Upload internal system clips", type=['mp4', 'mov', 'avi'],
                                          accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Internally saving to your system path defined in VIDEO_DIR
                path = os.path.join(VIDEO_DIR, uploaded_file.name)
                with open(path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"Internal System updated: {len(uploaded_files)} clips.")

        st.divider()
        st.header("2. Base Settings")
        style_name = st.selectbox("Visual Style", options=list(STYLES.keys()), index=0)
        use_semantic = st.toggle("Smart Clip Ordering", value=True)

        if st.button("Generate Base Video"):
            process_base_video(style_name, use_semantic)

    # --- MAIN CONTENT AREA: PREVIEW & AI PANEL ---
    if os.path.exists(FINAL_VIDEO):
        # UI split: Left for Internal View, Right for AI Control
        col_preview, col_panel = st.columns([2, 1])

        with col_preview:
            # st.subheader("System Preview (Internal View)")
            # This renders the file directly from your internal system path
            st.markdown('<div class="video-card">', unsafe_allow_html=True)
            st.video(FINAL_VIDEO)
            st.markdown('</div>', unsafe_allow_html=True)

            # st.success(f"File location: {os.path.abspath(FINAL_VIDEO)}")

        with col_panel:
            st.subheader("AI Edits")
            user_prompt = st.text_area(
                "Describe changes:",
                placeholder="e.g. 'Make it faster and apply cinematic color'",
                height=180
            )

            if st.button("Apply AI Edits"):
                if API_KEY:
                    process_ai_edits(user_prompt)
                else:
                    st.error("API Key not found.")

            st.divider()

            st.subheader("System Export")
            # User downloads the video only after viewing the internal data
            with open(FINAL_VIDEO, "rb") as f:
                st.download_button(
                    label="Download Final Result",
                    data=f,
                    file_name="ai_output.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
    else:
        st.info("System Ready. Please upload clips and generate the base video to view data.")


# --- CORE LOGIC FUNCTIONS ---

def process_base_video(style_name, use_semantic):
    with st.status("🎬 Processing internal files...", expanded=True) as status:
        videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi"))])
        if not videos:
            st.error("Internal VIDEO_DIR is empty.")
            return

        clips = []
        all_analysis = []
        for video_name in videos:
            video_path = os.path.join(VIDEO_DIR, video_name)
            analysis = extract_and_analyze_video(video_path)
            if analysis["frames"]:
                clips.append({"name": video_name, "path": video_path, "frames": analysis["frames"],
                              "embedding": analysis["embedding"], "duration": analysis["duration"],
                              "captions": analysis["captions"]})
                all_analysis.append(
                    {"file": video_name, "duration": analysis["duration"], "captions": analysis["captions"]})
            gc.collect()

        if use_semantic:
            clips = semantic_order(clips)

        temp_files = []
        for i, clip in enumerate(clips):
            temp_path = os.path.join(TEMP_DIR, f"temp_{i:03d}.mp4")
            process_clip_to_file(clip, STYLES[style_name], temp_path)
            temp_files.append(temp_path)
            del clip["frames"]
            gc.collect()

        combine_videos_with_transitions(temp_files, STYLES[style_name], FINAL_VIDEO)
        status.update(label="Internal View Ready!", state="complete")
        st.rerun()


def process_ai_edits(prompt):
    with st.spinner("Applying AI instructions..."):
        if os.path.exists(ANALYSIS_JSON):
            with open(ANALYSIS_JSON, 'r') as f:
                all_analysis = json.load(f)
        else:
            all_analysis = []

        edits = get_gemini_instructions(prompt, {"clips": all_analysis})
        ai_output = os.path.join(OUTPUT_DIR, "ai_final_edit.mp4")
        apply_ai_edits(FINAL_VIDEO, edits, ai_output, None)

        st.success("AI Edit Finished!")
        # Renders the final AI video directly from the system path
        st.video(ai_output)


if __name__ == "__main__":
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()