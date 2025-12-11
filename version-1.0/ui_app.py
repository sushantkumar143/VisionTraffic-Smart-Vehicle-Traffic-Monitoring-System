import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import easyocr
import json
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64
import plotly.express as px
import plotly.figure_factory as ff

# Custom utilities
from utils import (
    MODEL_PATH, VEHICLE_CLASSES, CLASS_COLORS, 
    append_to_data_stores, calculate_speed_kmh, check_anomaly, set_page_icon,
    create_download_report, text_to_speech, analyze_report_with_llm, 
    ANOMALY_SPEED_THRESHOLD_KMH, calculate_iou 
)

# --- CONFIGURATION & PATH SETUP ---
set_page_icon()
MODEL_ACCURACY_SCORE = "92.5%" 
OCR_READER = easyocr.Reader(['en'], verbose=False) 

# Ensure the data directory exists
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# File names for persistent storage (using full paths)
INPUT_FILE_PATH = os.path.join(DATA_DIR, "input_media.temp")
OUTPUT_FILE_PATH = os.path.join(DATA_DIR, "output_annotated.mp4")
# These must stay in the root/working directory for FPDF to easily find them
LAST_INPUT_FRAME_PATH = "temp_input.png" 
LAST_OUTPUT_FRAME_PATH = "temp_output.png" 


# --- MOCK DATA GENERATION FUNCTION (Ensures Charts are not Empty) ---

def create_mock_results(num_vehicles=30, speed_threshold=80):
    """Generates realistic mock data for chart testing."""
    
    types = np.random.choice(['Car', 'Truck', 'Bus', 'Two Wheeler'], size=num_vehicles, p=[0.5, 0.2, 0.15, 0.15])
    # Ensure speed has variance around the threshold
    speeds = np.random.normal(loc=speed_threshold * 0.9, scale=15, size=num_vehicles).clip(20, 150)
    confidences = np.random.normal(loc=0.9, scale=0.05, size=num_vehicles).clip(0.6, 1.0)

    log_rows = []
    vehicle_speeds_for_chart = {}
    final_anomaly_count = 0
    final_summary_counts = {}

    for i in range(num_vehicles):
        v_type = types[i]
        speed = speeds[i]
        anomaly = "YES" if speed > speed_threshold else "NO"
        if anomaly == "YES": final_anomaly_count += 1
        
        # Simulate speed over 50 frames with fluctuation for insightful speed charts
        simulated_track = np.random.normal(loc=speed, scale=5, size=50).clip(20, 150).tolist()
        vehicle_speeds_for_chart[f"{v_type}_ID{i+1}"] = simulated_track
        
        final_summary_counts[v_type] = final_summary_counts.get(v_type, 0) + 1
        
        log_rows.append({
            "Vehicle_Type": v_type, 
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Date": datetime.now().strftime("%Y-%m-%d"), 
            "Number_Plate": f"MOCK{i:03}",
            "Source": "MOCK", 
            "Detection_Accuracy": round(confidences[i], 4),
            "Avg_Speed_kmh": round(speed, 2), 
            "Anomaly_Detection": anomaly
        })
        
    return {
        'total_count': num_vehicles,
        'counts_by_type': final_summary_counts,
        'avg_speeds': {}, 
        'source': 'MOCK',
        'model_accuracy': MODEL_ACCURACY_SCORE,
        'anomaly_count': final_anomaly_count,
        'detailed_log_rows': log_rows,
        'vehicle_speeds_for_chart': vehicle_speeds_for_chart
    }


# --- Global State Initialization & Model Loading ---
if 'model_loaded' not in st.session_state:
    try:
        st.session_state.model = YOLO(MODEL_PATH)
        st.session_state.model_loaded = True
        st.toast("Model loaded successfully!", icon="✅")
    except Exception as e:
        st.session_state.model_loaded = False
        st.error(f"❌ Failed to load YOLO model: {e}. Run 'main.ipynb' for training/setup.")
        st.stop()
        
if 'ai_summary' not in st.session_state:
    st.session_state.ai_summary = None
    
if 'speed_threshold' not in st.session_state:
    st.session_state.speed_threshold = ANOMALY_SPEED_THRESHOLD_KMH 
    
if 'final_results' not in st.session_state:
    # Initialize session state with MOCK data for immediate chart visibility
    st.session_state.final_results = create_mock_results(speed_threshold=ANOMALY_SPEED_THRESHOLD_KMH)

if 'unique_vehicle_id_counter' not in st.session_state:
    st.session_state.unique_vehicle_id_counter = 0

def clean_temp_files():
    """Removes previous temporary files for memory efficiency."""
    files_to_delete = [INPUT_FILE_PATH, OUTPUT_FILE_PATH, LAST_INPUT_FRAME_PATH, LAST_OUTPUT_FRAME_PATH]
    for f in files_to_delete:
        if os.path.exists(f):
            try:
                os.unlink(f)
            except Exception as e:
                print(f"Error deleting file {f}: {e}")


# --- Custom Aesthetic Loading Animation Function ---

def show_processing_ui(text, progress, placeholder):
    """Displays a custom, full-width processing banner and progress bar in a placeholder."""
    
    placeholder.markdown(f"""
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    .custom-loader {{
        width: 100%;
        padding: 20px;
        background-color: #333; 
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: white;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }}
    .spinner-icon {{
        font-size: 40px;
        animation: spin 1.5s linear infinite;
        margin-bottom: 10px;
        color: #1E90FF; 
    }}
    .progress-bar-container {{
        width: 80%;
        margin-top: 10px;
    }}
    </style>
    <div class="custom-loader">
        <div class="spinner-icon">⚙️</div>
        <div>{text} ({progress}%)</div>
        <div class="progress-bar-container">
            <progress value="{progress}" max="100" style="width: 100%; height: 10px; border-radius: 5px;"></progress>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- Core Processing Function (Unchanged, ensures logging of plate text) ---

def process_frame(frame, fps=None, tracked_vehicles=None, is_video=False):
    """
    Performs detection, annotation, and tracking. 
    Returns frame and tracking data.
    """
    speed_threshold = st.session_state.speed_threshold 
    anomaly_detection_on = st.session_state.get('anomaly_toggle', True)

    log_rows = []
    res = st.session_state.model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
    
    total_count = 0
    counts_by_type = {}
    
    # --- IMAGE/CAMERA MODE LOGIC ---
    if not is_video:
        
        for i, box in enumerate(res.boxes.data.cpu().numpy()):
            x1, y1, x2, y2, conf, cls_float = map(float, box[:6])
            x1, y1, x2, y2, cls = map(int, [x1, y1, x2, y2, cls_float])
            
            class_name = VEHICLE_CLASSES.get(cls, "Unknown")
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            
            is_vehicle = cls not in [1, 2] 
            speed_text = ""
            speed_kmh = 0.0
            anomaly_status = "NA"
            plate_text = "N/A"
            
            if is_vehicle:
                total_count += 1
                counts_by_type[class_name] = counts_by_type.get(class_name, 0) + 1
                
                # OCR extraction logic (kept the same)
                for b in res.boxes.data.cpu().numpy(): 
                    plate_cls = int(b[5])
                    if plate_cls in [1, 2] and b[0] > x1 and b[1] > y1 and b[2] < x2 and b[3] < y2:
                        plate_x1, plate_y1, plate_x2, plate_y2, _, _ = map(float, b[:6])
                        crop = frame[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                        if crop is not None and crop.size > 0:
                            ocr_results = OCR_READER.readtext(crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                            if ocr_results:
                                plate_text = ocr_results[0][1].replace(" ", "")
                        break
                
                # NOTE: Avg_Speed_kmh is 0 for static image
                log_rows.append({
                    "Vehicle_Type": class_name, "Time": datetime.now().strftime("%H:%M:%S"),
                    "Date": datetime.now().strftime("%Y-%m-%d"), "Number_Plate": plate_text,
                    "Source": "Image", "Detection_Accuracy": round(conf, 4),
                    "Avg_Speed_kmh": round(speed_kmh, 2), 
                    "Anomaly_Detection": anomaly_status
                })
            
            label = f"{class_name} {conf:.2f}{speed_text}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) 
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        avg_speeds = {v_type: 0.0 for v_type in counts_by_type.keys()}
        
        return frame, total_count, counts_by_type, None, log_rows, avg_speeds

    # --- VIDEO MODE LOGIC (Tracking for Debouncing) ---
    
    # 1. Process Detections and Match with Tracked Vehicles
    for i, box in enumerate(res.boxes.data.cpu().numpy()):
        x1, y1, x2, y2, conf, cls_float = map(float, box[:6])
        x1, y1, x2, y2, cls = map(int, [x1, y1, x2, y2, cls_float])
        
        class_name = VEHICLE_CLASSES.get(cls, "Unknown")
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        
        is_vehicle = cls not in [1, 2] 
        
        if is_vehicle:
            current_bbox = [x1, y1, x2, y2]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            matched_id = None
            max_iou = 0.0
            
            # Simple tracking: Find best match from the previous frame's tracks
            for uid, track_data in tracked_vehicles.items():
                if track_data['class'] == class_name and track_data['latest_bbox'] is not None:
                    iou_score = calculate_iou(current_bbox, track_data['latest_bbox'])
                    
                    if iou_score > 0.5 and iou_score > max_iou: 
                        max_iou = iou_score
                        matched_id = uid
            
            # --- Update/Create Track ---
            
            if matched_id is not None:
                track_data = tracked_vehicles[matched_id]
                
                speed_text = ""
                anomaly_status = "NO"
                if track_data['latest_centroid']:
                    prev_cx, prev_cy = track_data['latest_centroid']
                    dx = cx - prev_cx
                    dy = cy - prev_cy
                    speed_kmh = calculate_speed_kmh(dx, dy, fps)
                    
                    track_data['speeds'].append(speed_kmh)
                    
                    anomaly_status = check_anomaly(speed_kmh, speed_threshold)
                    
                    if anomaly_detection_on and anomaly_status == "YES":
                        color = (0, 0, 255)
                        
                    speed_text = f" | {speed_kmh:.1f} km/h"
                
                track_data['latest_centroid'] = (cx, cy)
                track_data['latest_bbox'] = current_bbox
                
                track_data['log_row']['Detection_Accuracy'] = max(track_data['log_row']['Detection_Accuracy'], round(conf, 4))
                
            else:
                matched_id = st.session_state.unique_vehicle_id_counter
                st.session_state.unique_vehicle_id_counter += 1
                
                speed_text = ""
                anomaly_status = "NO"
                
                tracked_vehicles[matched_id] = {
                    'class': class_name,
                    'latest_centroid': (cx, cy),
                    'latest_bbox': current_bbox,
                    'speeds': [],
                    'log_row': { 
                        "Vehicle_Type": class_name,
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Date": datetime.now().strftime("%Y-%m-%d"),
                        "Number_Plate": "N/A", 
                        "Source": "Video",
                        "Detection_Accuracy": round(conf, 4),
                        "Avg_Speed_kmh": 0.0,
                        "Anomaly_Detection": "NO"
                    }
                }
            
            # OCR Extraction 
            plate_text = tracked_vehicles[matched_id]['log_row'].get("Number_Plate", "N/A")
            if plate_text == "N/A":
                for b in res.boxes.data.cpu().numpy(): 
                    plate_cls = int(b[5])
                    if plate_cls in [1, 2] and b[0] > x1 and b[1] > y1 and b[2] < x2 and b[3] < y2:
                        plate_x1, plate_y1, plate_x2, plate_y2, _, _ = map(float, b[:6])
                        crop = frame[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]
                        if crop is not None and crop.size > 0:
                            ocr_results = OCR_READER.readtext(crop, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                            if ocr_results:
                                plate_text = ocr_results[0][1].replace(" ", "")
                                tracked_vehicles[matched_id]['log_row']["Number_Plate"] = plate_text
                        break
                
            # Annotation
            label = f"{class_name} ID:{matched_id} {conf:.2f}{speed_text}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if anomaly_status == "YES" else 2) 
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            total_count += 1
            counts_by_type[class_name] = counts_by_type.get(class_name, 0) + 1
        
        else:
             # Just draw the plate boxes (cls 1 and 2)
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1) 
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


    return frame, total_count, counts_by_type, tracked_vehicles, log_rows, {}


# --- Graphical Analysis Helper Functions ---

def display_summary_dashboard(results: dict):
    st.header("📊 Summary Dashboard: Composition & Speed")
    df_analysis = pd.DataFrame(results.get('detailed_log_rows', []))
    
    if df_analysis.empty:
        st.warning("No data available for analysis. Please process a video or image first.")
        return

    # Tabular Analysis
    st.subheader("Tabular Summary (Detection & Violation)")
    df_display = df_analysis[[
        "Vehicle_Type", 
        "Avg_Speed_kmh", 
        "Detection_Accuracy", 
        "Anomaly_Detection", 
        "Number_Plate"
    ]].rename(columns={
        "Avg_Speed_kmh": f"Avg Speed (km/h) > {st.session_state.speed_threshold}",
        "Detection_Accuracy": "Accuracy",
        "Anomaly_Detection": "Violation"
    })
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Graphical Analysis
    st.subheader("Vehicle Composition Charts")
    df_counts = df_analysis.groupby('Vehicle_Type').size().reset_index(name='Count')

    col_pie, col_bar_speed = st.columns(2)
    
    # Pie Chart
    with col_pie:
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        ax_pie.pie(
            df_counts['Count'], 
            labels=df_counts['Vehicle_Type'], 
            autopct='%1.1f%%', 
            startangle=90,
            wedgeprops={'edgecolor': 'black'}
        )
        ax_pie.axis('equal') 
        ax_pie.set_title("Vehicle Type Composition", fontsize=14)
        st.pyplot(fig_pie)
        
    # Bar Chart (Average Speed / Accuracy Proxy) - FIX FOR IMAGE INPUT
    with col_bar_speed:
        source = results.get('source', 'N/A')
        
        if source == 'Video' or df_analysis['Avg_Speed_kmh'].max() > 0.1:
            # Display Average Speed for video input or if speed is non-zero
            df_plot = df_analysis.groupby('Vehicle_Type')['Avg_Speed_kmh'].mean().reset_index(name='Metric')
            y_label = 'Average Speed (km/h)'
            title_suffix = 'Speed'
        else:
            # Display Detection Accuracy for static image input (as speed is 0)
            df_plot = df_analysis.groupby('Vehicle_Type')['Detection_Accuracy'].mean().reset_index(name='Metric')
            y_label = 'Avg. Detection Accuracy'
            title_suffix = 'Detection Accuracy'
        
        fig_bar, ax_bar = plt.subplots(figsize=(6, 6))
        
        if not df_plot.empty:
            colors_rgb_norm = []
            for v_type in df_plot['Vehicle_Type']:
                bgr_color = CLASS_COLORS.get(next((k for k, v in VEHICLE_CLASSES.items() if v == v_type), 0), (255, 255, 255))
                colors_rgb_norm.append((bgr_color[2] / 255, bgr_color[1] / 255, bgr_color[0] / 255))
            
            ax_bar.bar(df_plot['Vehicle_Type'], df_plot['Metric'], color=colors_rgb_norm)
            ax_bar.set_ylabel(y_label)
            ax_bar.set_title(f'Vehicle Type by {title_suffix}', fontsize=14)
            ax_bar.tick_params(axis='x', rotation=45)
        else:
             ax_bar.text(0.5, 0.5, 'No Data', transform=ax_bar.transAxes, ha='center')

        st.pyplot(fig_bar)


def display_speed_variability(results: dict):
    st.header("⚡ Speed Variability Charts (Vehicle ID Tracking)")
    
    vehicle_speeds = results.get('vehicle_speeds_for_chart', {})
    if not vehicle_speeds:
        st.warning("Speed variability data is only available for **Video Upload** analysis or the Mock Data Generator.")
        return
    
    # Flatten the data for Plotly
    data_list = []
    # Only plot the first 10 tracks for visual clarity in the UI
    for uid, speeds in list(vehicle_speeds.items())[:10]:
        v_type = uid.split('_')[0]
        data_list.extend([{'Vehicle_ID': uid, 'Vehicle_Type': v_type, 'Speed': s, 'Frame': i} for i, s in enumerate(speeds)])
    
    df_speed = pd.DataFrame(data_list)
    
    if df_speed.empty:
        st.error("Speed data frame is empty. No speed readings recorded during video processing.")
        return

    # Speed over Time (Frame) - Plotly 
    st.subheader("Time-Series Speed Profile: Insights into Acceleration/Deceleration")
    fig_line = px.line(df_speed, x='Frame', y='Speed', color='Vehicle_ID', 
                       title='Speed of Tracked Vehicles Over Frame Index',
                       hover_data=['Vehicle_Type'])
    fig_line.add_hline(y=st.session_state.speed_threshold, line_dash="dash", line_color="red", 
                       annotation_text=f"Anomaly Threshold ({st.session_state.speed_threshold} km/h)")
    st.plotly_chart(fig_line, use_container_width=True)

    # Speed Distribution Histogram 
    st.subheader("Overall Speed Distribution (Identifying Peak Traffic Speed)")
    fig_hist = px.histogram(df_speed, x='Speed', color='Vehicle_Type', marginal="box",
                            nbins=20,
                            title="Distribution of Instantaneous Speed Readings")
    st.plotly_chart(fig_hist, use_container_width=True)


# --- REVISED Input/Output Comparison Function (Fixed Video Playback) ---
def display_input_output_comparison(results: dict):
    st.header("🖼️ Frame-by-Frame Video Analysis & Comparison")
    
    source = results.get('source', 'N/A')
    st.subheader(f"Input vs. Annotated Output ({source})")

    col_input_comp, col_output_comp = st.columns(2)
    
    if not results or source == 'N/A':
        st.warning("No media processed yet. Please run an analysis first.")
        return

    # --- VIDEO MODE DISPLAY ---
    if source == 'Video':
        # Input Video: Load from the persistent temp file
        col_input_comp.subheader("Original Input Video")
        if os.path.exists(INPUT_FILE_PATH):
             with open(INPUT_FILE_PATH, 'rb') as f_in:
                input_bytes = f_in.read()
             # FIX: Display the input video from its saved location
             col_input_comp.video(input_bytes, format='video/mp4') 
        else:
             col_input_comp.info("Original video file not found on disk.")
             
        # Annotated Video (Playable)
        col_output_comp.subheader("Annotated Output Video (Playable)")
        if os.path.exists(OUTPUT_FILE_PATH):
            with open(OUTPUT_FILE_PATH, 'rb') as f:
                video_bytes = f.read()
            # FIX: Display the output video from its saved location
            col_output_comp.video(video_bytes, format='video/mp4')
        else:
             col_output_comp.error("Annotated video file not found.")


    # --- IMAGE/CAMERA MODE DISPLAY ---
    elif source in ['Image', 'Camera', 'MOCK']: # MOCK data uses this section conceptually
        
        # We only display if the files exist (i.e., if processing was run on actual media)
        if os.path.exists(LAST_INPUT_FRAME_PATH) and os.path.exists(LAST_OUTPUT_FRAME_PATH):
            col_input_comp.subheader(f"Original Input ({source})")
            col_input_comp.image(LAST_INPUT_FRAME_PATH, channels="BGR", use_column_width=True)
            
            col_output_comp.subheader("Annotated Output")
            col_output_comp.image(LAST_OUTPUT_FRAME_PATH, channels="BGR", use_column_width=True)
        else:
            col_input_comp.info("Input image not available.")
            col_output_comp.info("Annotated output not available.")


    # Show detailed log for frame-level review (includes Plate Number)
    st.subheader("Detailed Log for Frame Review (First 10 Entries)")
    df_analysis = pd.DataFrame(results.get('detailed_log_rows', []))
    
    st.caption("First 10 Debounced Entries for Frame-Level Review (Check 'Number_Plate' column for detection results)")
    st.dataframe(df_analysis.head(10), use_container_width=True)


# --- MAIN STREAMLIT APP LOGIC ---

st.title("🚗 Advanced Vehicle Analytics Dashboard")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Model & Controls")
st.sidebar.metric(
    label="Model Prediction Accuracy",
    value=MODEL_ACCURACY_SCORE,
    delta="YOLOv8n"
)
# Custom Speed Limit Setter
st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Speed Limit Setter (km/h)")
st.session_state.speed_threshold = st.sidebar.slider(
    'Set Anomaly Speed Threshold', 
    min_value=40, max_value=200, 
    value=st.session_state.speed_threshold, 
    step=5,
    key="speed_slider",
    help="Vehicles detected faster than this limit will be flagged as anomalies (red boxes in video)."
)
# REMOVED: Generate Mock Data for Charts button


st.sidebar.markdown("---")
# AI Feature Toggles 
st.sidebar.header("🤖 AI & Utility Features")
ai_assistant_on = st.sidebar.toggle('AI Assistant & Analysis', value=False, key="ai_toggle")
voice_assistant_on = st.sidebar.toggle('Voice Assistant (Report Analysis)', value=False, key="voice_toggle")
anomaly_detection_on = st.sidebar.toggle('Anomaly Detection (Enabled)', value=True, key="anomaly_toggle")
st.sidebar.markdown("---")

# --- Sidebar Navigation for Graphical Analysis (Updated) ---
analysis_selection = st.sidebar.radio(
    "📊 Select Report Section",
    [
        "🏠 Input & Processing",
        "📊 Summary Dashboard",
        "🖼️ Frame-by-Frame Video Analysis", 
        "📄 Download Report"
    ],
    index=0 
)
st.sidebar.markdown("---")

# Global placeholder for custom loading 
loading_placeholder = st.empty()


# --- Main Content Rendered based on Sidebar Selection ---

if analysis_selection == "🏠 Input & Processing":
    st.header("Input & Processing: Source Selection")
    
    # Input Selection
    input_mode = st.radio(
        "Select Input Mode:",
        ('Image Upload', 'Video Upload', 'Live Camera (Image)'),
        horizontal=True
    )
    st.markdown("---")
    
    col_input, col_output = st.columns(2)

    # --- INPUT MODE PROCESSING LOGIC ---

    if input_mode == 'Image Upload':
        uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'], key="image_uploader")
        
        if uploaded_file is not None:
            clean_temp_files() 
            
            with loading_placeholder.container():
                show_processing_ui("Initial Processing...", 0, loading_placeholder)
            
            try:
                file_bytes = uploaded_file.read()
                with open(INPUT_FILE_PATH, "wb") as f_out:
                    f_out.write(file_bytes)
                
                file_bytes_np = np.asarray(bytearray(file_bytes), dtype=np.uint8)
                frame_input = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)

                if frame_input is None:
                    st.error("Error: Could not decode the image file.")
                    loading_placeholder.empty()
                    st.stop()
                
                show_processing_ui("Running YOLO Inference...", 30, loading_placeholder)
                
                cv2.imwrite(LAST_INPUT_FRAME_PATH, frame_input)

                frame_ann, total, counts, _, log_rows, avg_speeds = process_frame(frame_input, is_video=False)
                
                show_processing_ui("Logging Data (Excel & MySQL)...", 60, loading_placeholder)

                cv2.imwrite(LAST_OUTPUT_FRAME_PATH, frame_ann)
                
                with col_input:
                    st.header("🖼️ Input Image")
                    st.image(LAST_INPUT_FRAME_PATH, channels="BGR", use_column_width=True)
                with col_output:
                    st.header("✅ Annotated Output")
                    st.image(LAST_OUTPUT_FRAME_PATH, channels="BGR", use_column_width=True)
                
                append_to_data_stores(log_rows)
                
                show_processing_ui("Finalizing Analysis...", 90, loading_placeholder)
                
                st.session_state.final_results = {
                    'total_count': total,
                    'counts_by_type': counts,
                    'avg_speeds': avg_speeds,
                    'source': 'Image',
                    'model_accuracy': MODEL_ACCURACY_SCORE,
                    'anomaly_count': len([r for r in log_rows if r.get('Anomaly_Detection') == 'YES']),
                    'detailed_log_rows': log_rows,
                    'vehicle_speeds_for_chart': {} 
                }
                st.session_state.ai_summary = None 
                loading_placeholder.empty()

            except Exception as e:
                loading_placeholder.empty()
                st.error(f"An error occurred during image processing: {e}")
                st.stop()
                
    elif input_mode == 'Video Upload':
        uploaded_file = st.file_uploader("Upload a Video", type=['mp4', 'avi', 'mov'], key="video_uploader")
        
        if uploaded_file is not None:
            clean_temp_files()

            with loading_placeholder.container():
                show_processing_ui("Uploading Video File...", 0, loading_placeholder)
            
            file_bytes = uploaded_file.read()
            with open(INPUT_FILE_PATH, "wb") as f_out:
                f_out.write(file_bytes)
            
            video_path = INPUT_FILE_PATH 

            with loading_placeholder.container():
                show_processing_ui("Initializing Video Processing...", 5, loading_placeholder)

            col_input.header("🎥 Input Video")
            col_output.header("✨ Annotated Output")
            
            input_placeholder = col_input.empty()
            output_placeholder = col_output.empty()

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            out_path = OUTPUT_FILE_PATH
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            tracked_vehicles = {}
            all_log_rows = []
            frame_idx = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0: total_frames = 1000 
            
            last_input_frame = None
            last_annotated_frame = None
            
            try:
                while cap.isOpened():
                    ret, frame_input = cap.read()
                    if not ret: break
                    
                    frame_ann, total, counts, tracked_vehicles, _, _ = process_frame(
                        frame_input, fps, tracked_vehicles, is_video=True
                    )
                    
                    out.write(frame_ann)
                    frame_idx += 1
                    
                    last_input_frame = frame_input.copy()
                    last_annotated_frame = frame_ann.copy()

                    progress = min(1.0, frame_idx / total_frames)
                    progress_percent = int(progress * 100)
                    
                    with loading_placeholder.container():
                        show_processing_ui(f"🎬 Processing Frame {frame_idx}... Total Detections: {total}", progress_percent, loading_placeholder)
                    
                    if frame_idx % 5 == 0:
                        input_placeholder.image(last_input_frame, channels="BGR", caption="Input Frame", use_column_width=True)
                        output_placeholder.image(last_annotated_frame, channels="BGR", caption="Annotated Frame", use_column_width=True)
                    
                cap.release()
                out.release()
                
                # --- DEBOUNCING / FINAL LOGGING LOGIC ---
                
                final_summary_counts = {}
                final_anomaly_count = 0
                vehicle_speeds_for_chart = {}
                
                for uid, track_data in tracked_vehicles.items():
                    log_row = track_data['log_row']
                    avg_speed = np.mean(track_data['speeds']) if track_data['speeds'] else 0.0
                    log_row['Avg_Speed_kmh'] = round(avg_speed, 2)
                    
                    is_anomaly = check_anomaly(avg_speed, st.session_state.speed_threshold)
                    log_row['Anomaly_Detection'] = is_anomaly
                    
                    if is_anomaly == "YES":
                        final_anomaly_count += 1
                    
                    all_log_rows.append(log_row)
                    
                    v_type = log_row['Vehicle_Type']
                    final_summary_counts[v_type] = final_summary_counts.get(v_type, 0) + 1
                    vehicle_speeds_for_chart[f"{v_type}_ID{uid}"] = track_data['speeds']

                final_total_count = len(all_log_rows)

                if last_input_frame is not None and last_annotated_frame is not None:
                    cv2.imwrite(LAST_INPUT_FRAME_PATH, last_input_frame)
                    cv2.imwrite(LAST_OUTPUT_FRAME_PATH, last_annotated_frame)
                
                with loading_placeholder.container():
                    show_processing_ui("Finalizing and Logging Data...", 95, loading_placeholder)
                    time.sleep(1)
                
                loading_placeholder.empty()
                st.success("Video Analysis Complete! Charts are now populated.")
                
                # FIX: Display input video from saved file
                if os.path.exists(INPUT_FILE_PATH):
                    with open(INPUT_FILE_PATH, 'rb') as f_in:
                        input_bytes = f_in.read()
                    input_placeholder.video(input_bytes)
                
                # Display the final annotated video for viewing
                with open(out_path, 'rb') as f:
                    video_bytes = f.read()
                output_placeholder.video(video_bytes)
                
                # Set final results 
                st.session_state.final_results = {
                    'total_count': final_total_count,
                    'counts_by_type': final_summary_counts, 
                    'avg_speeds': {}, 
                    'source': 'Video',
                    'model_accuracy': MODEL_ACCURACY_SCORE,
                    'anomaly_count': final_anomaly_count,
                    'detailed_log_rows': all_log_rows,
                    'vehicle_speeds_for_chart': vehicle_speeds_for_chart 
                }
                st.session_state.ai_summary = None 
            
            except Exception as e:
                loading_placeholder.empty()
                st.error(f"An error occurred during video processing: {e}")

    elif input_mode == 'Live Camera (Image)':
        camera_image = st.camera_input("Take a Photo", key="camera_input")

        if camera_image is not None:
            clean_temp_files()
            
            with loading_placeholder.container():
                show_processing_ui("📸 Processing Live Photo...", 10, loading_placeholder)
            
            try:
                img = Image.open(camera_image)
                frame_input = np.array(img)
                frame_input = cv2.cvtColor(frame_input, cv2.COLOR_RGB2BGR)

                show_processing_ui("🔎 Running YOLO Inference...", 30, loading_placeholder)
                
                cv2.imwrite(LAST_INPUT_FRAME_PATH, frame_input)
                img.save(INPUT_FILE_PATH, format="PNG") 

                frame_ann, total, counts, _, log_rows, avg_speeds = process_frame(frame_input, is_video=False)
                
                show_processing_ui("💾 Logging Data (Excel & MySQL)...", 60, loading_placeholder)

                cv2.imwrite(LAST_OUTPUT_FRAME_PATH, frame_ann)
                
                with col_input:
                    st.header("🖼️ Input Photo")
                    st.image(LAST_INPUT_FRAME_PATH, channels="BGR", use_column_width=True)
                with col_output:
                    st.header("✅ Annotated Output")
                    st.image(LAST_OUTPUT_FRAME_PATH, channels="BGR", use_column_width=True)
                
                append_to_data_stores(log_rows)
                
                show_processing_ui("Finalizing Analysis...", 90, loading_placeholder)
                
                loading_placeholder.empty()
                st.success("Photo Analysis Complete!")

                st.session_state.final_results = {
                    'total_count': total,
                    'counts_by_type': counts,
                    'avg_speeds': avg_speeds,
                    'source': 'Camera',
                    'model_accuracy': MODEL_ACCURACY_SCORE,
                    'anomaly_count': 0,
                    'detailed_log_rows': log_rows,
                    'vehicle_speeds_for_chart': {}
                }
                st.session_state.ai_summary = None 

            except Exception as e:
                loading_placeholder.empty()
                st.error(f"An error occurred during camera processing: {e}")

    else:
        col_input.empty()
        col_output.empty()
        st.info("Upload an image or video, or take a photo to start the advanced analysis.")


    # --- Download Button in Input & Processing Section (NEW) ---
    st.markdown("---")
    st.subheader("📄 Generate & Download Report")
    
    if st.session_state.final_results and st.session_state.final_results['source'] != 'MOCK':
        
        # Check if AI Analysis is toggled on but not run yet
        if ai_assistant_on and st.session_state.ai_summary is None:
            st.info("AI Analysis is enabled. Synthesizing report summary...")
            with st.spinner('Synthesizing professional narrative...'):
                ai_summary_text = analyze_report_with_llm(st.session_state.final_results, st.session_state.speed_threshold)
            st.session_state.ai_summary = ai_summary_text
            st.toast("AI summary prepared!")

        # Create PDF
        pdf_base64 = create_download_report(st.session_state.final_results, st.session_state.speed_threshold) 
        report_filename = f"Vehicle_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        col_dl_ip_btn, col_dl_ip_voice = st.columns(2)
        
        # Download button
        href = f'<a href="data:application/octet-stream;base64,{pdf_base64}" download="{report_filename}" style="display: inline-flex; align-items: center; justify-content: center; background-color: #1E90FF; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; width: 100%;">Download Full PDF Report</a>'
        col_dl_ip_btn.markdown(href, unsafe_allow_html=True)
        
        # Voice Assistant Integration (re-enabled logic)
        if voice_assistant_on and st.session_state.ai_summary:
            mp3_path = text_to_speech(st.session_state.ai_summary)
            if os.path.exists(mp3_path):
                 with open(mp3_path, 'rb') as audio_file:
                     col_dl_ip_voice.audio(audio_file.read(), format='audio/mp3')
                 col_dl_ip_voice.info("AI Summary Audio Ready.")
    else:
        st.info("Complete video/image processing to enable report generation.")


elif st.session_state.get('final_results') is None:
    st.info("Please navigate to **🏠 Input & Processing** and analyze a video or image first to view the reports.")

elif analysis_selection == "📊 Summary Dashboard":
    display_summary_dashboard(st.session_state.final_results)
    
elif analysis_selection == "🖼️ Frame-by-Frame Video Analysis": 
    display_input_output_comparison(st.session_state.final_results)

elif analysis_selection == "📄 Download Report":
    
    st.header("📄 Download Full Report & Assets")
    results = st.session_state.final_results

    # --- AI Assistant Check ---
    if ai_assistant_on and st.session_state.ai_summary is None:
        st.info("AI Analysis is enabled. Running AI Assistant to synthesize the professional report...")
        
        with st.spinner('Synthesizing professional narrative...'):
            ai_summary_text = analyze_report_with_llm(results, st.session_state.speed_threshold)
        
        st.session_state.ai_summary = ai_summary_text
        results['ai_summary'] = ai_summary_text 
        st.session_state.final_results = results
        st.success("✅ AI Analysis Complete! Report is now ready.")
    
    if st.session_state.ai_summary is not None:
        
        st.subheader("AI Assistant Summary")
        st.markdown(st.session_state.ai_summary)
        
        if voice_assistant_on:
            mp3_path = text_to_speech(st.session_state.ai_summary)
            if os.path.exists(mp3_path):
                 st.audio(open(mp3_path, 'rb').read(), format='audio/mp3')

        st.markdown("---")
        
        # --- Download Options ---
        st.subheader("⬇️ Download Options")

        col_dl_output, col_dl_report = st.columns(2)

        # 1. Download Annotated Output 
        output_filename = f"annotated_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mime_type = "image/png" if results['source'] != 'Video' else "video/mp4"
        file_path = LAST_OUTPUT_FRAME_PATH if results['source'] != 'Video' else OUTPUT_FILE_PATH

        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                col_dl_output.download_button(
                    label=f"Download Annotated {results['source']}",
                    data=file,
                    file_name=f"{output_filename}.{mime_type.split('/')[1]}",
                    mime=mime_type,
                    type="primary"
                )
        
        # 2. Download Complete Report (PDF)
        report_filename = f"Vehicle_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_base64 = create_download_report(results, st.session_state.speed_threshold) 
        
        # Aesthetic HTML download button
        href = f'<a href="data:application/octet-stream;base64,{pdf_base64}" download="{report_filename}" style="display: inline-flex; align-items: center; justify-content: center; background-color: #1E90FF; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; width: 100%;">Download Complete PDF Report</a>'
        col_dl_report.markdown(href, unsafe_allow_html=True)
    
    else:
        st.info("Run AI Analysis via the sidebar toggle to unlock the PDF Report download.")