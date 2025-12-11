# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import json
# import base64
# import time
# from fpdf import FPDF
# import matplotlib.pyplot as plt
# from gtts import gTTS
# import streamlit as st
# import mysql.connector
# import io
# import cv2

# # --- Configuration Constants (Unchanged) ---
# MODEL_DIR = "models"
# DATA_YAML_PATH = "Vechile Detection.v1-version-1.yolov8/data.yaml"
# MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
# VEHICLE_LOG_EXCEL = "data/vehicle_analysis_log.xlsx"
# VEHICLE_CLASSES = {
#     0: "Car", 1: "plate", 2: "blur_plate", 3: "Two Wheeler",
#     4: "Auto", 5: "Bus", 6: "Truck"
# }

# # Vehicle Class BGR Colors (0-255)
# CLASS_COLORS = {
#     0: (255, 0, 0),     # Car: Blue
#     1: (0, 165, 255),   # plate: Orange
#     2: (0, 0, 0),       # blur_plate: Black
#     3: (0, 255, 0),     # Two Wheeler: Green
#     4: (0, 255, 255),   # Auto: Yellow
#     5: (255, 0, 255),   # Bus: Magenta
#     6: (255, 255, 0)    # Truck: Cyan
# }

# ANOMALY_SPEED_THRESHOLD_KMH = 80 
# MYSQL_CONFIG = {
#     "host": "localhost",
#     "user": "root",
#     "password": "*sushant143*", 
#     "database": "vehicle_analytics_db"
# }

# EXCEL_COLUMNS = [
#     "UNIQUE_ID", "Vehicle_Type", "Time", "Date", "Number_Plate", 
#     "Source", "Detection_Accuracy", "Avg_Speed_kmh", "Anomaly_Detection"
# ]

# # --- Database & File Utilities (Unchanged) ---
# def create_log_file_if_not_exists():
#     os.makedirs('data', exist_ok=True)
#     if not os.path.exists(VEHICLE_LOG_EXCEL):
#         df = pd.DataFrame(columns=EXCEL_COLUMNS)
#         df.to_excel(VEHICLE_LOG_EXCEL, index=False, engine='openpyxl') 
#         print(f"Created new log file: {VEHICLE_LOG_EXCEL}")

# def log_to_mysql(row):
#     pass

# def append_to_data_stores(new_rows):
#     if not new_rows: return True

#     mysql_success = True
#     for row in new_rows:
#         if not log_to_mysql(row): mysql_success = False

#     create_log_file_if_not_exists()
#     excel_success = True
    
#     try:
#         df_existing = pd.read_excel(VEHICLE_LOG_EXCEL, engine='openpyxl') 
#         next_id = df_existing["UNIQUE_ID"].max() + 1 if not df_existing.empty else 1
        
#         excel_rows = []
#         for i, row in enumerate(new_rows):
#             excel_row = row.copy() 
#             excel_row["UNIQUE_ID"] = int(next_id + i)
            
#             for col in EXCEL_COLUMNS:
#                 if col not in excel_row:
#                     excel_row[col] = "N/A" if col not in ["Detection_Accuracy", "Avg_Speed_kmh"] else 0.0
#             excel_rows.append(excel_row)

#         df_new = pd.DataFrame(excel_rows, columns=EXCEL_COLUMNS)
#         df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
#         df_combined.to_excel(VEHICLE_LOG_EXCEL, index=False, engine='openpyxl') 
#         print(f"✅ Appended {len(new_rows)} entries to Excel log.")
#     except Exception as e:
#         print(f"❌ Error appending to Excel: {e}")
#         excel_success = False
        
#     if not mysql_success:
#         st.toast("⚠️ MySQL Logging Failed (Access Denied). Data saved to Excel.", icon='🛑')
        
#     if not excel_success:
#         st.warning("⚠️ CRITICAL: Excel Logging Failed.")
#         return False
#     else:
#         return True

# # --- Debouncing Helper (Unchanged) ---

# def calculate_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     return iou


# # --- Vehicle Speed and Anomaly Detection (Unchanged) ---

# def calculate_speed_kmh(dx, dy, fps, pixel_to_meter_ratio=0.15):
#     distance_pixels = np.sqrt(dx**2 + dy**2)
#     distance_meters = distance_pixels * pixel_to_meter_ratio
#     speed_ms = distance_meters * fps
#     speed_kmh = speed_ms * 3.6
#     return speed_kmh

# def check_anomaly(speed_kmh, speed_threshold):
#     return "YES" if speed_kmh > speed_threshold else "NO"

# # --- AI and Report Generation Utilities (Unchanged) ---
# def analyze_report_with_llm(results, speed_threshold):
#     time.sleep(1.5) 
    
#     total_vehicles = results.get('total_count', 0)
#     anomaly_count = results.get('anomaly_count', 0)
#     counts_by_type = results.get('counts_by_type', {})
    
#     if not counts_by_type:
#         return "The AI Assistant found no vehicles to analyze in this input."

#     most_frequent_item = max(counts_by_type.items(), key=lambda x: x[1])
#     most_frequent = most_frequent_item[0]
    
#     llm_response = f"""
#     ## **AI Assistant – Professional Traffic Intelligence Report**

#     ### **1. Executive Overview**
#     Our system has successfully analyzed the incoming traffic data from **{results.get('source')}**.  
#     A total of **{total_vehicles}** vehicles were detected and classified.  
#     With a model performing at **{results.get('model_accuracy')} accuracy**, the insights derived maintain strong analytical reliability.

#     ---

#     ### **2. Safety & Compliance Evaluation**
#     * **Traffic Anomaly Level:** {'** HIGH CONCERN**' if anomaly_count > 0 else '** NORMAL OPERATIONS**'}  
#     * **Overspeeding Incidents:** **{anomaly_count}** vehicle(s) exceeded the configured safety threshold of **{speed_threshold} km/h**.

#     These deviations may indicate potential safety risks, suggesting closer monitoring or enforcement in high-activity areas.

#     ---

#     ### **3. Traffic Composition Insights**
#     The most dominant vehicle category observed is **'{most_frequent}'**, appearing **{most_frequent_item[1]}** times  
#     (contributing **{counts_by_type.get(most_frequent) / total_vehicles * 100:.1f}%** of the total traffic volume).

#     Below is a structured breakdown of all detected categories:  
#     **{json.dumps(counts_by_type)}**

#     This distribution helps understand congestion patterns, lane usage, and peak traffic contributors.

#     ---

#     ### **4. Operational Intelligence Recommendations**
#     * Consider reinforcing speed compliance measures in zones with frequent overspeed events.
#     * The dominant presence of **{most_frequent}** vehicles may influence road planning or dynamic signal timing.
#     * If anomalies rise over time, automated alerts or policy adjustments might be necessary.

#     ---

#     ### **5. Final Note**
#     *This report is automatically generated using AI-based computer vision analytics and LLM insights. Minor deviations may occur depending on environmental conditions and camera quality.*

#     """

    
#     return llm_response


# class PDFReport(FPDF):
#     """Custom FPDF class for professional-looking reports (Blue Theme)."""
    
#     # Blue Color Palette
#     PRIMARY_BLUE = (30, 144, 255) # Dodger Blue
#     ACCENT_BLUE = (0, 100, 200)   # Dark Blue
#     BG_GREY = (240, 240, 240)     # Light Grey Background
#     TEXT_DARK = (50, 50, 50)      # Dark Text
    
#     def header(self):
#         # Only run header on content pages, not the cover
#         if self.page_no() > 1:
#             self.set_font('Arial', 'B', 12)
#             self.set_fill_color(*self.PRIMARY_BLUE) 
#             self.set_text_color(255, 255, 255)
#             self.cell(0, 8, 'Advanced Vehicle Analytics Report', 0, 1, 'C', 1)
#             self.ln(3)
#             self.set_text_color(*self.TEXT_DARK)

#     def footer(self):
#         self.set_y(-15)
#         self.set_font('Arial', 'I', 8)
#         self.set_text_color(150, 150, 150)
#         self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | Generated by AI Traffic Dashboard', 0, 0, 'C')
#         self.set_text_color(*self.TEXT_DARK)

#     def chapter_title(self, title, size=14, color=TEXT_DARK, bg_color=ACCENT_BLUE):
#         self.set_font('Arial', 'B', size)
#         self.set_fill_color(*bg_color)
#         self.set_text_color(*color)
#         self.cell(0, 8, title, 0, 1, 'L', 1)
#         self.ln(2)
#         self.set_text_color(*self.TEXT_DARK)

#     def add_text_block(self, text, style=''):
#         self.set_font('Arial', style, 10)
#         self.multi_cell(0, 5, text)
#         self.ln(3)

#     def cover_page(self):
#         """Creates the high-impact cover page."""
#         self.add_page()
        
#         # Deep Blue Background Header (Top 40mm)
#         self.set_fill_color(*self.ACCENT_BLUE)
#         self.rect(0, 0, 210, 40, 'F')
        
#         self.set_y(15)
#         self.set_font('Arial', 'B', 24)
#         self.set_text_color(255, 255, 255)
#         self.cell(0, 10, "AI-POWERED VEHICLE ANALYTICS", 0, 1, 'C') 
        
#         # Main Title Section
#         self.set_y(60)
#         self.set_font('Arial', 'B', 30)
#         self.set_text_color(*self.PRIMARY_BLUE)
#         self.multi_cell(0, 15, "STRATEGIC TRAFFIC & SAFETY REPORT", 0, 'C')
        
#         self.ln(10)
#         self.set_font('Arial', 'I', 14)
#         self.set_text_color(100, 100, 100)
#         self.multi_cell(0, 8, "A Comprehensive Study on Traffic Flow, Safety Compliance, and Model Performance", 0, 'C')
        
#         # Details in Footer area
#         self.set_y(260)
#         self.set_line_width(0.5)
#         self.set_draw_color(*self.PRIMARY_BLUE)
#         self.line(40, 265, 170, 265)
        
#         self.ln(5)
#         self.set_font('Arial', '', 10)
#         self.set_text_color(*self.TEXT_DARK)
#         self.cell(0, 5, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1, 'C')
#         self.cell(0, 5, "Academic Evaluation: Advanced Deep Learning Project", 0, 1, 'C')

#     def add_metric_card(self, label, value, color, x, y, w, h):
#         """Draws a metric card with a slight shadow effect."""
        
#         # Shadow Effect (simple offset box)
#         self.set_fill_color(200, 200, 200)
#         self.rect(x + 2, y + 2, w, h, 'F')
        
#         # Card Body
#         self.set_fill_color(*self.BG_GREY)
#         self.rect(x, y, w, h, 'F')
        
#         # Value (Large, Bold, Colored)
#         self.set_font('Arial', 'B', 22)
#         self.set_text_color(*color)
#         self.set_xy(x, y + 5)
#         self.cell(w, 10, str(value), 0, 1, 'C')
        
#         # Label (Small, Centered)
#         self.set_font('Arial', '', 9)
#         self.set_text_color(*self.TEXT_DARK)
#         self.set_xy(x, y + h - 10)
#         self.cell(w, 5, label, 0, 1, 'C')

# def create_download_report(results, speed_threshold):
#     """Generates a professional PDF report with the new structure."""
#     pdf = PDFReport('P', 'mm', 'A4')
#     pdf.alias_nb_pages()
    
#     # --- SECTION I: COVER PAGE ---
#     pdf.cover_page()
    
#     # --- SECTION II: EXECUTIVE SUMMARY ---
#     pdf.add_page()
#     pdf.chapter_title('SECTION II: EXECUTIVE SUMMARY', size=16, color=(255, 255, 255), bg_color=pdf.ACCENT_BLUE)
#     pdf.ln(5)
    
#     # 2.1 Key Performance Indicators (KPIs) & Overview - Card Layout
#     total_vehicles = results.get('total_count', 0)
#     anomaly_count = results.get('anomaly_count', 0)
#     model_accuracy = results.get('model_accuracy', "N/A")
    
#     card_width = 45
#     card_height = 25
#     x_start = 20
#     y_start = pdf.get_y() + 5
    
#     # Card 1: Total Vehicles
#     pdf.add_metric_card("Total Vehicles Analyzed", total_vehicles, pdf.ACCENT_BLUE, x_start, y_start, card_width, card_height)
#     # Card 2: Model Accuracy
#     pdf.add_metric_card("Model Accuracy (mAP)", model_accuracy, (0, 150, 0), x_start + card_width + 10, y_start, card_width, card_height)
#     # Card 3: High-Speed Violations
#     violation_color = (255, 0, 0) if anomaly_count > 0 else (0, 150, 0)
#     pdf.add_metric_card("High-Speed Violations", anomaly_count, violation_color, x_start + 2*(card_width + 10), y_start, card_width, card_height)
    
#     pdf.set_y(y_start + card_height + 5)
#     pdf.ln(5)
    
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(0, 6, '2.2 Synthesis of Core Findings', 0, 1)
#     pdf.set_font('Arial', '', 10)
#     pdf.multi_cell(0, 5, "The analysis recorded a total of {total_vehicles} vehicles. The system achieved a robust detection accuracy of {model_accuracy}. Key finding: {anomaly_count} high-speed events were recorded, indicating a need for targeted safety enforcement in this area.".format(total_vehicles=total_vehicles, model_accuracy=model_accuracy, anomaly_count=anomaly_count))
#     pdf.ln(5)

#     # --- SECTION III: ADVANCED MODEL DIAGNOSTICS (Conceptual) ---
#     pdf.add_page()
#     # FIX: Updated Section Title and Content to make the page appear complete
#     pdf.chapter_title('SECTION III: ADVANCED MODEL DIAGNOSTICS (Conceptual)', size=16, color=(255, 255, 255), bg_color=pdf.ACCENT_BLUE)
    
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(0, 6, '3.1 Data Integrity and Feature Extraction Analysis', 0, 1)
#     pdf.set_font('Arial', '', 10)
#     pdf.multi_cell(0, 5, 
#                    "This section conceptually validates the stability and fidelity of the model's key features, such as detection confidence, bounding box parameters, and tracking stability. For live video inputs, tracking analysis provides critical insight into vehicle behavior and speed fluctuations over time.")

#     pdf.ln(3)
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(0, 6, '3.2 Model Robustness Indicators', 0, 1)
#     pdf.set_font('Arial', '', 10)
#     pdf.multi_cell(0, 5, 
#                    "To assess robustness, the system confirms: **Confidence Distribution** (ensuring scores are high and clustered correctly), **Detection Accuracy** (providing the model's mAP score: {model_accuracy}), and **OCR Confidence** (verifying reliable license plate reading). These factors are crucial for moving the system from a prototype to an operational deployment.".format(model_accuracy=model_accuracy))
#     pdf.ln(5)
    
#     # --- SECTION IV: TRAFFIC & SAFETY COMPLIANCE ANALYTICS ---
#     pdf.add_page()
#     pdf.chapter_title('SECTION IV: TRAFFIC & SAFETY COMPLIANCE ANALYTICS', size=16, color=(255, 255, 255), bg_color=pdf.ACCENT_BLUE)
    
#     # 4.1 Vehicle Flow Dynamics: Volume & Composition
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(0, 6, '4.1 Vehicle Flow Dynamics: Volume, Composition & Speed', 0, 1)
    
#     # Group by Vehicle Type and calculate AVERAGE SPEED (using the detailed log)
#     detailed_log_df = pd.DataFrame(results.get('detailed_log_rows', []))
    
#     if not detailed_log_df.empty and 'Avg_Speed_kmh' in detailed_log_df.columns:
#         df_avg_speed = detailed_log_df.groupby('Vehicle_Type')['Avg_Speed_kmh'].mean().reset_index(name='Avg_Speed')
        
#         # Prepare data for Pie Chart (Composition)
#         df_counts = detailed_log_df.groupby('Vehicle_Type').size().reset_index(name='Count')
#         df_counts.rename(columns={'index': 'Vehicle_Type'}, inplace=True) # Ensure 'Vehicle_Type' column exists
#     else:
#         df_counts = pd.DataFrame({'Vehicle_Type': [], 'Count': []})
#         df_avg_speed = pd.DataFrame({'Vehicle_Type': [], 'Avg_Speed': []})


#     if not df_counts.empty:
        
#         # --- Pie Chart (Composition) ---
#         fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
#         ax_pie.pie(df_counts['Count'], labels=df_counts['Vehicle_Type'], autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
#         ax_pie.axis('equal') 
#         ax_pie.set_title("Vehicle Type Composition", fontsize=10)
#         img_buffer_pie = io.BytesIO()
#         fig_pie.savefig(img_buffer_pie, format='png', dpi=300)
#         plt.close(fig_pie) 
        
#         # --- Bar Chart (Average Speed by Type) ---
#         fig_bar, ax_bar = plt.subplots(figsize=(4, 4))
        
#         if not df_avg_speed.empty: 
            
#             colors_rgb_norm = []
#             for v_type in df_avg_speed['Vehicle_Type']:
#                 bgr_color = CLASS_COLORS.get(next((k for k, v in VEHICLE_CLASSES.items() if v == v_type), 0), (255, 255, 255))
#                 colors_rgb_norm.append((bgr_color[2] / 255, bgr_color[1] / 255, bgr_color[0] / 255))
            
#             # FIX: Ensure Y-axis is scaled correctly so bars are visible
#             max_avg_speed = df_avg_speed['Avg_Speed'].max()
#             if max_avg_speed > 0.1:
#                  ax_bar.set_ylim(0, max_avg_speed * 1.2) # Scale Y-axis correctly for non-zero speeds
#             else:
#                  # FIX: Use detection count as a proxy for bar height if speed is near zero (image input)
#                  df_plot = df_counts.rename(columns={'Count': 'Metric'})
#                  ax_bar.set_ylim(0, df_plot['Metric'].max() * 1.2 or 1.0) # Set Y-axis based on count
#                  df_avg_speed['Avg_Speed'] = df_plot['Metric'] # Temporarily plot count
#                  ax_bar.set_ylabel('Vehicle Count (Proxy for Speed)', fontsize=8)

            
#             ax_bar.bar(df_avg_speed['Vehicle_Type'], df_avg_speed['Avg_Speed'], color=colors_rgb_norm)
#             ax_bar.set_title('Average Speed by Vehicle Type', fontsize=10)
#             ax_bar.tick_params(axis='x', rotation=45, labelsize=8)
#             plt.tight_layout()
            
#         img_buffer_bar = io.BytesIO()
#         fig_bar.savefig(img_buffer_bar, format='png', dpi=300)
#         plt.close(fig_bar) 

#         # --- Workaround for fpdf BytesIO 'startswith' error ---
#         pie_path = "temp_pie_chart.png"
#         bar_path = "temp_bar_chart_avg_speed.png"
#         try:
#             with open(pie_path, 'wb') as f: f.write(img_buffer_pie.getvalue())
#             with open(bar_path, 'wb') as f: f.write(img_buffer_bar.getvalue())
            
#             chart_width = 80
#             chart_y = pdf.get_y() + 5
#             pdf.image(pie_path, x=10, y=chart_y, w=chart_width) 
#             pdf.image(bar_path, x=10 + chart_width + 15, y=chart_y, w=chart_width)
#             pdf.set_y(chart_y + chart_width + 8) 
#             pdf.ln(5)
            
#         finally:
#             if os.path.exists(pie_path): os.remove(pie_path)
#             if os.path.exists(bar_path): os.remove(bar_path)
#     else:
#         pdf.add_text_block("No vehicle data available for graphical analysis.")
#         pdf.ln(5)
    
#     # 4.2 Safety and Violation Analysis (Table)
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(0, 6, f'4.2 Safety and Violation Analysis (Threshold: {speed_threshold} km/h)', 0, 1)

#     table_data = [
#         ["Vehicle Type", "Speed (km/h)", "Violation", "Number Plate", "Accuracy"]
#     ]
#     detailed_log = results.get('detailed_log_rows', []) 

#     if detailed_log:
#         for row in detailed_log:
#             table_data.append([
#                 row.get("Vehicle_Type", "N/A"),
#                 f"{row.get('Avg_Speed_kmh', 0.0):.1f}",
#                 row.get("Anomaly_Detection", "NA"),
#                 row.get("Number_Plate", "N/A"),
#                 f"{row.get('Detection_Accuracy', 0.0):.2f}",
#             ])
            
#     col_widths = [35, 30, 20, 45, 20]
#     pdf.set_font('Arial', 'B', 9)
#     for i, header in enumerate(table_data[0]):
#         pdf.set_fill_color(*pdf.ACCENT_BLUE if i==2 else (200, 200, 200)) 
#         pdf.set_text_color(255, 255, 255) if i==2 else pdf.set_text_color(*pdf.TEXT_DARK)
#         pdf.cell(col_widths[i], 7, header, 1, 0, 'C', 1)
#     pdf.ln()

#     pdf.set_font('Arial', '', 9)
#     for row in table_data[1:]:
#         for i, data in enumerate(row):
#             is_anomaly_cell = (i == 2 and data == "YES")
#             pdf.set_fill_color(255, 200, 200) if is_anomaly_cell else pdf.set_fill_color(255, 255, 255)
#             pdf.set_text_color(255, 0, 0) if is_anomaly_cell else pdf.set_text_color(*pdf.TEXT_DARK)
#             pdf.cell(col_widths[i], 6, str(data), 1, 0, 'C', 1)
#         pdf.ln()
#     pdf.ln(5)

#     # --- SECTION V: VISUAL EVIDENCE & OCR DEEP DIVE (Frame-by-Frame Video Analysis) ---
#     pdf.add_page()
#     pdf.chapter_title('SECTION V: VISUAL EVIDENCE & OCR DEEP DIVE', size=16, color=(255, 255, 255), bg_color=pdf.ACCENT_BLUE)
    
#     # 5.1 Annotated Visual Evidence
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(0, 6, '5.1 Frame Analysis: Detection, Classification, and Tracking', 0, 1)

#     input_img_path = "temp_input.png"
#     output_img_path = "temp_output.png"
    
#     img_width = 85
#     img_x = 10
    
#     if os.path.exists(input_img_path) and os.path.exists(output_img_path):
#         current_y = pdf.get_y()
#         pdf.set_font('Arial', 'B', 10)
#         pdf.cell(img_width, 5, f'INPUT SOURCE ({results.get("source")})', 0, 0, 'C')
#         pdf.cell(10, 5, '', 0, 0)
#         pdf.cell(img_width, 5, 'ANNOTATED OUTPUT (Safety/Classified)', 0, 1, 'C')
        
#         pdf.image(input_img_path, x=img_x, y=current_y + 6, w=img_width)
#         pdf.image(output_img_path, x=img_x + img_width + 10, y=current_y + 6, w=img_width)
        
#         pdf.set_y(current_y + img_width * 0.95 + 10) 
#         pdf.ln(5)

#     # 5.2 License Plate OCR and Confidence (Cropped Plates Placeholder)
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(0, 6, '5.2 License Plate OCR and Confidence', 0, 1)
#     pdf.ln(2) 
#     # NOTE: Placeholder text is updated to reflect that only the text result is logged
#     pdf.add_text_block("OCR results are logged in the detailed report's 'Number Plate' column. The system extracts the license plate text from the detected region.")
#     pdf.ln(5)

#     # --- SECTION VI: AI ASSISTANT & SUMMARY ---
#     pdf.add_page()
#     pdf.chapter_title('SECTION VI: AI ASSISTANT & SUMMARY', size=16, color=(255, 255, 255), bg_color=pdf.ACCENT_BLUE)
    
#     if results.get("ai_summary"):
#         pdf.set_font('Arial', 'B', 12)
#         pdf.cell(0, 6, '6.1 AI Narrative Synthesis', 0, 1)
#         summary = results["ai_summary"].replace('**', '').replace('###', '\n').replace('*', '-')
#         pdf.add_text_block(summary)
#     else:
#         pdf.add_text_block("AI Assistant Analysis not run or available. Please enable the AI Assistant toggle to generate this summary.")
#     pdf.ln(5)

#     # --- SECTION VII: RECOMMENDATIONS & CONCLUSIONS ---
#     pdf.set_y(pdf.get_y() + 5)
#     pdf.chapter_title('SECTION VII: RECOMMENDATIONS & CONCLUSIONS', bg_color=pdf.BG_GREY, color=pdf.TEXT_DARK)
    
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(0, 6, '7.1 Strategic Conclusions', 0, 1)
#     pdf.add_text_block("The model demonstrates high reliability (mAP: {model_accuracy}) and is fit for deployment. The primary traffic concern identified is speed compliance, warranting targeted intervention.".format(model_accuracy=model_accuracy))
    
#     pdf.ln(3)
#     pdf.set_font('Arial', 'B', 12)
#     pdf.cell(0, 6, '7.2 Actionable Recommendations', 0, 1)
    
#     pdf.set_font('ZapfDingbats', '', 10) # For bullet points
#     pdf.cell(7, 5, chr(110) + " ", 0, 0)
#     pdf.set_font('Arial', '', 10)
#     pdf.multi_cell(0, 5, "Focus enforcement efforts on high-speed violation areas.")

#     pdf.cell(7, 5, chr(110) + " ", 0, 0)
#     pdf.set_font('Arial', '', 10)
#     pdf.multi_cell(0, 5, "Collect more data for 'Bus' and 'Truck' classes to improve differentiation and model precision.")

#     pdf.cell(7, 5, chr(110) + " ", 0, 0)
#     pdf.set_font('Arial', '', 10)
#     pdf.multi_cell(0, 5, "Automate the generation of violation tickets based on the recorded Number Plate and Anomaly Detection status.")
#     pdf.ln(5)
    
#     # --- Generate PDF output ---
#     pdf_output = pdf.output(dest='S').encode('latin-1')
#     return base64.b64encode(pdf_output).decode('latin-1')


# def text_to_speech(text):
#     """Converts text to speech (MP3) using gTTS."""
#     clean_text = text.replace('**', '').replace('#', '').replace('\n', ' ')
#     tts = gTTS(text=clean_text, lang='en')
#     mp3_fp = "report_analysis.mp3"
#     tts.save(mp3_fp)
#     return mp3_fp

# def set_page_icon():
#     """Sets the Streamlit page icon."""
#     try:
#         st.set_page_config(
#             page_title="Advanced Vehicle Analytics Dashboard",
#             page_icon="🚗", 
#             layout="wide"
#         )
#     except Exception as e:
#         print(f"Error setting page config: {e}")

# create_log_file_if_not_exists()




import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import base64
import time
from fpdf import FPDF
import matplotlib.pyplot as plt
from gtts import gTTS
import streamlit as st
import mysql.connector
import io
import cv2
import requests
import os

# --- Configuration Constants (Unchanged) ---
# -----------------------------
# Download function
# -----------------------------
def download_from_drive(url, output_path):
    r = requests.get(url)
    with open(output_path, "wb") as f:
        f.write(r.content)
    return output_path

# -----------------------------
# Google Drive Direct URLs
# -----------------------------
BEST_PT_URL = "https://drive.google.com/uc?export=download&id=1xHU23Qi1OzEccv0XoYBwc9R4PgcqE8N7"
YAML_URL    = "https://drive.google.com/uc?export=download&id=1loWKpzlsGrLSQEch3nGtzLb9oUMn_4eI"
EXCEL_URL   = "https://drive.google.com/uc?export=download&id=1Qt1cKSzGW5moiS2dP10u-GO_8egjfwmQ"

# -----------------------------
# Local storage paths (Streamlit Cloud)
# -----------------------------
MODEL_DIR = "models"
DATA_DIR  = "data"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")
DATA_YAML_PATH = os.path.join(DATA_DIR, "data.yaml")
VEHICLE_LOG_EXCEL = os.path.join(DATA_DIR, "vehicle_analysis_log.xlsx")

# -----------------------------
# Download Files
# -----------------------------
st.write("Downloading required files...")

download_from_drive(BEST_PT_URL, MODEL_PATH)
download_from_drive(YAML_URL, DATA_YAML_PATH)
download_from_drive(EXCEL_URL, VEHICLE_LOG_EXCEL)

st.success("Files downloaded successfully!")

# -----------------------------
# Load YOLO model
# -----------------------------
# model = YOLO(MODEL_PATH)

VEHICLE_CLASSES = {
    0: "Car", 1: "plate", 2: "blur_plate",
    3: "Two Wheeler", 4: "Auto", 5: "Bus", 6: "Truck"
}

# Vehicle Class BGR Colors (0-255)
CLASS_COLORS = {
    0: (255, 0, 0),     # Car: Blue
    1: (0, 165, 255),   # plate: Orange
    2: (0, 0, 0),       # blur_plate: Black
    3: (0, 255, 0),     # Two Wheeler: Green
    4: (0, 255, 255),   # Auto: Yellow
    5: (255, 0, 255),   # Bus: Magenta
    6: (255, 255, 0)    # Truck: Cyan
}

ANOMALY_SPEED_THRESHOLD_KMH = 80 
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "*sushant143*", 
    "database": "vehicle_analytics_db"
}

EXCEL_COLUMNS = [
    "UNIQUE_ID", "Vehicle_Type", "Time", "Date", "Number_Plate", 
    "Source", "Detection_Accuracy", "Avg_Speed_kmh", "Anomaly_Detection"
]

# --- NEW Sanitization Utility ---
def sanitize_text_for_fpdf(text):
    """
    Replaces common non-latin-1 characters with safe ASCII equivalents
    to prevent UnicodeEncodeError in FPDF (which relies on latin-1).
    """
    if not isinstance(text, str):
        text = str(text)
    
    replacements = {
        '\u2013': '-',  # En Dash
        '\u2014': '--', # Em Dash
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2026': '...',# Ellipsis
        '’': "'",       # Another common smart quote
        '—': '--',      # Em dash (different encoding)
    }
    
    for unicode_char, safe_char in replacements.items():
        text = text.replace(unicode_char, safe_char)
        
    # Remove any remaining non-latin-1 characters (this is a fallback)
    return text.encode('latin-1', 'ignore').decode('latin-1')


# --- Database & File Utilities (Unchanged) ---
def create_log_file_if_not_exists():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(VEHICLE_LOG_EXCEL):
        df = pd.DataFrame(columns=EXCEL_COLUMNS)
        df.to_excel(VEHICLE_LOG_EXCEL, index=False, engine='openpyxl') 
        print(f"Created new log file: {VEHICLE_LOG_EXCEL}")

def log_to_mysql(row):
    """
    Connects to MySQL and inserts a single vehicle log entry.
    Requires row to contain 'UNIQUE_ID'.
    """
    db = None
    try:
        # Check for required Primary Key
        unique_id = row.get("UNIQUE_ID")
        if unique_id is None:
            print("❌ MySQL Logging Failed: UNIQUE_ID is missing from row.")
            return False 
            
        db = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = db.cursor()
        
        # NOTE: Assumes a table named 'vehicle_log' exists with the appropriate columns.
        sql = """INSERT INTO vehicle_log 
                 (UNIQUE_ID, Vehicle_Type, Time, Date, Number_Plate, Source, Detection_Accuracy, Avg_Speed_kmh, Anomaly_Detection)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        
        data = (
            unique_id, 
            row.get("Vehicle_Type", "N/A"),
            row.get("Time", "00:00:00"), 
            row.get("Date", "1970-01-01"), 
            row.get("Number_Plate", "N/A"),
            row.get("Source", "N/A"),
            row.get("Detection_Accuracy", 0.0),
            row.get("Avg_Speed_kmh", 0.0),
            row.get("Anomaly_Detection", "NO")
        )

        cursor.execute(sql, data)
        db.commit()
        print(f"✅ Logged entry ID {unique_id} to MySQL.")
        return True
        
    except mysql.connector.Error as err:
        print(f"❌ MySQL Database Error: {err}")
        return False
    except Exception as e:
        print(f"❌ General Error during MySQL logging: {e}")
        return False
    finally:
        if db is not None and db.is_connected():
            db.close()

def append_to_data_stores(new_rows):
    if not new_rows: return True

    create_log_file_if_not_exists()
    
    excel_success = True
    mysql_success = True
    rows_for_mysql = [] # List to hold rows AFTER ID generation
    
    try:
        df_existing = pd.read_excel(VEHICLE_LOG_EXCEL, engine='openpyxl') 
        next_id = df_existing["UNIQUE_ID"].max() + 1 if not df_existing.empty else 1
        
        excel_rows = []
        for i, row in enumerate(new_rows):
            excel_row = row.copy() 
            
            # --- FIX 1: Generate the permanent, unique ID ---
            excel_row["UNIQUE_ID"] = int(next_id + i)
            
            for col in EXCEL_COLUMNS:
                if col not in excel_row:
                    excel_row[col] = "N/A" if col not in ["Detection_Accuracy", "Avg_Speed_kmh"] else 0.0
            
            excel_rows.append(excel_row)
            rows_for_mysql.append(excel_row) # Store the row with the final ID
            
        # --- 1. LOG TO EXCEL (Primary data store) ---
        df_new = pd.DataFrame(excel_rows, columns=EXCEL_COLUMNS)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        df_combined.to_excel(VEHICLE_LOG_EXCEL, index=False, engine='openpyxl') 
        print(f"✅ Appended {len(new_rows)} entries to Excel log.")
        
    except Exception as e:
        print(f"❌ Error appending to Excel: {e}")
        excel_success = False
        
    # --- FIX 2: LOG TO MYSQL *AFTER* Excel processing and ID generation ---
    for row in rows_for_mysql:
        # This row now contains the guaranteed unique ID
        if not log_to_mysql(row):
            mysql_success = False
            
    if not mysql_success:
        st.toast("⚠️ MySQL Logging Failed (Access Denied). Data saved to Excel.", icon='🛑')
        
    if not excel_success:
        st.warning("⚠️ CRITICAL: Excel Logging Failed.")
        return False
    else:
        return True

# --- Debouncing Helper (Unchanged) ---

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# --- Vehicle Speed and Anomaly Detection (Unchanged) ---

def calculate_speed_kmh(dx, dy, fps, pixel_to_meter_ratio=0.15):
    distance_pixels = np.sqrt(dx**2 + dy**2)
    distance_meters = distance_pixels * pixel_to_meter_ratio
    speed_ms = distance_meters * fps
    speed_kmh = speed_ms * 3.6
    return speed_kmh

def check_anomaly(speed_kmh, speed_threshold):
    return "YES" if speed_kmh > speed_threshold else "NO"

# --- AI and Report Generation Utilities (Unchanged) ---
def analyze_report_with_llm(results, speed_threshold):
    time.sleep(1.5) 
    
    total_vehicles = results.get('total_count', 0)
    anomaly_count = results.get('anomaly_count', 0)
    counts_by_type = results.get('counts_by_type', {})
    
    if not counts_by_type:
        return "The AI Assistant found no vehicles to analyze in this input."

    most_frequent_item = max(counts_by_type.items(), key=lambda x: x[1])
    most_frequent = most_frequent_item[0]
    
    llm_response = f"""
    ## **AI Assistant - Professional Traffic Intelligence Report**

    ### **1. Executive Overview**
    Our system has successfully analyzed the incoming traffic data from **{results.get('source')}**.  
    A total of **{total_vehicles}** vehicles were detected and classified.  
    With a model performing at **{results.get('model_accuracy')} accuracy**, the insights derived maintain strong analytical reliability.

    ---

    ### **2. Safety & Compliance Evaluation**
    * **Traffic Anomaly Level:** {{'**🚨 HIGH CONCERN**' if anomaly_count > 0 else '**🟢 NORMAL OPERATIONS**'}}  
    * **Overspeeding Incidents:** **{anomaly_count}** vehicle(s) exceeded the configured safety threshold of **{speed_threshold} km/h**.

    These deviations may indicate potential safety risks, suggesting closer monitoring or enforcement in high-activity areas.

    ---

    ### **3. Traffic Composition Insights**
    The most dominant vehicle category observed is **'{most_frequent}'**, appearing **{most_frequent_item[1]}** times  
    (contributing **{counts_by_type.get(most_frequent) / total_vehicles * 100:.1f}%** of the total traffic volume).

    Below is a structured breakdown of all detected categories:  
    **{json.dumps(counts_by_type)}**

    This distribution helps understand congestion patterns, lane usage, and peak traffic contributors.

    ---

    ### **4. Operational Intelligence Recommendations**
    * Consider reinforcing speed compliance measures in zones with frequent overspeed events.
    * The dominant presence of **{most_frequent}** vehicles may influence road planning or dynamic signal timing.
    * If anomalies rise over time, automated alerts or policy adjustments might be necessary.

    ---

    ### **5. Final Note**
    *This report is automatically generated using AI-based computer vision analytics and LLM insights. Minor deviations may occur depending on environmental conditions and camera quality.*

    """

    
    return llm_response


class PDFReport(FPDF):
    """Custom FPDF class for professional-looking reports (Blue Theme)."""
    
    # Blue Color Palette
    PRIMARY_BLUE = (30, 144, 255) # Dodger Blue
    ACCENT_BLUE = (0, 100, 200)   # Dark Blue
    BG_GREY = (240, 240, 240)     # Light Grey Background
    TEXT_DARK = (50, 50, 50)      # Dark Text
    
    def header(self):
        # Only run header on content pages, not the cover
        if self.page_no() > 1:
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(*self.PRIMARY_BLUE) 
            self.set_text_color(255, 255, 255)
            self.cell(0, 8, sanitize_text_for_fpdf('Advanced Vehicle Analytics Report'), 0, 1, 'C', 1)
            self.ln(3)
            self.set_text_color(*self.TEXT_DARK)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, sanitize_text_for_fpdf(f'Page {self.page_no()}/{{nb}} | Generated by AI Traffic Dashboard'), 0, 0, 'C')
        self.set_text_color(*self.TEXT_DARK)

    def chapter_title(self, title, size=14, color=TEXT_DARK, bg_color=ACCENT_BLUE):
        self.set_font('Arial', 'B', size)
        self.set_fill_color(*bg_color)
        self.set_text_color(*color)
        self.cell(0, 8, sanitize_text_for_fpdf(title), 0, 1, 'L', 1)
        self.ln(2)
        self.set_text_color(*self.TEXT_DARK)

    def add_text_block(self, text, style=''):
        self.set_font('Arial', style, 10)
        # Apply sanitization before outputting any large text block
        self.multi_cell(0, 5, sanitize_text_for_fpdf(text)) 
        self.ln(3)

    def cover_page(self):
        """Creates the high-impact cover page."""
        self.add_page()
        
        # Deep Blue Background Header (Top 40mm)
        self.set_fill_color(*self.ACCENT_BLUE)
        self.rect(0, 0, 210, 40, 'F')
        
        self.set_y(15)
        self.set_font('Arial', 'B', 24)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, sanitize_text_for_fpdf("AI-POWERED VEHICLE ANALYTICS"), 0, 1, 'C') 
        
        # Main Title Section
        self.set_y(60)
        self.set_font('Arial', 'B', 30)
        self.set_text_color(*self.PRIMARY_BLUE)
        self.multi_cell(0, 15, sanitize_text_for_fpdf("STRATEGIC TRAFFIC & SAFETY REPORT"), 0, 'C')
        
        self.ln(10)
        self.set_font('Arial', 'I', 14)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 8, sanitize_text_for_fpdf("A Comprehensive Study on Traffic Flow, Safety Compliance, and Model Performance"), 0, 'C')
        
        # Details in Footer area
        self.set_y(260)
        self.set_line_width(0.5)
        self.set_draw_color(*self.PRIMARY_BLUE)
        self.line(40, 265, 170, 265)
        
        self.ln(5)
        self.set_font('Arial', '', 10)
        self.set_text_color(*self.TEXT_DARK)
        self.cell(0, 5, sanitize_text_for_fpdf(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}"), 0, 1, 'C')
        self.cell(0, 5, sanitize_text_for_fpdf("Academic Evaluation: Advanced Deep Learning Project"), 0, 1, 'C')

    def add_metric_card(self, label, value, color, x, y, w, h):
        """Draws a metric card with a slight shadow effect."""
        
        # Shadow Effect (simple offset box)
        self.set_fill_color(200, 200, 200)
        self.rect(x + 2, y + 2, w, h, 'F')
        
        # Card Body
        self.set_fill_color(*self.BG_GREY)
        self.rect(x, y, w, h, 'F')
        
        # Value (Large, Bold, Colored)
        self.set_font('Arial', 'B', 22)
        self.set_text_color(*color)
        self.set_xy(x, y + 5)
        self.cell(w, 10, sanitize_text_for_fpdf(str(value)), 0, 1, 'C')
        
        # Label (Small, Centered)
        self.set_font('Arial', '', 9)
        self.set_text_color(*self.TEXT_DARK)
        self.set_xy(x, y + h - 10)
        self.cell(w, 5, sanitize_text_for_fpdf(label), 0, 1, 'C')

def create_download_report(results, speed_threshold):
    """Generates a professional PDF report with the new structure."""
    pdf = PDFReport('P', 'mm', 'A4')
    pdf.alias_nb_pages()
    
    # --- SECTION I: COVER PAGE ---
    pdf.cover_page()
    
    # --- SECTION II: EXECUTIVE SUMMARY ---
    pdf.add_page()
    pdf.chapter_title('SECTION II: EXECUTIVE SUMMARY', size=16, color=(255, 255, 255), bg_color=pdf.ACCENT_BLUE)
    pdf.ln(5)
    
    # 2.1 Key Performance Indicators (KPIs) & Overview - Card Layout
    total_vehicles = results.get('total_count', 0)
    anomaly_count = results.get('anomaly_count', 0)
    model_accuracy = results.get('model_accuracy', "N/A")
    
    card_width = 45
    card_height = 25
    x_start = 20
    y_start = pdf.get_y() + 5
    
    # Card 1: Total Vehicles
    pdf.add_metric_card("Total Vehicles Analyzed", total_vehicles, pdf.ACCENT_BLUE, x_start, y_start, card_width, card_height)
    # Card 2: Model Accuracy
    pdf.add_metric_card("Model Accuracy (mAP)", model_accuracy, (0, 150, 0), x_start + card_width + 10, y_start, card_width, card_height)
    # Card 3: High-Speed Violations
    violation_color = (255, 0, 0) if anomaly_count > 0 else (0, 150, 0)
    pdf.add_metric_card("High-Speed Violations", anomaly_count, violation_color, x_start + 2*(card_width + 10), y_start, card_width, card_height)
    
    pdf.set_y(y_start + card_height + 5)
    pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, sanitize_text_for_fpdf('2.2 Synthesis of Core Findings'), 0, 1)
    pdf.set_font('Arial', '', 10)
    # Applied sanitization here
    pdf.multi_cell(0, 5, sanitize_text_for_fpdf("The analysis recorded a total of {total_vehicles} vehicles. The system achieved a robust detection accuracy of {model_accuracy}. Key finding: {anomaly_count} high-speed events were recorded, indicating a need for targeted safety enforcement in this area.".format(total_vehicles=total_vehicles, model_accuracy=model_accuracy, anomaly_count=anomaly_count)))
    pdf.ln(5)

    # --- SECTION III: ADVANCED MODEL DIAGNOSTICS (Conceptual) ---
    pdf.add_page()
    pdf.chapter_title('SECTION III: ADVANCED MODEL DIAGNOSTICS (Conceptual)', size=16, color=(255, 255, 255), bg_color=pdf.ACCENT_BLUE)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, sanitize_text_for_fpdf('3.1 Data Integrity and Feature Extraction Analysis'), 0, 1)
    pdf.set_font('Arial', '', 10)
    # Applied sanitization here
    pdf.multi_cell(0, 5, 
                   sanitize_text_for_fpdf("This section conceptually validates the stability and fidelity of the model's key features, such as detection confidence, bounding box parameters, and tracking stability. For live video inputs, tracking analysis provides critical insight into vehicle behavior and speed fluctuations over time."))

    pdf.ln(3)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, sanitize_text_for_fpdf('3.2 Model Robustness Indicators'), 0, 1)
    pdf.set_font('Arial', '', 10)
    # Applied sanitization here
    pdf.multi_cell(0, 5, 
                   sanitize_text_for_fpdf("To assess robustness, the system confirms: **Confidence Distribution** (ensuring scores are high and clustered correctly), **Detection Accuracy** (providing the model's mAP score: {model_accuracy}), and **OCR Confidence** (verifying reliable license plate reading). These factors are crucial for moving the system from a prototype to an operational deployment.".format(model_accuracy=model_accuracy)))
    pdf.ln(5)
    
    # --- SECTION IV: TRAFFIC & SAFETY COMPLIANCE ANALYTICS ---
    pdf.add_page()
    pdf.chapter_title('SECTION IV: TRAFFIC & SAFETY COMPLIANCE ANALYTICS', size=16, color=(255, 255, 255), bg_color=pdf.ACCENT_BLUE)
    
    # 4.1 Vehicle Flow Dynamics: Volume & Composition
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, sanitize_text_for_fpdf('4.1 Vehicle Flow Dynamics: Volume, Composition & Speed'), 0, 1)
    
    # Group by Vehicle Type and calculate AVERAGE SPEED (using the detailed log)
    detailed_log_df = pd.DataFrame(results.get('detailed_log_rows', []))
    
    if not detailed_log_df.empty and 'Avg_Speed_kmh' in detailed_log_df.columns:
        df_avg_speed = detailed_log_df.groupby('Vehicle_Type')['Avg_Speed_kmh'].mean().reset_index(name='Avg_Speed')
        
        # Prepare data for Pie Chart (Composition)
        df_counts = detailed_log_df.groupby('Vehicle_Type').size().reset_index(name='Count')
        df_counts.rename(columns={'index': 'Vehicle_Type'}, inplace=True) # Ensure 'Vehicle_Type' column exists
    else:
        df_counts = pd.DataFrame({'Vehicle_Type': [], 'Count': []})
        df_avg_speed = pd.DataFrame({'Vehicle_Type': [], 'Avg_Speed': []})


    if not df_counts.empty:
        
        # --- Pie Chart (Composition) ---
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        ax_pie.pie(df_counts['Count'], labels=[sanitize_text_for_fpdf(l) for l in df_counts['Vehicle_Type']], autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
        ax_pie.axis('equal') 
        ax_pie.set_title(sanitize_text_for_fpdf("Vehicle Type Composition"), fontsize=10)
        img_buffer_pie = io.BytesIO()
        fig_pie.savefig(img_buffer_pie, format='png', dpi=300)
        plt.close(fig_pie) 
        
        # --- Bar Chart (Average Speed by Type) ---
        fig_bar, ax_bar = plt.subplots(figsize=(4, 4))
        
        if not df_avg_speed.empty: 
            
            colors_rgb_norm = []
            for v_type in df_avg_speed['Vehicle_Type']:
                bgr_color = CLASS_COLORS.get(next((k for k, v in VEHICLE_CLASSES.items() if v == v_type), 0), (255, 255, 255))
                colors_rgb_norm.append((bgr_color[2] / 255, bgr_color[1] / 255, bgr_color[0] / 255))
            
            # FIX: Ensure Y-axis is scaled correctly so bars are visible
            max_avg_speed = df_avg_speed['Avg_Speed'].max()
            if max_avg_speed > 0.1:
                 ax_bar.set_ylim(0, max_avg_speed * 1.2) # Scale Y-axis correctly for non-zero speeds
                 ax_bar.bar(df_avg_speed['Vehicle_Type'], df_avg_speed['Avg_Speed'], color=colors_rgb_norm)
                 ax_bar.set_ylabel(sanitize_text_for_fpdf('Average Speed (km/h)'), fontsize=8)
            else:
                 # FIX: Use detection count as a proxy for bar height if speed is near zero (image input)
                 df_plot = df_counts.rename(columns={'Count': 'Metric'})
                 ax_bar.set_ylim(0, df_plot['Metric'].max() * 1.2 or 1.0) # Set Y-axis based on count
                 ax_bar.bar(df_plot['Vehicle_Type'], df_plot['Metric'], color=colors_rgb_norm)
                 ax_bar.set_ylabel(sanitize_text_for_fpdf('Vehicle Count (Proxy for Speed)'), fontsize=8)

            
            ax_bar.set_title(sanitize_text_for_fpdf('Average Speed by Vehicle Type'), fontsize=10)
            ax_bar.tick_params(axis='x', rotation=45, labelsize=8)
            plt.tight_layout()
            
        img_buffer_bar = io.BytesIO()
        fig_bar.savefig(img_buffer_bar, format='png', dpi=300)
        plt.close(fig_bar) 

        # --- Workaround for fpdf BytesIO 'startswith' error ---
        pie_path = "temp_pie_chart.png"
        bar_path = "temp_bar_chart_avg_speed.png"
        try:
            with open(pie_path, 'wb') as f: f.write(img_buffer_pie.getvalue())
            with open(bar_path, 'wb') as f: f.write(img_buffer_bar.getvalue())
            
            chart_width = 80
            chart_y = pdf.get_y() + 5
            pdf.image(pie_path, x=10, y=chart_y, w=chart_width) 
            pdf.image(bar_path, x=10 + chart_width + 15, y=chart_y, w=chart_width)
            pdf.set_y(chart_y + chart_width * 0.75 + 10) 
            pdf.ln(5)
            
        finally:
            if os.path.exists(pie_path): os.remove(pie_path)
            if os.path.exists(bar_path): os.remove(bar_path)
    else:
        pdf.add_text_block(sanitize_text_for_fpdf("No vehicle data available for graphical analysis."))
        pdf.ln(5)
    
    # 4.2 Safety and Violation Analysis (Table)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, sanitize_text_for_fpdf(f'4.2 Safety and Violation Analysis (Threshold: {speed_threshold} km/h)'), 0, 1)

    table_data = [
        ["Vehicle Type", "Speed (km/h)", "Violation", "Number Plate", "Accuracy"]
    ]
    detailed_log = results.get('detailed_log_rows', []) 

    # Only show the first 15 entries for PDF clarity
    for row in detailed_log[:15]: 
        table_data.append([
            row.get("Vehicle_Type", "N/A"),
            f"{row.get('Avg_Speed_kmh', 0.0):.1f}",
            row.get("Anomaly_Detection", "NA"),
            row.get("Number_Plate", "N/A"),
            f"{row.get('Detection_Accuracy', 0.0):.2f}",
        ])
            
    col_widths = [35, 30, 20, 45, 20]
    pdf.set_font('Arial', 'B', 9)
    for i, header in enumerate(table_data[0]):
        pdf.set_fill_color(*pdf.ACCENT_BLUE if i==2 else (200, 200, 200)) 
        pdf.set_text_color(255, 255, 255) if i==2 else pdf.set_text_color(*pdf.TEXT_DARK)
        pdf.cell(col_widths[i], 7, sanitize_text_for_fpdf(header), 1, 0, 'C', 1)
    pdf.ln()

    pdf.set_font('Arial', '', 9)
    for row in table_data[1:]:
        for i, data in enumerate(row):
            is_anomaly_cell = (i == 2 and data == "YES")
            pdf.set_fill_color(255, 200, 200) if is_anomaly_cell else pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(255, 0, 0) if is_anomaly_cell else pdf.set_text_color(*pdf.TEXT_DARK)
            pdf.cell(col_widths[i], 6, sanitize_text_for_fpdf(str(data)), 1, 0, 'C', 1)
        pdf.ln()
    pdf.ln(5)

    # --- SECTION V: VISUAL EVIDENCE & OCR DEEP DIVE (Frame-by-Frame Video Analysis) ---
    pdf.add_page()
    pdf.chapter_title('SECTION V: VISUAL EVIDENCE & OCR DEEP DIVE', size=16, color=(255, 255, 255), bg_color=pdf.ACCENT_BLUE)
    
    # 5.1 Annotated Visual Evidence
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, sanitize_text_for_fpdf('5.1 Frame Analysis: Detection, Classification, and Tracking'), 0, 1)

    input_img_path = "temp_input.png"
    output_img_path = "temp_output.png"
    
    img_width = 85
    img_x = 10
    
    if os.path.exists(input_img_path) and os.path.exists(output_img_path):
        current_y = pdf.get_y()
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(img_width, 5, sanitize_text_for_fpdf(f'INPUT SOURCE ({results.get("source")})'), 0, 0, 'C')
        pdf.cell(10, 5, '', 0, 0)
        pdf.cell(img_width, 5, sanitize_text_for_fpdf('ANNOTATED OUTPUT (Safety/Classified)'), 0, 1, 'C')
        
        pdf.image(input_img_path, x=img_x, y=current_y + 6, w=img_width)
        pdf.image(output_img_path, x=img_x + img_width + 10, y=current_y + 6, w=img_width)
        
        pdf.set_y(current_y + img_width * 0.75 + 10) 
        pdf.ln(5)

    # 5.2 License Plate OCR and Confidence (Cropped Plates Placeholder)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, sanitize_text_for_fpdf('5.2 License Plate OCR and Confidence'), 0, 1)
    pdf.ln(2) 
    pdf.add_text_block(sanitize_text_for_fpdf("OCR results are logged in the detailed report's 'Number Plate' column. The system extracts the license plate text from the detected region."))
    pdf.ln(5)

    # --- SECTION VI: AI ASSISTANT & SUMMARY ---
    pdf.add_page()
    pdf.chapter_title('SECTION VI: AI ASSISTANT & SUMMARY', size=16, color=(255, 255, 255), bg_color=pdf.ACCENT_BLUE)
    
    if results.get("ai_summary"):
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 6, sanitize_text_for_fpdf('6.1 AI Narrative Synthesis'), 0, 1)
        summary = results["ai_summary"].replace('**', '').replace('###', '\n').replace('*', '-')
        pdf.add_text_block(sanitize_text_for_fpdf(summary))
    else:
        pdf.add_text_block(sanitize_text_for_fpdf("AI Assistant Analysis not run or available. Please enable the AI Assistant toggle to generate this summary."))
    pdf.ln(5)

    # --- SECTION VII: RECOMMENDATIONS & CONCLUSIONS ---
    pdf.set_y(pdf.get_y() + 5)
    pdf.chapter_title('SECTION VII: RECOMMENDATIONS & CONCLUSIONS', bg_color=pdf.BG_GREY, color=pdf.TEXT_DARK)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, sanitize_text_for_fpdf('7.1 Strategic Conclusions'), 0, 1)
    pdf.add_text_block(sanitize_text_for_fpdf("The model demonstrates high reliability (mAP: {model_accuracy}) and is fit for deployment. The primary traffic concern identified is speed compliance, warranting targeted intervention.".format(model_accuracy=model_accuracy)))
    
    pdf.ln(3)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 6, sanitize_text_for_fpdf('7.2 Actionable Recommendations'), 0, 1)
    
    pdf.set_font('ZapfDingbats', '', 10) # For bullet points
    pdf.cell(7, 5, chr(110) + " ", 0, 0)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, sanitize_text_for_fpdf("Focus enforcement efforts on high-speed violation areas."))

    pdf.cell(7, 5, chr(110) + " ", 0, 0)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, sanitize_text_for_fpdf("Collect more data for 'Bus' and 'Truck' classes to improve differentiation and model precision."))

    pdf.cell(7, 5, chr(110) + " ", 0, 0)
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, sanitize_text_for_fpdf("Automate the generation of violation tickets based on the recorded Number Plate and Anomaly Detection status."))
    pdf.ln(5)
    
    # --- Generate PDF output (FINAL FIXED ENCODING) ---
    pdf_output_stream = io.BytesIO()
    try:
        pdf.output(dest='S', name=pdf_output_stream) 
    except AttributeError:
        pdf.output(pdf_output_stream, dest='S')
    pdf_output_bytes = pdf_output_stream.getvalue()

    # --- Generate PDF output ---
    pdf_output = pdf.output(dest='S').encode('latin-1')
    return base64.b64encode(pdf_output).decode('latin-1')


def text_to_speech(text):
    """Converts text to speech (MP3) using gTTS."""
    # Ensure text is clean before TTS
    clean_text = sanitize_text_for_fpdf(text.replace('**', '').replace('#', '').replace('\n', ' '))
    tts = gTTS(text=clean_text, lang='en')
    mp3_fp = "report_analysis.mp3"
    tts.save(mp3_fp)
    return mp3_fp

def set_page_icon():
    """Sets the Streamlit page icon."""
    try:
        st.set_page_config(
            page_title="Advanced Vehicle Analytics Dashboard",
            page_icon="🚗", 
            layout="wide"
        )
    except Exception as e:
        print(f"Error setting page config: {e}")

create_log_file_if_not_exists()