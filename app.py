# import streamlit as st
# import cv2
# import numpy as np
# import tempfile
# import math
# from mmpose.apis import MMPoseInferencer

# # --- CẤU HÌNH ---
# MODEL_CFG = 'rtmpose-m_8xb256-420e_coco-256x192' 
# CONFIDENCE_THRESHOLD = 0.5 
# DOWN_ANGLE_THRESH = 90
# UP_ANGLE_THRESH = 160
# FRAME_SKIP = 5 

# # --- HÀM TÍNH GÓC KHUỶU TAY ---
# def calculate_angle(a, b, c):
#     a, b, c = np.array(a), np.array(b), np.array(c)
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#     if angle > 180.0: angle = 360 - angle
#     return angle

# def check_body_posture(kpts, scores):
#     """
#     Kiểm tra tư thế Push-up hỗ trợ cả góc nghiêng (Side) và trực diện (Front).
#     Input: kpts (mảng 17x2), scores (mảng 17)
#     """
#     # Index COCO: L_Shoulder(5), R_Shoulder(6), L_Hip(11), R_Hip(12), L_Ankle(15), R_Ankle(16)
    
#     # 1. Lấy tọa độ trung bình Vai và Hông
#     shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
#     shoulder_x_dist = abs(kpts[5][0] - kpts[6][0]) # Chiều rộng vai
    
#     hip_y = (kpts[11][1] + kpts[12][1]) / 2
    
#     # Lấy điểm chân (ưu tiên Ankle, nếu không thấy thì lấy Hip làm chuẩn tạm)
#     if scores[15] > 0.3 or scores[16] > 0.3:
#         ankle_y = (kpts[15][1] + kpts[16][1]) / 2
#         ankle_x = (kpts[15][0] + kpts[16][0]) / 2
#     else:
#         # Nếu chân bị khuất (thường gặp ở front view), dùng Hông để tính góc
#         ankle_y, ankle_x = hip_y, (kpts[11][0] + kpts[12][0]) / 2

#     shoulder_center_x = (kpts[5][0] + kpts[6][0]) / 2
    
#     # 2. Tính góc nghiêng cơ thể (Side View Logic)
#     dy = ankle_y - shoulder_y
#     dx = ankle_x - shoulder_center_x
#     angle_rad = math.atan2(abs(dy), abs(dx))
#     angle_deg = math.degrees(angle_rad)
    
#     # CASE A: Nằm ngang (Side View) -> Góc < 60 độ
#     if angle_deg < 60:
#         return True, angle_deg, "Side View"

#     # CASE B: Hướng đầu vào Cam (Front View) -> Góc ~ 90 độ (Đứng)
#     # Lúc này ta check tỷ lệ Thân / Vai
#     # Ở góc trực diện, thân (Vai xuống Hông) bị ngắn lại do phối cảnh
#     torso_length = abs(hip_y - shoulder_y)
    
#     # Ngưỡng: Nếu chiều dài thân < 1.4 lần chiều rộng vai -> Đang nằm hướng vào cam
#     # (Người đứng bình thường thì thân dài hơn vai nhiều)
#     if shoulder_x_dist > 0: # Tránh chia cho 0
#         ratio = torso_length / shoulder_x_dist
#         if ratio < 1.4: 
#             return True, angle_deg, f"Front View (R={ratio:.1f})"
    
#     return False, angle_deg, "Stand Up"

# # --- GIAO DIỆN ---
# st.set_page_config(page_title="Smart AI Push-up Counter", layout="wide")
# st.title("🛡️ Smart AI Push-up (Anti-Cheat)")
# st.markdown("Phiên bản tích hợp **Posture Check**: Chỉ đếm khi người dùng nằm ở tư thế Push-up.")

# st.sidebar.header("Cài đặt")
# source_option = st.sidebar.selectbox("Chọn nguồn video", ["Webcam", "Upload Video"])

# @st.cache_resource
# def load_model():
#     return MMPoseInferencer(pose2d=MODEL_CFG)

# with st.spinner('Đang tải mô hình AI...'):
#     inferencer = load_model()

# # --- HÀM TÌM KIẾM WEBCAM ---
# def get_webcams():
#     """Kiểm tra 5 cổng đầu tiên (0-4) để tìm camera."""
#     available_cams = []
#     # Kiểm tra 5 index đầu tiên (thường camera chỉ nằm trong khoảng 0-3)
#     for i in range(5): 
#         # Thử mở camera với backend DirectShow (tốt cho Windows/Camera ảo)
#         cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
        
#         # Nếu không dùng Windows, hoặc code trên lỗi, hãy thử dòng dưới (bỏ comment):
#         # cap = cv2.VideoCapture(i) 
        
#         if cap.isOpened():
#             ret, _ = cap.read()
#             if ret:
#                 available_cams.append(i)
#             cap.release()
#     return available_cams

# input_path = None
# if source_option == "Upload Video":
#     uploaded_file = st.sidebar.file_uploader("Tải lên video...", type=['mp4', 'mov', 'avi'])
#     if uploaded_file is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())
#         input_path = tfile.name
# elif source_option == "Webcam":
#     webcam_indices = get_webcams()
#     if webcam_indices:
#         # Tạo danh sách hiển thị
#         webcam_options = {f"Webcam {i}": i for i in webcam_indices}
        
#         selected_key = st.sidebar.selectbox(
#             "Chọn Webcam:", 
#             list(webcam_options.keys())
#         )
#         input_path = webcam_options[selected_key]
#         st.sidebar.info(f"Đang sử dụng Webcam {input_path}")
#     else:
#         st.sidebar.error("Không tìm thấy webcam nào. Vui lòng kiểm tra kết nối.")

# start_button = st.sidebar.button("Bắt đầu Phân tích", type="primary")
# stop_button = st.sidebar.button("Dừng lại")

# if start_button and input_path is not None:
#     cap = cv2.VideoCapture(input_path, cv2.CAP_DSHOW)
    
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # Adaptive Layout
#     if width < height: 
#         col1, col2, col3 = st.columns([2, 2, 2])
#         with col2: st_frame = st.empty()
#     else:
#         col1, col2, col3 = st.columns([0.5, 4, 0.5])
#         with col2: st_frame = st.empty()

#     st_info = st.sidebar.empty()
    
#     counter = 0
#     stage = "UP"
#     active_arm = "None"
#     frame_count = 0
#     posture_status = "Waiting..." # Trạng thái tư thế
#     posture_valid = False
    
#     last_viz_frame = None 

#     while cap.isOpened():
#         if stop_button: break
        
#         ret, frame = cap.read()
#         if not ret:
#             st.sidebar.warning("Kết thúc video.")
#             break
            
#         if frame.shape[1] > 1280:
#             scale = 1280 / frame.shape[1]
#             frame = cv2.resize(frame, None, fx=scale, fy=scale)

#         frame_count += 1
        
#         if frame_count % FRAME_SKIP == 0:
#             result_generator = inferencer(frame, return_vis=False)
#             result = next(result_generator)
#             viz_frame = frame.copy()
            
#             predictions = result['predictions'][0]
#             if predictions:
#                 person = predictions[0]
#                 kpts = np.array(person['keypoints'])
#                 scores = np.array(person['keypoint_scores'])

#                 l_pts = [kpts[5], kpts[7], kpts[9]] # Vai, Khuỷu, Cổ tay trái
#                 r_pts = [kpts[6], kpts[8], kpts[10]] # Vai, Khuỷu, Cổ tay phải
                
#                 # Điểm dùng để check tư thế: Vai và Cổ chân (Ankle)
#                 # COCO Ankle: Left(15), Right(16)
#                 # Nếu không thấy chân, dùng Hông: Left(11), Right(12)
#                 l_body_pts = [kpts[5], kpts[15] if scores[15] > 0.3 else kpts[11]] 
#                 r_body_pts = [kpts[6], kpts[16] if scores[16] > 0.3 else kpts[12]]

#                 l_conf = (scores[5] + scores[7] + scores[9]) / 3
#                 r_conf = (scores[6] + scores[8] + scores[10]) / 3

#                 current_angle = 0
#                 selected_color = (0, 0, 0)
                
#                 # --- CHỌN TAY ---
#                 if l_conf > CONFIDENCE_THRESHOLD and l_conf >= r_conf:
#                     active_arm = "Left"
#                     current_angle = calculate_angle(l_pts[0], l_pts[1], l_pts[2])
#                     target_pts, target_elbow = l_pts, l_pts[1]
#                     body_segment = l_body_pts
#                     selected_color = (0, 255, 0)
#                 elif r_conf > CONFIDENCE_THRESHOLD and r_conf > l_conf:
#                     active_arm = "Right"
#                     current_angle = calculate_angle(r_pts[0], r_pts[1], r_pts[2])
#                     target_pts, target_elbow = r_pts, r_pts[1]
#                     body_segment = r_body_pts
#                     selected_color = (255, 165, 0)
#                 else:
#                     active_arm = "Lost"
                
#                 if active_arm != "Lost":
#                     # Truyền toàn bộ kpts và scores vào hàm mới
#                     is_valid_pose, body_angle, view_mode = check_body_posture(kpts, scores)
                    
#                     if is_valid_pose:
#                         posture_valid = True
#                         posture_status = f"Push-up ({view_mode})" # Hiện rõ đang view góc nào
                        
#                         # Logic đếm (giữ nguyên)
#                         if current_angle > UP_ANGLE_THRESH: stage = "UP"
#                         if current_angle < DOWN_ANGLE_THRESH and stage == "UP":
#                             stage = "DOWN"
#                             counter += 1
                        
#                         # Vẽ màu Xanh/Cam
#                         cv2.line(viz_frame, (int(target_pts[0][0]), int(target_pts[0][1])), (int(target_pts[1][0]), int(target_pts[1][1])), selected_color, 4)
#                         cv2.line(viz_frame, (int(target_pts[1][0]), int(target_pts[1][1])), (int(target_pts[2][0]), int(target_pts[2][1])), selected_color, 4)
#                         cv2.putText(viz_frame, str(int(current_angle)), (int(target_elbow[0]), int(target_elbow[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#                     else:
#                         posture_status = "Stand Up"
#                         # Vẽ màu Đỏ
#                         cv2.line(viz_frame, (int(target_pts[0][0]), int(target_pts[0][1])), (int(target_pts[1][0]), int(target_pts[1][1])), (0, 0, 255), 4)
#                         cv2.line(viz_frame, (int(target_pts[1][0]), int(target_pts[1][1])), (int(target_pts[2][0]), int(target_pts[2][1])), (0, 0, 255), 4)
#             last_viz_frame = viz_frame
            
#             # Cập nhật thông số
#             st_info.markdown(f"""
#             ### 📊 Thống kê
#             - **Số lần:** {counter}
#             - **Tay:** {active_arm}
#             - **Tư thế:** {posture_status}
#             - **Góc gập khuỷu tay:** {int(current_angle) if active_arm != 'Lost' else 0}°
#             - **Góc nghiêng thân:** {int(body_angle) if active_arm != 'Lost' else 0}°
#             """)
        
#         else:
#             if last_viz_frame is not None: viz_frame = last_viz_frame
#             else: viz_frame = frame

#         # Vẽ Info Box
#         # Đổi màu box nếu sai tư thế
#         box_color = (0, 200, 0) if posture_valid else (50, 50, 200)
        
#         cv2.rectangle(viz_frame, (0,0), (310, 85), box_color, -1)
#         cv2.putText(viz_frame, f'REPS: {counter}', (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
#         cv2.putText(viz_frame, f'{posture_status}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

#         st_frame.image(viz_frame, channels="BGR", width='stretch')
        
#     cap.release()



import streamlit as st
import cv2
import numpy as np
import tempfile
import math
from mmpose.apis import MMPoseInferencer

# --- CẤU HÌNH HỆ THỐNG ---
st.set_page_config(
    page_title="AI Push-up Trainer",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Config Model
MODEL_CFG = 'rtmpose-m_8xb256-420e_coco-256x192' 
CONFIDENCE_THRESHOLD = 0.5 
DOWN_ANGLE_THRESH = 90
UP_ANGLE_THRESH = 160
FRAME_SKIP = 5 
MAX_DIMENSION = 1280 # Giới hạn cạnh lớn nhất (để tối ưu FPS và hiển thị)

# --- CSS: TỐI ƯU GIAO DIỆN ---
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1e1e24; color: white; }
    .stMetric { background-color: #f0f2f6; border-radius: 10px; padding: 10px; border: 1px solid #e0e0e0; }
    /* Căn giữa video */
    div.stImage { display: flex; justify-content: center; }
    div.stImage > img { object-fit: contain; max-height: 80vh; } 
</style>
""", unsafe_allow_html=True)

# --- CÁC HÀM XỬ LÝ LOGIC (CORE LOGIC) ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def check_body_posture(kpts, scores):
    """
    LOGIC V2: Yêu cầu thấy toàn thân (hoặc ít nhất là đầu gối) mới bắt đầu xét duyệt.
    """
    # 1. Định nghĩa các điểm quan trọng
    # COCO Keypoints: 
    # Vai: 5,6 | Hông: 11,12 | Đầu gối: 13,14 | Cổ chân: 15,16
    
    avg_shoulder_conf = (scores[5] + scores[6]) / 2
    avg_hip_conf = (scores[11] + scores[12]) / 2
    
    # Kiểm tra xem có thấy Chân (Knee hoặc Ankle) không?
    # Chỉ cần thấy 1 trong 2 bên (Trái hoặc Phải) là đủ
    has_knees = scores[13] > 0.4 or scores[14] > 0.4
    has_ankles = scores[15] > 0.4 or scores[16] > 0.4
    
    # --- ĐIỀU KIỆN 1: PHẢI THẤY NGƯỜI ---
    # Nếu không thấy vai, hông HOẶC (không thấy đầu gối VÀ không thấy cổ chân)
    if avg_shoulder_conf < 0.4 or avg_hip_conf < 0.4 or (not has_knees and not has_ankles):
        return False, 0, "Show Full Body" # Bắt buộc phải lùi ra xa để thấy chân

    # 2. Lấy tọa độ
    shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
    shoulder_center_x = (kpts[5][0] + kpts[6][0]) / 2
    shoulder_width = abs(kpts[5][0] - kpts[6][0])
    
    hip_y = (kpts[11][1] + kpts[12][1]) / 2
    
    # Ưu tiên lấy Ankle làm mốc, nếu không thì lấy Knee (cho kiểu hít đất quỳ gối)
    if has_ankles:
        foot_y = (kpts[15][1] + kpts[16][1]) / 2
        foot_x = (kpts[15][0] + kpts[16][0]) / 2
        body_part = "Ankles"
    else:
        foot_y = (kpts[13][1] + kpts[14][1]) / 2
        foot_x = (kpts[13][0] + kpts[14][0]) / 2
        body_part = "Knees"

    # 3. Tính góc nghiêng thân người (Vai nối tới Chân)
    dy = foot_y - shoulder_y
    dx = foot_x - shoulder_center_x
    angle_rad = math.atan2(abs(dy), abs(dx))
    angle_deg = math.degrees(angle_rad)
    
    # --- ĐIỀU KIỆN 2: PHÂN LOẠI VIEW ---
    
    # CASE A: Side View (Nằm ngang) -> Góc < 50 độ (Siết chặt hơn 60)
    if angle_deg < 50:
        return True, angle_deg, f"Side View ({body_part})"

    # CASE B: Front View (Trực diện)
    # Logic: Khi nằm trực diện, thân người (Vai->Hông) sẽ bị ngắn lại so với Vai.
    torso_length = abs(hip_y - shoulder_y)
    
    if shoulder_width > 0:
        ratio = torso_length / shoulder_width
        # Nếu ngồi: Thân rất dài so với vai (Ratio > 1.5 - 2.0)
        # Nếu hít đất trực diện: Thân ngắn lại (Ratio < 1.3)
        if ratio < 1.3: 
            return True, angle_deg, f"Front View (R={ratio:.1f})"
        else:
            return False, angle_deg, "Sitting/Standing" # Phát hiện ngồi
            
    return False, angle_deg, "Wrong Pose"

# --- HÀM XỬ LÝ HÌNH ẢNH & GIAO DIỆN (NEW) ---

def smart_resize(frame, max_dim=1280):
    """
    Resize thông minh: Giữ nguyên tỷ lệ khung hình.
    Chỉ resize nếu ảnh quá lớn để đảm bảo tốc độ xử lý.
    """
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame

def draw_hud_responsive(img, counter, stage, posture_status, angle, is_valid):
    """
    Vẽ HUD tự động co giãn theo độ phân giải của video.
    """
    h, w = img.shape[:2]
    
    # Tính toán Scale Factor (Dựa trên chiều rộng chuẩn 1280px)
    # Nếu ảnh rộng 640px -> scale = 0.5. Nếu 1920px -> scale = 1.5
    s = w / 1280.0 
    s = max(s, 0.5) # Không cho nhỏ quá mức đọc được

    overlay = img.copy()
    
    # Màu sắc
    status_color = (0, 255, 0) if is_valid else (0, 0, 255)
    
    # 1. Header Bar (Chiều cao thay đổi theo scale)
    bar_height = int(80 * s)
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    
    # Trộn màu (Transparency)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 2. Thông số (Font size và vị trí nhân với s)
    # REPS
    cv2.putText(img, "REPS", (int(20*s), int(30*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*s, (200, 200, 200), 1)
    cv2.putText(img, str(counter), (int(20*s), int(70*s)), cv2.FONT_HERSHEY_DUPLEX, 1.2*s, (255, 255, 255), int(2*s))
    
    # STAGE
    stage_color = (0, 255, 255) if stage == "DOWN" else (255, 255, 255)
    cv2.putText(img, "STAGE", (int(150*s), int(30*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*s, (200, 200, 200), 1)
    cv2.putText(img, stage, (int(150*s), int(70*s)), cv2.FONT_HERSHEY_DUPLEX, 1.0*s, stage_color, int(2*s))
    
    # ANGLE
    cv2.putText(img, "ANGLE", (int(300*s), int(30*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*s, (200, 200, 200), 1)
    cv2.putText(img, f"{int(angle)}", (int(300*s), int(70*s)), cv2.FONT_HERSHEY_DUPLEX, 1.0*s, (255, 255, 255), int(2*s))

    # WARNING BAR (Bottom)
    if not is_valid:
        warn_height = int(40 * s)
        cv2.rectangle(img, (0, h - warn_height), (w, h), (0, 0, 255), -1)
        cv2.putText(img, f"WARN: {posture_status}", (int(20*s), h - int(10*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7*s, (255, 255, 255), int(2*s))
    else:
        cv2.putText(img, f"Mode: {posture_status}", (w - int(250*s), int(30*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*s, (0, 255, 0), 1)

    return img

def get_webcams():
    """Scan webcam tự động"""
    available_cams = []
    for i in range(4): 
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cams.append(i)
            cap.release()
    return available_cams

# --- KHỞI TẠO HỆ THỐNG ---
st.sidebar.title("🛠️ Control Panel")
source_option = st.sidebar.radio("Nguồn Video:", ["Webcam", "Upload Video"])

input_path = None
if source_option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Chọn file video...", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        input_path = tfile.name
elif source_option == "Webcam":
    webcam_indices = get_webcams()
    if webcam_indices:
        webcam_dict = {f"Camera {i}": i for i in webcam_indices}
        selected_cam = st.sidebar.selectbox("Chọn thiết bị:", list(webcam_dict.keys()))
        input_path = webcam_dict[selected_cam]
    else:
        st.sidebar.error("⚠️ Không tìm thấy Webcam!")

# Load Model
@st.cache_resource
def load_model():
    return MMPoseInferencer(pose2d=MODEL_CFG)

if 'is_active' not in st.session_state:
    st.session_state['is_active'] = False

# 2. Nút bấm chỉ làm nhiệm vụ "Bật công tắc" (Đổi trạng thái)
if st.sidebar.button("🚀 KÍCH HOẠT HỆ THỐNG", type="primary"):
    st.session_state['is_active'] = True

# 3. Load Model dựa trên trạng thái (Chạy mỗi khi reload trang)
inferencer = None
if st.session_state['is_active']:
    with st.spinner('Đang xử lý AI...'):
        inferencer = load_model()

if st.sidebar.button("⏹️ Dừng lại"): st.session_state['is_active'] = False

# --- LAYOUT CHÍNH ---
st.title("💪 AI Push-up Trainer")
st.markdown("Hệ thống chấm điểm Push-up chuẩn thi đấu (Strict Form).")

# Layout Responsive: Cột video tự co giãn, Cột chỉ số cố định
main_col1, main_col2 = st.columns([3, 1]) 

with main_col2:
    st.subheader("📊 Chỉ số")
    metric_count = st.empty()
    metric_stage = st.empty()
    metric_pose = st.empty()
    
    metric_count.metric("Số lần (Reps)", "0")
    metric_stage.metric("Trạng thái", "READY")
    metric_pose.info("Đang chờ video...")

with main_col1:
    st_frame = st.empty()

# --- MAIN LOOP ---
if st.session_state['is_active'] and input_path is not None:
    # Auto backend select (Fix màn hình đen)
    cap = cv2.VideoCapture(input_path)
    
    counter = 0
    stage = "UP"
    active_arm = "None"
    frame_count = 0
    posture_status = "Waiting..."
    posture_valid = False
    last_viz_frame = None 

    while cap.isOpened() and st.session_state['is_active']:
        ret, frame = cap.read()
        if not ret:
            st.warning("Kết thúc video.")
            st.session_state['is_active'] = False
            break
            
        # --- BƯỚC 1: SMART RESIZE ---
        # Đưa về kích thước chuẩn xử lý (không quá 1280px chiều dài)
        # Giúp FPS ổn định và HUD hiển thị đồng nhất
        frame = smart_resize(frame, MAX_DIMENSION)

        frame_count += 1
        
        if frame_count % FRAME_SKIP == 0:
            result_generator = inferencer(frame, return_vis=False)
            result = next(result_generator)
            viz_frame = frame.copy()
            
            predictions = result['predictions'][0]
            if predictions:
                person = predictions[0]
                kpts = np.array(person['keypoints'])
                scores = np.array(person['keypoint_scores'])

                l_pts = [kpts[5], kpts[7], kpts[9]]
                r_pts = [kpts[6], kpts[8], kpts[10]]
                
                l_conf = (scores[5] + scores[7] + scores[9]) / 3
                r_conf = (scores[6] + scores[8] + scores[10]) / 3

                current_angle = 0
                s = frame.shape[1] / 1280.0 # Scale factor cho độ dày nét vẽ
                thick = max(int(4 * s), 1)

                if l_conf > CONFIDENCE_THRESHOLD and l_conf >= r_conf:
                    active_arm = "Left"
                    current_angle = calculate_angle(l_pts[0], l_pts[1], l_pts[2])
                    target_pts = l_pts
                elif r_conf > CONFIDENCE_THRESHOLD and r_conf > l_conf:
                    active_arm = "Right"
                    current_angle = calculate_angle(r_pts[0], r_pts[1], r_pts[2])
                    target_pts = r_pts
                else:
                    active_arm = "Lost"
                
                if active_arm != "Lost":
                    is_valid_pose, body_angle, view_mode = check_body_posture(kpts, scores)
                    posture_valid = is_valid_pose
                    posture_status = view_mode 

                    if is_valid_pose:
                        if current_angle > UP_ANGLE_THRESH: stage = "UP"
                        if current_angle < DOWN_ANGLE_THRESH and stage == "UP":
                            stage = "DOWN"
                            counter += 1
                        
                        # Vẽ Xương Xanh
                        cv2.line(viz_frame, (int(target_pts[0][0]), int(target_pts[0][1])), (int(target_pts[1][0]), int(target_pts[1][1])), (0, 255, 0), thick)
                        cv2.line(viz_frame, (int(target_pts[1][0]), int(target_pts[1][1])), (int(target_pts[2][0]), int(target_pts[2][1])), (0, 255, 0), thick)
                    else:
                        # Vẽ Xương Đỏ
                        cv2.line(viz_frame, (int(target_pts[0][0]), int(target_pts[0][1])), (int(target_pts[1][0]), int(target_pts[1][1])), (0, 0, 255), thick)
                        cv2.line(viz_frame, (int(target_pts[1][0]), int(target_pts[1][1])), (int(target_pts[2][0]), int(target_pts[2][1])), (0, 0, 255), thick)
            
            # --- BƯỚC 2: VẼ HUD RESPONSIVE ---
            viz_frame = draw_hud_responsive(viz_frame, counter, stage, posture_status, current_angle if active_arm != 'Lost' else 0, posture_valid)
            last_viz_frame = viz_frame
            
            # Cập nhật thông số
            metric_count.metric("Số lần (Reps)", f"{counter}")
            metric_stage.metric("Trạng thái", f"{stage}")
            if posture_valid: metric_pose.success(f"{posture_status}")
            else: metric_pose.error(f"{posture_status}")

        else:
            if last_viz_frame is not None: viz_frame = last_viz_frame
            else: viz_frame = frame

        # --- BƯỚC 3: HIỂN THỊ STREAMLIT ---
        # use_container_width=True: Chìa khóa để video tự co giãn vừa khít cột
        st_frame.image(viz_frame, channels="BGR", use_container_width=True)
        
    cap.release()