import streamlit as st
import cv2
import numpy as np
import tempfile
import math
from mmpose.apis import MMPoseInferencer

st.set_page_config(
    page_title="AI Push-up Counter",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_CFG = 'rtmpose-m_8xb256-420e_coco-256x192' 
CONFIDENCE_THRESHOLD = 0.5 
DOWN_ANGLE_THRESH = 90
UP_ANGLE_THRESH = 160
FRAME_SKIP = 5 
MAX_DIMENSION = 1280

st.markdown("""
<style>
    /* Tổng thể ứng dụng nền trắng */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Sidebar màu trắng, chữ đen, có viền phải nhẹ */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f0f0f0;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #333333 !important;
    }
    
    /* Metric Card (Thẻ chỉ số) màu trắng, bóng đổ nhẹ */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Màu chữ trong Metric */
    [data-testid="stMetricLabel"] { color: #888888; }
    [data-testid="stMetricValue"] { color: #000000; font-weight: 700; }
    
    /* Tiêu đề chính */
    h1, h2, h3 {
        color: #000000 !important;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Căn giữa video */
    div.stImage { display: flex; justify-content: center; }
    div.stImage > img { object-fit: contain; max-height: 80vh; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); } 
</style>
""", unsafe_allow_html=True)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def check_body_posture(kpts, scores):
    avg_shoulder_conf = (scores[5] + scores[6]) / 2
    avg_hip_conf = (scores[11] + scores[12]) / 2
    
    has_knees = scores[13] > 0.4 or scores[14] > 0.4
    has_ankles = scores[15] > 0.4 or scores[16] > 0.4
    
    if avg_shoulder_conf < 0.4 or avg_hip_conf < 0.4 or (not has_knees and not has_ankles):
        return False, 0, "Chua thay toan than" 

    shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
    shoulder_center_x = (kpts[5][0] + kpts[6][0]) / 2
    shoulder_width = abs(kpts[5][0] - kpts[6][0])
    
    hip_y = (kpts[11][1] + kpts[12][1]) / 2
    
    if has_ankles:
        foot_y = (kpts[15][1] + kpts[16][1]) / 2
        foot_x = (kpts[15][0] + kpts[16][0]) / 2
        body_part = "Co chan"
    else:
        foot_y = (kpts[13][1] + kpts[14][1]) / 2
        foot_x = (kpts[13][0] + kpts[14][0]) / 2
        body_part = "Đau goi"

    dy = foot_y - shoulder_y
    dx = foot_x - shoulder_center_x
    angle_rad = math.atan2(abs(dy), abs(dx))
    angle_deg = math.degrees(angle_rad)
    
    if angle_deg < 50:
        return True, angle_deg, f"Goc nghieng ({body_part})"

    torso_length = abs(hip_y - shoulder_y)
    
    if shoulder_width > 0:
        ratio = torso_length / shoulder_width
        if ratio < 1.3: 
            return True, angle_deg, f"Truc dien (R={ratio:.1f})"
        else:
            return False, angle_deg, "Dang ngoi/dung"
            
    return False, angle_deg, "Sai tu the"

def smart_resize(frame, max_dim=1280):
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame

def draw_hud_white_theme(img, counter, stage, posture_status, angle, is_valid):
    h, w = img.shape[:2]
    s = w / 1280.0 
    s = max(s, 0.5)

    overlay = img.copy()

    bar_height = int(80 * s)
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (255, 255, 255), -1) 
   
    alpha = 0.85 
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
   
    cv2.putText(img, "REPS", (int(20*s), int(30*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*s, (100, 100, 100), 1)
    cv2.putText(img, str(counter), (int(20*s), int(70*s)), cv2.FONT_HERSHEY_DUPLEX, 1.2*s, (0, 0, 0), int(2*s)) 

    stage_text_color = (0, 200, 0) if stage == "DOWN" else (0, 0, 0) # Xanh hoặc Đen
    cv2.putText(img, "STAGE", (int(150*s), int(30*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*s, (100, 100, 100), 1)
    cv2.putText(img, stage, (int(150*s), int(70*s)), cv2.FONT_HERSHEY_DUPLEX, 1.0*s, stage_text_color, int(2*s))

    cv2.putText(img, "ANGLE", (int(300*s), int(30*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*s, (100, 100, 100), 1)
    cv2.putText(img, f"{int(angle)}", (int(300*s), int(70*s)), cv2.FONT_HERSHEY_DUPLEX, 1.0*s, (0, 0, 0), int(2*s))

    if not is_valid:
        warn_height = int(40 * s)
        cv2.rectangle(img, (0, h - warn_height), (w, h), (240, 240, 255), -1)
        cv2.putText(img, f"WARN: {posture_status}", (int(20*s), h - int(10*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7*s, (0, 0, 255), int(2*s)) 
    else:
        cv2.putText(img, f"Mode: {posture_status}", (w - int(300*s), int(50*s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6*s, (0, 150, 0), int(2*s)) 

    return img

def get_webcams():
    available_cams = []
    for i in range(4): 
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cams.append(i)
            cap.release()
    return available_cams

st.sidebar.title("BẢNG ĐIỀU KHIỂN")
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

@st.cache_resource
def load_model():
    return MMPoseInferencer(pose2d=MODEL_CFG)

if 'is_active' not in st.session_state:
    st.session_state['is_active'] = False

if st.sidebar.button("KÍCH HOẠT HỆ THỐNG", type="primary"):
    st.session_state['is_active'] = True

inferencer = None
if st.session_state['is_active']:
    with st.spinner('Đang xử lý AI...'):
        inferencer = load_model()

if st.sidebar.button("DỪNG LẠI"): st.session_state['is_active'] = False

st.title("AI Push-up Counter")
st.markdown("Hệ thống đếm số lần hít đất chuẩn thi đấu.")

main_col1, main_col2 = st.columns([3, 1]) 

with main_col2:
    st.subheader("Chỉ số")
    metric_count = st.empty()
    metric_stage = st.empty()
    metric_pose = st.empty()
    
    metric_count.metric("Số lần (Reps)", "0")
    metric_stage.metric("Trạng thái", "READY")
    metric_pose.info("Đang chờ video...")

with main_col1:
    st_frame = st.empty()

if st.session_state['is_active'] and input_path is not None:
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
                s = frame.shape[1] / 1280.0 
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

                        cv2.line(viz_frame, (int(target_pts[0][0]), int(target_pts[0][1])), (int(target_pts[1][0]), int(target_pts[1][1])), (0, 255, 0), thick)
                        cv2.line(viz_frame, (int(target_pts[1][0]), int(target_pts[1][1])), (int(target_pts[2][0]), int(target_pts[2][1])), (0, 255, 0), thick)
                    else:
                        cv2.line(viz_frame, (int(target_pts[0][0]), int(target_pts[0][1])), (int(target_pts[1][0]), int(target_pts[1][1])), (0, 0, 255), thick)
                        cv2.line(viz_frame, (int(target_pts[1][0]), int(target_pts[1][1])), (int(target_pts[2][0]), int(target_pts[2][1])), (0, 0, 255), thick)

            viz_frame = draw_hud_white_theme(viz_frame, counter, stage, posture_status, current_angle if active_arm != 'Lost' else 0, posture_valid)
            last_viz_frame = viz_frame
            
            metric_count.metric("Số lần (Reps)", f"{counter}")
            metric_stage.metric("Trạng thái", f"{stage}")
            if posture_valid: metric_pose.success(f"{posture_status}")
            else: metric_pose.error(f"{posture_status}")

        else:
            if last_viz_frame is not None: viz_frame = last_viz_frame
            else: viz_frame = frame

        st_frame.image(viz_frame, channels="BGR", use_container_width=True)
        
    cap.release()