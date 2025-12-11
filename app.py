import streamlit as st
import cv2
import numpy as np
import tempfile
import math
from mmpose.apis import MMPoseInferencer

# --- CẤU HÌNH ---
MODEL_CFG = 'rtmpose-m_8xb256-420e_coco-256x192' 
CONFIDENCE_THRESHOLD = 0.5 
DOWN_ANGLE_THRESH = 90
UP_ANGLE_THRESH = 160
FRAME_SKIP = 5 

# --- HÀM TÍNH GÓC KHUỶU TAY ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def check_body_posture(kpts, scores):
    """
    Kiểm tra tư thế Push-up hỗ trợ cả góc nghiêng (Side) và trực diện (Front).
    Input: kpts (mảng 17x2), scores (mảng 17)
    """
    # Index COCO: L_Shoulder(5), R_Shoulder(6), L_Hip(11), R_Hip(12), L_Ankle(15), R_Ankle(16)
    
    # 1. Lấy tọa độ trung bình Vai và Hông
    shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
    shoulder_x_dist = abs(kpts[5][0] - kpts[6][0]) # Chiều rộng vai
    
    hip_y = (kpts[11][1] + kpts[12][1]) / 2
    
    # Lấy điểm chân (ưu tiên Ankle, nếu không thấy thì lấy Hip làm chuẩn tạm)
    if scores[15] > 0.3 or scores[16] > 0.3:
        ankle_y = (kpts[15][1] + kpts[16][1]) / 2
        ankle_x = (kpts[15][0] + kpts[16][0]) / 2
    else:
        # Nếu chân bị khuất (thường gặp ở front view), dùng Hông để tính góc
        ankle_y, ankle_x = hip_y, (kpts[11][0] + kpts[12][0]) / 2

    shoulder_center_x = (kpts[5][0] + kpts[6][0]) / 2
    
    # 2. Tính góc nghiêng cơ thể (Side View Logic)
    dy = ankle_y - shoulder_y
    dx = ankle_x - shoulder_center_x
    angle_rad = math.atan2(abs(dy), abs(dx))
    angle_deg = math.degrees(angle_rad)
    
    # CASE A: Nằm ngang (Side View) -> Góc < 60 độ
    if angle_deg < 60:
        return True, angle_deg, "Side View"

    # CASE B: Hướng đầu vào Cam (Front View) -> Góc ~ 90 độ (Đứng)
    # Lúc này ta check tỷ lệ Thân / Vai
    # Ở góc trực diện, thân (Vai xuống Hông) bị ngắn lại do phối cảnh
    torso_length = abs(hip_y - shoulder_y)
    
    # Ngưỡng: Nếu chiều dài thân < 1.4 lần chiều rộng vai -> Đang nằm hướng vào cam
    # (Người đứng bình thường thì thân dài hơn vai nhiều)
    if shoulder_x_dist > 0: # Tránh chia cho 0
        ratio = torso_length / shoulder_x_dist
        if ratio < 1.4: 
            return True, angle_deg, f"Front View (R={ratio:.1f})"
    
    return False, angle_deg, "Stand Up"

# --- GIAO DIỆN ---
st.set_page_config(page_title="Smart AI Push-up Counter", layout="wide")
st.title("🛡️ Smart AI Push-up (Anti-Cheat)")
st.markdown("Phiên bản tích hợp **Posture Check**: Chỉ đếm khi người dùng nằm ở tư thế Push-up.")

st.sidebar.header("Cài đặt")
source_option = st.sidebar.selectbox("Chọn nguồn video", ["Webcam", "Upload Video"])

@st.cache_resource
def load_model():
    return MMPoseInferencer(pose2d=MODEL_CFG)

with st.spinner('Đang tải mô hình AI...'):
    inferencer = load_model()

# --- HÀM TÌM KIẾM WEBCAM ---
def get_webcams():
    """Kiểm tra 5 cổng đầu tiên (0-4) để tìm camera."""
    available_cams = []
    # Kiểm tra 5 index đầu tiên (thường camera chỉ nằm trong khoảng 0-3)
    for i in range(5): 
        # Thử mở camera với backend DirectShow (tốt cho Windows/Camera ảo)
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
        
        # Nếu không dùng Windows, hoặc code trên lỗi, hãy thử dòng dưới (bỏ comment):
        # cap = cv2.VideoCapture(i) 
        
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cams.append(i)
            cap.release()
    return available_cams

input_path = None
if source_option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Tải lên video...", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        input_path = tfile.name
elif source_option == "Webcam":
    webcam_indices = get_webcams()
    if webcam_indices:
        # Tạo danh sách hiển thị
        webcam_options = {f"Webcam {i}": i for i in webcam_indices}
        
        selected_key = st.sidebar.selectbox(
            "Chọn Webcam:", 
            list(webcam_options.keys())
        )
        input_path = webcam_options[selected_key]
        st.sidebar.info(f"Đang sử dụng Webcam {input_path}")
    else:
        st.sidebar.error("Không tìm thấy webcam nào. Vui lòng kiểm tra kết nối.")

start_button = st.sidebar.button("Bắt đầu Phân tích", type="primary")
stop_button = st.sidebar.button("Dừng lại")

if start_button and input_path is not None:
    cap = cv2.VideoCapture(input_path, cv2.CAP_DSHOW)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Adaptive Layout
    if width < height: 
        col1, col2, col3 = st.columns([2, 2, 2])
        with col2: st_frame = st.empty()
    else:
        col1, col2, col3 = st.columns([0.5, 4, 0.5])
        with col2: st_frame = st.empty()

    st_info = st.sidebar.empty()
    
    counter = 0
    stage = "UP"
    active_arm = "None"
    frame_count = 0
    posture_status = "Waiting..." # Trạng thái tư thế
    posture_valid = False
    
    last_viz_frame = None 

    while cap.isOpened():
        if stop_button: break
        
        ret, frame = cap.read()
        if not ret:
            st.sidebar.warning("Kết thúc video.")
            break
            
        if frame.shape[1] > 1280:
            scale = 1280 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

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

                l_pts = [kpts[5], kpts[7], kpts[9]] # Vai, Khuỷu, Cổ tay trái
                r_pts = [kpts[6], kpts[8], kpts[10]] # Vai, Khuỷu, Cổ tay phải
                
                # Điểm dùng để check tư thế: Vai và Cổ chân (Ankle)
                # COCO Ankle: Left(15), Right(16)
                # Nếu không thấy chân, dùng Hông: Left(11), Right(12)
                l_body_pts = [kpts[5], kpts[15] if scores[15] > 0.3 else kpts[11]] 
                r_body_pts = [kpts[6], kpts[16] if scores[16] > 0.3 else kpts[12]]

                l_conf = (scores[5] + scores[7] + scores[9]) / 3
                r_conf = (scores[6] + scores[8] + scores[10]) / 3

                current_angle = 0
                selected_color = (0, 0, 0)
                
                # --- CHỌN TAY ---
                if l_conf > CONFIDENCE_THRESHOLD and l_conf >= r_conf:
                    active_arm = "Left"
                    current_angle = calculate_angle(l_pts[0], l_pts[1], l_pts[2])
                    target_pts, target_elbow = l_pts, l_pts[1]
                    body_segment = l_body_pts
                    selected_color = (0, 255, 0)
                elif r_conf > CONFIDENCE_THRESHOLD and r_conf > l_conf:
                    active_arm = "Right"
                    current_angle = calculate_angle(r_pts[0], r_pts[1], r_pts[2])
                    target_pts, target_elbow = r_pts, r_pts[1]
                    body_segment = r_body_pts
                    selected_color = (255, 165, 0)
                else:
                    active_arm = "Lost"
                
                if active_arm != "Lost":
                    # Truyền toàn bộ kpts và scores vào hàm mới
                    is_valid_pose, body_angle, view_mode = check_body_posture(kpts, scores)
                    
                    if is_valid_pose:
                        posture_valid = True
                        posture_status = f"Push-up ({view_mode})" # Hiện rõ đang view góc nào
                        
                        # Logic đếm (giữ nguyên)
                        if current_angle > UP_ANGLE_THRESH: stage = "UP"
                        if current_angle < DOWN_ANGLE_THRESH and stage == "UP":
                            stage = "DOWN"
                            counter += 1
                        
                        # Vẽ màu Xanh/Cam
                        cv2.line(viz_frame, (int(target_pts[0][0]), int(target_pts[0][1])), (int(target_pts[1][0]), int(target_pts[1][1])), selected_color, 4)
                        cv2.line(viz_frame, (int(target_pts[1][0]), int(target_pts[1][1])), (int(target_pts[2][0]), int(target_pts[2][1])), selected_color, 4)
                        cv2.putText(viz_frame, str(int(current_angle)), (int(target_elbow[0]), int(target_elbow[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    else:
                        posture_status = "Stand Up"
                        # Vẽ màu Đỏ
                        cv2.line(viz_frame, (int(target_pts[0][0]), int(target_pts[0][1])), (int(target_pts[1][0]), int(target_pts[1][1])), (0, 0, 255), 4)
                        cv2.line(viz_frame, (int(target_pts[1][0]), int(target_pts[1][1])), (int(target_pts[2][0]), int(target_pts[2][1])), (0, 0, 255), 4)
            last_viz_frame = viz_frame
            
            # Cập nhật thông số
            st_info.markdown(f"""
            ### 📊 Thống kê
            - **Số lần:** {counter}
            - **Tay:** {active_arm}
            - **Tư thế:** {posture_status}
            - **Góc gập khuỷu tay:** {int(current_angle) if active_arm != 'Lost' else 0}°
            - **Góc nghiêng thân:** {int(body_angle) if active_arm != 'Lost' else 0}°
            """)
        
        else:
            if last_viz_frame is not None: viz_frame = last_viz_frame
            else: viz_frame = frame

        # Vẽ Info Box
        # Đổi màu box nếu sai tư thế
        box_color = (0, 200, 0) if posture_valid else (50, 50, 200)
        
        cv2.rectangle(viz_frame, (0,0), (310, 85), box_color, -1)
        cv2.putText(viz_frame, f'REPS: {counter}', (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(viz_frame, f'{posture_status}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        st_frame.image(viz_frame, channels="BGR", width='stretch')
        
    cap.release()