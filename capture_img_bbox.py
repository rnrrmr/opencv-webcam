import cv2

tracker_type = 'DRPN'
if tracker_type == 'CSRT':
    tracker = cv2.TrackerCSRT.create()
elif tracker_type == 'DRPN':
    params = cv2.TrackerDaSiamRPN_Params()
    params.model = "model/DaSiamRPN/dasiamrpn_model.onnx"
    params.kernel_r1 = "model/DaSiamRPN/dasiamrpn_kernel_r1.onnx"
    params.kernel_cls1 = "model/DaSiamRPN/dasiamrpn_kernel_cls1.onnx"
    tracker = cv2.TrackerDaSiamRPN.create(params)
    # tracker = cv2.TrackerDaSiamRPN_create(params)
    
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN.create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF.create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL.create()
elif tracker_type == 'NANO':
    params = cv2.TrackerNano_Params()
    params.backbone = 'model/Nano/nanotrack_backbone.onnx'
    params.neckhead = 'model/Nano/nanotrack_head.onnx'
    tracker = cv2.TrackerNano.create()
    score = cv2.TrackerNano.getTrackingScore()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("camera open failed")
    exit()

frame = None  # 초기 프레임 설정
bbox = (0, 0, 0, 0)  # 초기 바운딩 박스 설정

while True:
    if frame is None or bbox == (0, 0, 0, 0):
        ret, frame = cap.read()
        if not ret:
            print("Can't read camera")
            break

        frame = cv2.resize(frame, (640, 480))
        bbox = cv2.selectROI(frame, False)  # True일 경우 십자모양 박스
        while bbox == (0, 0, 0, 0):
            bbox = cv2.selectROI(frame, False)

        # 트래커 초기화
        tracker.init(frame, bbox)
        start = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret:
        cv2.destroyAllWindows()
        break

    # 처리시간 측정
    timer = cv2.getTickCount()

    # 추적 실행
    frame = cv2.resize(frame, (640, 480))
    ret_tracker, bbox = tracker.update(frame)

    # 처리시간 측정
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    elapsed_time = (cv2.getTickCount() - start) / cv2.getTickFrequency()  # 트래커가 작동한 시간

    m = int(elapsed_time // 60)
    s = int(elapsed_time % 60)

    # 추적 결과 표시
    if ret_tracker:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.putText(frame, "Program will close automatically", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        break
        
    # BGR 000 - blk, 255255255 - wht
    cv2.putText(frame, tracker_type + " Tracker", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "FPS: " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"Elapsed Time: {m:02d}:{s:02d}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 동작 설명
    cv2.putText(frame, "Press P to capture", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "R to reselect, Q to quit", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Tracking", frame)
    

    # 키 입력 처리
    key = cv2.waitKey(1)

    if key == ord('p'):
        cv2.imwrite('my_file/img/img_cap_ori.png', frame)
        cv2.imwrite('my_file/img/img_cap_gray.png', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        new_img = cv2.imread("my_file/img/img_cap_ori.png", cv2.IMREAD_COLOR)
        denoised_img = cv2.fastNlMeansDenoisingColored(new_img, None, 15, 15, 5, 10)
        cv2.imwrite('my_file/img/img_cap_denoised.png', denoised_img)

        print("Captured image saved")

    if key == ord('r'):
        frame = None  # 초기 프레임 재설정
        bbox = (0, 0, 0, 0)  # 초기 바운딩 박스 재설정

    if key == ord('q'):
        break

cap.release()

# 이미지 불러오기
img_cap_ori = cv2.imread('my_file/img/img_cap_ori.png', cv2.IMREAD_COLOR)
img_cap_gray = cv2.imread('my_file/img/img_cap_gray.png', cv2.IMREAD_COLOR)
img_cap_denoised = cv2.imread('my_file/img/img_cap_denoised.png', cv2.IMREAD_COLOR)

# 이미지 창 띄우기
# cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
# cv2.moveWindow('Original', 100, 100)
cv2.imshow('Original', img_cap_ori)

# cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
# cv2.moveWindow('Grayscale', 580, 200)
cv2.imshow('Grayscale', img_cap_gray)

# cv2.namedWindow('Denoised', cv2.WINDOW_NORMAL)
# cv2.moveWindow('Denoised', 1060, 300)
cv2.imshow('Denoised', img_cap_denoised)

cv2.waitKey(0)
cv2.destroyAllWindows()

