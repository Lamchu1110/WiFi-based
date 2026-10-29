# Thay đổi độ tin cậy — v2, 29/10/2026

## Lỗi/giới hạn đã sửa

1. WiPose CSI-TCN trước đây nhận T=1; nay pooling chỉ theo frequency trước TCN, giữ T=5 rồi mới pool về decoder.
2. Có pose-TCN dọc W; forward eval đã kiểm chứng frame lân cận tác động output, không chỉ ràng buộc W trong loss.
3. MM-Fi các kịch bản dùng cùng window/stride/split/batch; thêm xyz/xyz_tcn/xyz_kinematic để so đóng góp trong không gian XYZ.
4. XY loss mặc định bỏ Z-weighting và phép chia32 thừa. Legacy giữ qua flag rõ ràng. Confidence WiPose được mask/weight có chuẩn hóa.
5. Symmetry mặc định khớp asymmetry của GT; pred=GT cho mọi thành phần loss bằng0. Có prior mode nếu nghiên cứu prior đối xứng.
6. GT cache và kiểm tra index thực tế cho folder/ZIP MM-Fi; không np.load GT nguyên sequence mỗi frame.
7. Validation/test MM-Fi tách theo sequence mặc định. Window không vượt sequence hoặc khoảng frame thiếu.
8. Checkpoint atomic, mid-epoch cursor, RNG Python/NumPy/Torch/CUDA, optimizer/scheduler/scaler và best weights; resume cùng protocol đã kiểm thử exact trên CPU. Khóa OS chống ghi đồng thời.
9. Metric streaming có visibility, PA alignment và degeneracy handling; thêm velocity/acceleration error trong window.
10. Bỏ torchvision import thừa; các entrypoint dùng chung engine để tránh hai implementation training lệch nhau.

## Lỗi phát hiện khi chạy test GPU

- FP16 autocast tạo NaN ngay ở Conv2d đầu nhánh SKConv trên GPU/môi trường nguồn trong bài test tổng hợp, trước optimizer update. Loss finite check đã chặn cập nhật.
- Không bật AMP mặc định. AMP tùy chọn dùng BF16 mặc định, kiểm tra hỗ trợ GPU; bài test BF16 forward/backward/update/checkpoint đã PASS. `--amp-dtype float16` vẫn chỉ là tùy chọn pilot, không bảo đảm ổn định trên mọi GPU.
- FP32 CPU regression và các smoke test đã PASS. Đây không phải bằng chứng hết mọi lỗi hoặc đã đạt chất lượng tối ưu.

## Điều kiện chưa giải quyết bằng code

- Chưa có nhãn XYZ WiPose thật đã xác minh. Không tự suy ra Z từ confidence.
- Chưa đánh giá nhiều seed hoặc tối ưu siêu tham số thật.
- Native WiPose chưa có sequence/time metadata, nên chưa khẳng định sequence-disjoint split hay velocity metric trên native frame.
- Backbone gốc còn tham số đăng ký không được gọi; giữ để bảo toàn baseline. WiPose Params vẫn trên 2.5M.
- Pose-TCN là noncausal window. AMP/deterministic và resume không hứa kết quả bit-exact qua phần cứng/thư viện khác.
- Checkpoint v1, setting v1 và số đo v1 không dùng trực tiếp để kết luận so sánh v2.

## Bằng chứng kiểm thử

`validation/summary.json` và log có: sanity frame/sequence, train/eval/best/last checkpoint cho hai dataset tổng hợp; 13 regression test về cross-frame dependence, TCN T=5, cùng initialization backbone/decoder cho XYZ ablation, objective, loss zero tại GT, PA/degenerate torso, visibility, windows không vượt sequence, ZIP/folder equivalence, schema bool, exact mid-epoch resume và OS lock. Dữ liệu giả không đo độ chính xác nghiên cứu.

Smoke test được tách output MM-Fi/WiPose để có thể chạy lại nhiều lần mà không chọn nhầm thư mục kết quả cũ. Mặc định W=8 để nhánh dilation4 có frame lân cận thật; giữ cùng W/batch/stride cho các ablation và pilot VRAM trước run dài.
