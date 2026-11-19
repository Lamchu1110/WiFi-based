# Kiểm chứng nhánh WiPose — 19/11/2026

Các file trong thư mục này là bằng chứng vận hành, không phải kết quả benchmark hoặc mô hình đã train hội tụ.

| Kiểm tra | Môi trường / dữ liệu | Kết quả |
|---|---|---|
| `test_wipose_trial.py` | Python 3.12.8, PyTorch 2.7.0+cpu, Windows; MAT tổng hợp v5/v7.3 | 8 test PASS |
| `test_reliability.py` | PyTorch 2.13.0+cu126, chạy CPU | 13 test PASS |
| `test_wipose_smoke.py` | PyTorch 2.13.0+cu126, CPU; MAT/XYZ-NPZ tổng hợp | PASS |
| `test_training_smoke.py` | PyTorch 2.13.0+cu126, CPU; MM-Fi tổng hợp | PASS |
| Pilot WiPose thật | PyTorch 2.13.0+cu126, GTX 1650 Ti 4 GB, FP32 | Baseline và CSI-TCN complete |

Kiểm thử PyTorch 2.7 ở đây dùng **CPU**, không phải build CUDA 11.8 hay GPU RTX 3060 bên Đài. Chưa chạy trực tiếp trên máy đó. Khi tải nhánh về, vẫn cần chạy pilot trên máy train.

## Pilot dữ liệu thật

Đã đọc inventory archive `Wi-Pose.rar` cục bộ: 132847 MAT Train và 33753 MAT Test. Chọn ngẫu nhiên cố định 256 Train + 256 Test, seed 20261119, giải nén các file này. Tập 256 Test được tách 128 validation / 128 test với seed 41. File đã chọn được ghi ở `real_probe_members.txt`; không đóng gói dữ liệu.

Lệnh đã chạy tại repository:

```text
python run_wipose_comparison.py --dataset-root data/real_probe/Wi-Pose --mode pilot --device cuda --batch-size 64 --output runs_wipose_trial/real_data_validation
```

Mỗi model: 1 epoch, **4 update**, batch 64; AdamW lr=0.001, objective upstream, FP32. Mỗi model hoàn thành forward/backward, validation, lưu best/last checkpoint và đánh giá test. Xem `real_pilot/suite.log`, `comparison.csv`, `comparison.json`, `environment.json`.

Các số MPJPE/PCK ở đây phản ánh mô hình chỉ mới học 4 batch. Không lấy chênh lệch rất nhỏ giữa hai model để kết luận cải tiến tốt hơn; không so các số pilot với bài báo. Cần train đủ dữ liệu và ngân sách, rồi đánh giá cùng protocol.

## Phạm vi 8 test mới

- Axis/normalization và nhãn giống loader upstream, cho cả MAT v5 và v7.3.
- Schema `csi_serial/jointsVector`, trường hợp flat vector cần khai báo thứ tự.
- Từ chối nhầm trục CSI và confidence âm.
- Split/pilot lặp lại được, không trùng file validation/test.
- Metric native trả đơn vị pixel.
- Baseline cho output đúng bằng `HPEWiPoseModel`; backbone TCN có cùng khởi tạo.
- AdamW resume giữa epoch cho trọng số/scheduler đúng bằng chạy liên tục trên CPU.
- Suite hoàn thành cả hai model, xuất bảng, resume không train lại model đã xong; từ chối dữ liệu thay đổi và từ chối ghép contract khác nhau.

## Giới hạn

Không huấn luyện 50 epoch trong lượt chuẩn bị nhánh; chưa có kết quả hội tụ, nhiều seed hoặc tái lập bảng bài báo. Các kiểm thử hồi quy dùng dữ liệu tổng hợp. Dữ liệu thật mới kiểm tra trên 512 file đã chọn, không phải quét nội dung toàn bộ archive. Checkpoint, dataset và môi trường Python không được commit.
