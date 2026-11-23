# Bằng chứng kiểm thử WiPose v3

Ngày 23/11/2026. Kết quả dưới đây kiểm tra triển khai và quy trình, chưa xác nhận cải thiện accuracy trên đầy đủ WiPose.

| Kiểm tra | Môi trường | Kết quả |
|---|---|---|
| `test_wipose_ablation.py` | Python 3.12, PyTorch 2.7.0 CPU | 10/10 PASS |
| `test_wipose_paper.py` | Python 3.12, PyTorch 2.7.0 CPU | 8/8 PASS, gồm Windows workers và resume |
| `test_training_smoke.py` | PyTorch 2.7.0 CPU | PASS, gồm MM-Fi giả lập, XYZ và kinematic |
| Preflight và pilot GPU | PyTorch 2.13.0+cu126, GTX 1650 Ti 4 GB | Ba cấu hình chính hoàn tất |

Pilot GPU dùng tập trích nhỏ có sẵn gồm 512 file WiPose thật: 256 mẫu train pilot, 36 validation; 154 mẫu outer test không được đọc/đánh giá trong develop. Mỗi mô hình chạy 1 epoch, 8 bước batch 32. PCK còn bằng 0 sau pilot ngắn này; không lấy điểm này so với lượt 50 epoch hoặc để chọn mô hình tốt nhất.

Pilot local nằm ở `runs_wipose_v3/real_validation/pilot_develop_20261123_212755_671767/`; checkpoint/predictions/dataset không đưa lên Git. Log đầy đủ có ở [real_gpu_pilot.log](real_gpu_pilot.log). Code snapshot và các file của suite giữ trên máy local. Chưa trực tiếp chạy trên GPU RTX 3060 của máy bên Đài; khi tải code, chạy preflight/pilot trên chính máy đó.

Kiểm thử xác nhận attention đều và attention có trọng số trên cả ba kiểu pooling giữ đúng vị trí tensor; khởi tạo gate=0 khớp Baseline cả train/eval; gate và nhánh TCN học sau các bước cập nhật; split disjoint; cache đúng byte tensor; không đọc test trong develop; resume khớp từng tensor trên CPU; từ chối đổi code/kiến trúc khi resume; giữ nguyên checkpoint các run đã hoàn thành.

Lỗi môi trường và fixture ban đầu được giữ trong các log cũ và [ERROR_LOG.md](../../ERROR_LOG.md). Các log kết thúc `_final.log` và `_regression.log` là lượt xác nhận sau khi xử lý lỗi fixture.

[complexity.json](complexity.json) ghi Params và THOP MACs. Baseline đăng ký 6.174.484 tham số, trong đó 3.315.092 tham số có kết nối tới loss ở phép kiểm tra. Phần chênh lệch là các module kế thừa không dùng trong forward. Bản `tcn_gated` thêm 8.769 tham số; THOP MACs tăng từ 58.419.072 lên 59.705.472 (khoảng 2,2%). FLOPs trong JSON là 2 × MACs và chưa bao phủ đầy đủ functional ops; không coi đây là số FLOPs chính xác tuyệt đối hoặc phép đo latency trên RTX 3060.
