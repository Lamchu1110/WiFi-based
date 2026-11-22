# ST-DSKPose — WiPose với thông số công bố trong ECCV 2024

Nhánh `experiment/wipose-paper-protocol`, cập nhật 23/11/2026. Hai mô hình **Baseline HPEWiPoseModel** và **Baseline + CSI-TCN** được train từ đầu trong cùng điều kiện. Nhánh trước `experiment/wipose-baseline-tcn` lưu lại thử nghiệm AdamW 3 epoch.

## Cấu hình của lượt chính thức

| Thành phần | Giá trị |
|---|---|
| Epoch | **50** |
| Optimizer | **SGD, momentum 0.9** |
| Learning rate / batch train | **0.001 / 32** |
| Loss | MSE tọa độ XY, không confidence weighting và không chia thêm 32 |
| Split | Gộp các file từ Train/Test, chia ngẫu nhiên **70% train / 30% test** |
| Chọn checkpoint | **Epoch cuối**, theo lựa chọn của người dùng; không dùng test chọn model |
| Precision | FP32, TF32 tắt |
| Đầu ra | XY của 18 khớp; không dùng confidence làm Z |

Nguồn: [mục 4 và Bảng 1, trang 8–9 của bài báo](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04496.pdf). Đây là thiết lập theo **các thông số được công bố**, có những lựa chọn triển khai phải ghi rõ: seed chia tập 41, LR cố định vì paper không mô tả lịch giảm, weight decay 0, MSE reduction mean và giữ scale XY × 0.001 của loader. Paper không công bố file-list/seed cụ thể hoặc chính sách validation; không tuyên bố tái lập chính xác bảng kết quả.

**Metric của nhãn ảnh native là pixel**, không có căn cứ đổi thành mm vật lý. Xem [protocol và các khác biệt](docs/WIPOSE_PAPER_PROTOCOL.md). TCN gồm cả đổi pooling để giữ 5 mẫu CSI trong một frame; chưa phải pose-TCN hoặc mô hình XYZ/kinematic.

## Tải code và kiểm tra

Trên máy Windows bên Đài, mở CMD trong môi trường Python đã nhận RTX 3060:

```bat
git clone --branch experiment/wipose-paper-protocol --single-branch https://github.com/Lamchu1110/WiFi-based.git WiFi-based-paper
cd WiFi-based-paper
python -m pip install -r requirements-wipose-trial.txt
python verify_package.py
python test_wipose_paper.py
```

Dùng PyTorch CUDA đang hoạt động trên máy. Bộ test mới đã chạy bằng Python 3.12 / PyTorch 2.7 CPU; bằng chứng GPU cục bộ và giới hạn xem [validation/wipose_paper](validation/wipose_paper/README.md). Dataset không nằm trong repository.

## Chạy lần lượt

Thay đường dẫn ví dụ bằng nơi đã giải nén, chứa hai thư mục `Train` và `Test`.

Kiểm tra một số mẫu trước, chưa tạo cache toàn bộ:

```bat
python run_wipose_paper.py --dataset-root "D:\Datasets\Wi-Pose" --mode preflight --cache off
```

Pilot 1 epoch trên tối đa 256 train / 128 test để kiểm tra luồng thực thi; cache chỉ tạo cho các mẫu pilot:

```bat
python run_wipose_paper.py --dataset-root "D:\Datasets\Wi-Pose" --mode pilot
```

Chạy **50 epoch mỗi mô hình**, Baseline rồi TCN:

```bat
python run_wipose_paper.py --dataset-root "D:\Datasets\Wi-Pose" --mode train
```

Lượt train đầu tiên tạo cache từ toàn bộ MAT: có thể mất nhiều phút, terminal in tiến độ mỗi 5.000 file. Các epoch và lượt chạy sau dùng lại cache. Với đủ 166.600 mẫu, split là **116.620 train / 49.980 test**, không có validation. Cache float32 CSI/nhãn khoảng 0.94 GB thập phân, chưa tính metadata; không cần đưa toàn bộ vào RAM/GPU.

Trong quá trình train, `metrics.csv` ghi loss/LR; **không có PCK/MPJPE test theo từng epoch**. Test chỉ chạy sau epoch cuối. Không so loss này trực tiếp với loss của nhánh AdamW vì công thức loss đã khác.

## Đọc kết quả và resume

`runs_wipose_paper/train_<timestamp>/` chứa:

- `protocol.json`, `suite_config.json`, `split.json`, `environment.json`: cấu hình, lựa chọn triển khai, danh sách mẫu và môi trường.
- `baseline/`, `tcn/`: `train.log`, `metrics.csv`, `last.pt`, `final.pt`, `test_metrics.json`, `performance.jsonl`, `test_timing.json`.
- `comparison.csv/json`: MPJPE, PA-MPJPE, PCK20/PCK50, PCK hai cổ tay, chênh lệch TCN − Baseline.
- `suite.log`, `status.json`: tiến độ và lỗi.

`final.pt` là checkpoint cuối, không gọi là checkpoint “tốt nhất” theo test. Mỗi model được đánh giá trên cùng test. Chỉ so hai dòng khi suite có trạng thái `complete`.

```bat
python run_wipose_paper.py --resume-suite "D:\WiFi-based-paper\runs_wipose_paper\train_TIMESTAMP"
```

Resume tự đọc cấu hình, không train lại model đã hoàn thành. Có thể thêm `--workers 2` hoặc `--device cuda`; không thay epoch/split/loss. **Không resume các checkpoint AdamW 3 epoch vào protocol mới.**

## Tăng tốc mà vẫn kiểm tra tính tương đương

- Cache mmap lưu đúng tensor float32 của loader gốc; có checksum và từ chối cache hỏng.
- DataLoader giữ thứ tự mẫu; hỗ trợ `--workers 2` hoặc `4` và persistent workers. Mặc định 0 vì cache đã rất nhẹ và Windows có chi phí spawn; đo trên máy thật trước khi chọn.
- Gom kiểm tra finite và chuyển số liệu từ GPU, giảm đồng bộ CPU/GPU; vẫn kiểm tra loss/gradient/parameter và lưu checkpoint atomic.
- `performance.jsonl` tách thời gian chờ dữ liệu, train và ghi checkpoint. Khi resume giữa epoch, dòng timing chỉ tính phần chạy từ vị trí resume.
- Giữ batch 32. Không tăng batch, đổi loss hoặc bật AMP để báo một con số tốc độ “cùng cấu hình”.
- `--precision bf16` là tùy chọn thử nghiệm nếu GPU hỗ trợ; mặc định FP32. BF16 được ghi là biến thể số học, cần kiểm tra riêng trước khi dùng.
- `--deterministic` giảm nguồn ngẫu nhiên trên cùng môi trường, có thể chậm hơn; không bảo đảm bit-exact giữa thiết bị/phiên bản khác nhau.

Đo riêng đọc dữ liệu, không bao gồm GPU train, với cùng các mẫu:

```bat
python benchmark_wipose_io.py --dataset-root "D:\Datasets\Wi-Pose"
```

Benchmark kiểm tra tensor raw/cache bằng nhau trước khi đo; kết quả ở `logs/wipose_io_benchmark.json`. Mức tăng tốc đọc dữ liệu không đồng nghĩa mức tăng tốc toàn bộ train.

Nhánh cũ vẫn có script `run_wipose_comparison.py`, nhưng script đó dùng **AdamW / batch 64 / split theo archive**, không phải lệnh cho protocol mới. Chi tiết lịch sử ở [WIPOSE_TRIAL.md](docs/WIPOSE_TRIAL.md).
