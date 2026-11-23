# ST-DSKPose WiPose v3

Nhánh `experiment/wipose-dsk-tcn-v3` bổ sung DSK giữ đúng vị trí kênh/tần số và một TCN nhỏ có gate, cùng các đối chứng để đo đóng góp từng phần. Đây là phiên bản thực nghiệm; kiểm thử đúng tensor/gradient không bảo đảm độ chính xác cao hơn Baseline.

Code v2 và kết quả cũ giữ ở nhánh `experiment/wipose-paper-protocol`. [Hướng dẫn v2 được giữ lại](docs/WIPOSE_PAPER_V2.md). Không resume checkpoint v2 bằng trainer v3.

## Các cấu hình

| Tên | DSK | Pooling | TCN |
|---|---|---|---|
| `baseline_legacy` | Cách bố trí trục kế thừa | Gốc `(2,2)` | Không |
| `baseline_corrected` | Giữ vị trí `[B,M,C,F,T]` | Gốc `(2,2)` | Không |
| `tcn_gated` | Đã sửa | Gốc `(2,2)` | Trước pooling đầu; dilation 1/2, gate học được |
| `pooling_legacy` | Kế thừa | Giữ T, `(2,1)` | Không |
| `tcn_legacy` | Kế thừa | Giữ T, `(2,1)` | TCN cũ cuối backbone |
| `pooling_corrected` | Đã sửa | Giữ T, `(2,1)` | Không |
| `tcn_corrected` | Đã sửa | Giữ T, `(2,1)` | TCN cũ cuối backbone |

Mặc định chạy ba dòng đầu. So `baseline_corrected` với `baseline_legacy` để đo sửa DSK; so `tcn_gated` với `baseline_corrected` để đo TCN mới. So `tcn_legacy` với `pooling_legacy` để tách TCN cũ khỏi đổi pooling. Mọi cấu hình trong một suite dùng cùng split và recipe.

Input WiPose là `[B,9,30,5]`, output `[B,18,2]`. Năm mẫu CSI nằm trong một frame ảnh, không phải năm pose liên tiếp. Loss trong thí nghiệm này giữ plain MSE XY; chưa thêm loss cấu trúc 2D. Confidence trong nhãn không phải tọa độ Z. Không bật `direct_3d` hoặc loss XYZ cho dữ liệu native WiPose.

## Tải code và kiểm tra môi trường

Mở terminal trên máy train, giữ môi trường PyTorch đã nhận RTX 3060:

```powershell
git clone --branch experiment/wipose-dsk-tcn-v3 https://github.com/Lamchu1110/WiFi-based.git WiFi-based-v3
cd WiFi-based-v3
python -m pip install -r requirements-wipose-trial.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python verify_package.py
```

Requirements không tự thay PyTorch/torchvision. Python đang chạy phải có `h5py` và `mat73` để đọc MATLAB 7.3, ngoài NumPy/SciPy và các phụ thuộc trong requirements. Có thể chạy `python test_wipose_ablation.py` để kiểm tra DSK, gradient, split, cache, resume và lưu code trên dữ liệu giả.

## Chạy ngay trên WiPose

Giả sử dữ liệu đã giải nén có `E:\Wi-Pose\Train` và `E:\Wi-Pose\Test`. Chạy từng lệnh riêng, chờ lệnh trước hoàn tất:

```powershell
python run_wipose_ablation.py --dataset-root "E:\Wi-Pose" --mode preflight --cache off
python run_wipose_ablation.py --dataset-root "E:\Wi-Pose" --mode pilot
python run_wipose_ablation.py --dataset-root "E:\Wi-Pose" --mode train --stage develop --epochs 10
```

`preflight` kiểm tra dữ liệu/cấu hình. `pilot` chạy 1 epoch trên tập nhỏ để phát hiện lỗi; điểm pilot không dùng làm kết quả bài báo. Lệnh cuối chạy ba cấu hình chính để sàng lọc bằng validation, chưa đánh giá test. Nếu cần 30/50 epoch để chọn kiến trúc, bắt đầu suite develop mới có ngân sách đó từ đầu; resume không cho đổi tổng epoch hoặc recipe giữa chừng.

Mặc định: SGD momentum 0,9, lr 0,001 cố định, batch 32, weight decay 0, FP32, TF32 tắt, model seed 0. Có thể thử `--optimizer adamw --lr 0.0003 --weight-decay 0.0001 --schedule cosine` trong suite develop mới; cùng recipe áp dụng cho tất cả cấu hình được chọn, không tự động tuyên bố đây là tối ưu.

## Validation và đánh giá cuối

`develop` giữ outer split 70/30, seed 41; lấy 10% của phần 70% làm validation với seed riêng 17. Với 166.600 mẫu sẽ có 104.958 train, 11.662 validation, 49.980 test giữ lại. Cache và huấn luyện develop không đọc tensor của test. Chọn checkpoint bằng validation MPJPE, không theo test.

`final` dùng toàn bộ 70% để train, không có validation, đánh giá checkpoint cuối. Sau khi đã chọn kiến trúc bằng develop, ví dụ xác nhận Baseline sửa DSK và TCN mới:

```powershell
python run_wipose_ablation.py --dataset-root "E:\Wi-Pose" --mode train --stage final --epochs 50 --variants baseline_corrected tcn_gated --seeds 0 1 2
```

Lệnh trên chạy sáu lượt train tuần tự. Nếu ngân sách chưa đủ có thể bắt đầu `--seeds 0`; chưa có nhiều seed thì chưa báo cáo độ lệch chuẩn như đã đo. Nếu đã chọn recipe khác trên validation, truyền chính recipe đó cho cả hai mô hình ở lệnh final.

Có thể truyền `--split-file "duong_dan\split.json"` để dùng danh sách outer train/test đã lưu. File phải chứa đầy đủ mẫu đúng dataset, không có overlap, validation hoặc pilot subset. Mặc định v3 dùng cách sắp xếp tên file và outer seed như nhánh paper v2. Đây là random-frame split, không phải đánh giá cross-subject/cross-sequence và không chứng minh tái lập đúng bảng số của bài báo.

## Log và bằng chứng cho ablation

Mỗi suite có thư mục mới trong `runs_wipose_v3/`, với các run `seed_0/baseline_corrected/`, `seed_0/tcn_gated/`… Không ghi đè suite cũ. Thư mục gồm:

- `source_snapshot.zip`, `source_manifest.json`, `source.diff`: code thực tế, SHA256, commit và thay đổi chưa commit.
- `command.json`, `suite_config.json`, `protocol.json`, `environment.json`, `split.json`: lệnh, cấu hình, môi trường và danh sách mẫu.
- Mỗi run: `train.log`, `config.json`, `metrics.csv`, `performance.jsonl`, `status.json`, `last.pt` và `best.pt` hoặc `final.pt`.
- Develop: `validation_metrics.json`, `validation_predictions.npz`. Final: `test_metrics.json`, `test_predictions.npz`. NPZ chứa prediction XY và target XY/confidence theo thứ tự ID trong split (dùng mục pilot nếu là pilot).
- `comparison.csv/json`: điểm từng seed; JSON thêm trung bình/độ lệch chuẩn mẫu và chênh lệch cho các cặp ablation hợp lệ. Độ lệch chuẩn là `null` khi chỉ có một seed.
- `tcn_alpha` được ghi mỗi epoch cho TCN mới để xem nhánh có thực sự học hay không.

Lưu cả thư mục suite, không chỉ file CSV. Giữ cả kết quả kém và các run thất bại. Source snapshot không chứa dataset, checkpoint hay file `.env`.

## Resume và tốc độ

Khi bị gián đoạn, dùng đường dẫn suite được in trên terminal:

```powershell
python run_wipose_ablation.py --resume-suite "runs_wipose_v3\TEN_SUITE_DANG_CHAY"
```

Resume giữ nguyên split, seed, optimizer, scheduler, mục tiêu, kiến trúc và code. Chỉ cho đổi `--device`/`--workers`. Nếu code đã sửa, khôi phục code của run hoặc tạo suite mới; không gộp kết quả hai kiến trúc. Những run đã hoàn thành được giữ nguyên.

Cache mmap tránh đọc MATLAB lại ở mỗi epoch. Mặc định `--workers 0`; có thể đo thử 2 hoặc 4 trên máy của bạn. Batch mặc định vẫn 32. Kiểm tra loss/gradient mỗi bước; kiểm tra tham số mỗi 100 bước và trước mọi checkpoint để giảm chi phí kiểm tra lặp lại. Có thể đặt `--parameter-check-every 1` nếu cần chẩn đoán sát từng bước. `--precision bf16` chỉ dùng khi GPU hỗ trợ và đã thử pilot, với cùng precision cho hai mô hình.

Thay precision, optimizer hoặc số epoch là một điều kiện thí nghiệm cần ghi rõ. Không có cam kết rằng thay đổi này sẽ cho kết quả tốt nhất.
