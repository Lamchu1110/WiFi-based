# WiPose v4: kiểm tra khả năng học của TCN

Nhánh này tiếp nối chẩn đoán checkpoint ngày 25/11/2026: scalar gate gần 0 làm nhánh TCN cũ cập nhật rất ít. V4 thêm **ShortResidualTCN**, dùng lớp chiếu đầu ra khởi tạo bằng 0 và không có scalar gate. Mục tiêu là kiểm tra khả năng học của nhánh; chưa có kết quả full-data chứng minh vượt HPE-Li.

Baseline HPE-Li giữ nguyên kiến trúc và phép tính forward. Mã DSK, pooling, decoder và các ID lịch sử không bị thay thế. DSK corrected là biến thể của chúng ta, không phải baseline của tác giả.

## Năm cấu hình đối chứng

| ID | DSK | TCN | Vai trò |
|---|---|---|---|
| `baseline_legacy` | Gốc | Không | Đối chứng chính HPE-Li |
| `baseline_corrected` | Đã sửa | Không | Ablation DSK của chúng ta |
| `tcn_gated` | Đã sửa | Scalar gate khởi tạo 0 | Mốc TCN v3 |
| `tcn_residual_legacy` | Gốc | Projection khởi tạo 0, không scalar gate | Tách đóng góp TCN mới |
| `tcn_residual_corrected` | Đã sửa | Projection khởi tạo 0, không scalar gate | Ours kết hợp DSK + TCN mới |

Các model có cùng khởi tạo backbone theo seed. TCN mới ban đầu cho đầu ra giống bản không TCN **cùng DSK**. Projection nhận gradient ở bước đầu; trunk temporal nhận gradient sau khi projection khác 0. Điều này cần được theo dõi thực tế, không bảo đảm cải thiện độ chính xác.

## Cài đặt và kiểm tra

Dùng môi trường PyTorch CUDA đang chạy được trên máy train. Các phụ thuộc nằm trong `requirements-wipose-trial.txt`; không cài lại torch CPU đè lên môi trường GPU. Kiểm tra:

```powershell
python -m pip install -r requirements-wipose-trial.txt
python verify_package.py
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Mỗi clone mới có thư mục riêng, không giải nén chồng v4 lên v3. Có thể lấy nhánh này bằng:

```powershell
git clone --branch experiment/wipose-tcn-trainability-v4 https://github.com/Lamchu1110/WiFi-based.git WiFi-based-v4
cd WiFi-based-v4
```

## Chạy trên máy bên Đài: dữ liệu E:\Wi-Pose

Chạy lần lượt trong thư mục chứa `run_wipose_ablation.py`:

```powershell
python run_wipose_ablation.py --dataset-root "E:\Wi-Pose" --stage develop --mode preflight --cache off
python run_wipose_ablation.py --dataset-root "E:\Wi-Pose" --stage develop --mode pilot --epochs 1
python run_wipose_ablation.py --dataset-root "E:\Wi-Pose" --stage develop --mode train --epochs 10 --seeds 0
```

Mặc định chạy cả năm cấu hình trên cùng split. Recipe giữ SGD lr=0.001, momentum=0.9, weight_decay=0, batch=32, MSE XY, FP32, lịch lr constant. Các lựa chọn optimizer khác chỉ là ablation riêng, không được đổi riêng cho một model rồi quy toàn bộ chênh lệch cho kiến trúc.

`develop` giữ outer 70/30 seed 41; tách 10% của phần train làm validation với seed 17, không đọc tensor test ngay cả khi tạo cache. Với bộ 166.600 mẫu: 104.958 train / 11.662 validation / 49.980 test giữ riêng. Mốc 10 epoch là vòng development chọn phương pháp, không phải thay thế bảng final 50 epoch. Pilot chỉ là kiểm tra ngắn.

Nếu còn `split.json` của suite final trước, thêm cùng tùy chọn `--split-file "DUONG_DAN_THAT_TOI_SPLIT.json"` ở cả ba lệnh. File phải là full split không có validation/pilot, với ID khớp toàn bộ dữ liệu; không lấy split của 512 mẫu local. Không có file này thì dùng phép chia mặc định, và cần đối chiếu danh sách ID với lượt cũ trước khi ghép kết quả.

Test cũ đã được phân tích để định hướng nghiên cứu; không dùng nó để dò gate/loss tiếp. Chốt cấu hình bằng development rồi mới tạo suite `--stage final --mode train --epochs 50` với các biến thể đã chốt. Nếu muốn tăng độ tin cậy, chạy development nhiều seed (`--seeds 0 1 2`), giữ cả kết quả kém. Đánh giá final dùng checkpoint cuối; development chọn checkpoint theo validation MPJPE. Hai loại điểm phải báo cáo riêng.

## Đọc log và gửi kết quả

Mỗi lượt có thư mục riêng trong `runs_wipose_v4/`, giữ command, môi trường, source snapshot/hash/diff, split, config, metrics, checkpoint và trạng thái. Gửi lại **toàn bộ suite**, không chỉ comparison.csv.

- `comparison.csv/json`: metric từng model và seed; JSON ghi các cặp đối chứng và độ lệch chuẩn khi có nhiều seed.
- `seed_0/<variant>/trainability.jsonl`: gradient sau clipping, cập nhật trọng số thực tế sau optimizer, norm nhánh so với khởi tạo và tỷ lệ residual/input trước cập nhật. Mặc định lấy mẫu ở 5 bước đầu, mỗi 100 bước và cuối epoch; không thêm forward.
- Projection có norm khởi tạo 0 nên tỷ lệ thay đổi tương đối là `null`; đọc `change_norm` và `changed_elements`. Không coi `null` là lỗi học.
- Mỗi lần resume có `attempt` riêng trong log chẩn đoán. Nếu crash và chạy lại từ checkpoint, bước trùng thuộc lần thử khác; không cộng thành nhiều bước train độc lập.
- `metrics.csv`, `train.log`, `performance.jsonl`, `last.pt` và `best.pt`/`final.pt` vẫn được giữ như trước. Không có trainability log cho baseline không TCN.

```powershell
python run_wipose_ablation.py --resume-suite "runs_wipose_v4\TEN_SUITE_DANG_CHAY"
```

Chỉ resume bằng đúng source/config của suite. V4 từ chối resume suite v3; dùng nhánh cũ để tiếp tục v3. Không sửa kiến trúc giữa chừng rồi nạp optimizer cũ để gọi là cùng thí nghiệm.

## Kiểm thử và giới hạn

```powershell
python test_temporal_trainability.py
python test_wipose_ablation.py
```

`probe_trainability.py` là công cụ chẩn đoán SGD ngắn dùng đúng ID train từ split đã lưu. Nó không tính điểm validation/test và không thay thế ablation full-data. Checkpoint của probe không dùng để resume runner chính.

WiPose ở đây là đầu vào [B,9,30,5], đầu ra [B,18,2]; confidence không phải Z. TCN xử lý năm mẫu CSI trong một đầu vào, không phải năm pose liên tiếp. Loss vẫn plain MSE; chưa thêm bone/smooth/sym hay loss robust. PCK dùng ngưỡng theo kích thước thân người, không phải 20/50 pixel cố định.

Recipe chung này khác script tác giả phát hành (AdamW, split theo thư mục và confidence-weighted loss). Vì vậy gọi baseline là **kiến trúc HPE-Li gốc dưới protocol chung**, không gọi đây là tái lập toàn bộ pipeline tác giả. Split random-frame cũng chưa kiểm tra tổng quát sang người/sequence mới.
