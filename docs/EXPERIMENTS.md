> Historical V1 documentation. For current behavior see [training v2](README_V2.md).

# MM-Fi ablation

Chạy từ thư mục chứa `run_experiments.py`:

```powershell
# Cả ba kịch bản chạy tuần tự
python run_experiments.py --dataset-zip D:\Downloads\MMFi_Dataset.zip --scenario all

# Chạy riêng một kịch bản
python run_experiments.py --dataset-zip D:\Downloads\MMFi_Dataset.zip --scenario baseline
python run_experiments.py --dataset-zip D:\Downloads\MMFi_Dataset.zip --scenario tcn
python run_experiments.py --dataset-zip D:\Downloads\MMFi_Dataset.zip --scenario full

# Nếu đã giải nén, dùng --dataset-root thay cho --dataset-zip
python run_experiments.py --dataset-root D:\Dataset\MMFi_Dataset --scenario baseline --preflight-only
```

Mặc định: 50 epochs, batch 32, SGD lr 0.001 momentum 0.9, seed 0. Giữ lịch lr gốc: 20 epochs đầu hằng số rồi giảm tuyến tính. Giữ protocol3/random_split theo YAML, tách heldout thành validation/test bằng random_state=41. Baseline dùng `OriginalHPE` như `main.py`; `att_mmfi.py` dùng một biến thể Transformer khác. Các thông số người dùng yêu cầu được ưu tiên (mã cũ còn có các giá trị khác như 60 epochs, momentum mặc định 0).

`baseline`: TCN=False, direct_3d=False, kinematic=False.
`tcn`: TCN=True, direct_3d=False, kinematic=False.
`full`: TCN=True, direct_3d=True, kinematic=True.

Mỗi lần chạy tạo thư mục `runs/<thời gian>/<kịch bản>/`:

- `train.log`: stdout, traceback, tiến độ mỗi 100 batch và metric mỗi epoch.
- `metrics.csv`: validation metric từng epoch, train_loss, lr; R.Hand index 13 và L.Hand index 16.
- `config.json`: cấu hình và quy ước metric/loss đầy đủ.
- `split_indices.npz`: chỉ số validation/test dùng lại giống nhau giữa các kịch bản.
- `best.pt`: checkpoint validation PCK20 tốt nhất; `last.pt`: checkpoint cuối cùng.
- `test_metrics.json`: test metric từ checkpoint tốt nhất sau khi đủ epochs.
- `status.json`: trạng thái hoàn thành hoặc lỗi.

Metric so sánh chung: MPJPE/PA-MPJPE trên XY (mm, giả định nhãn gốc tính bằng mét), PCK20/PCK50 là phần trăm điểm có sai số <=0.2/0.5 lần khoảng cách joint 1–11 theo hàm gốc; không phải ngưỡng 20/50 mm. Full có thêm MPJPE/PA-MPJPE XYZ.

Để bảo toàn hành vi baseline, baseline/TCN giữ loss gốc `MSE(Z * predXY, Z * gtXY) / 32`. Trong MM-Fi Z là tọa độ, dù mã cũ gọi biến này là confidence. Full dùng loss XYZ mới. Vì vậy đây là so sánh hệ thống hoàn chỉnh, không phải thí nghiệm chỉ thay TCN và loss trong cùng một objective 3D.

Full dùng window pose 4 frame trong cùng subject/action, không nối qua ranh giới sequence; baseline/TCN dùng frame đơn. Warm-up phụ trợ từ epoch 0 đến 10. Window chồng lấn thay đổi số mẫu/updates so với baseline; điều này được ghi vào config.

Đọc ZIP tránh giải nén 17.70 GB CSI/GT khi ổ C thiếu chỗ. Adapter chỉ đọc CSI/nhãn, chuẩn hóa như loader gốc, kiểm tra shape và chặn CSI hằng số/không hữu hạn. Kiểm tra đầu/cuối mỗi split trước train; dữ liệu còn lại được đọc và kiểm tra khi chạy, chưa CRC toàn bộ archive. Hỏng file giữa chừng sẽ ghi traceback và dừng.

Theo dõi tiến trình Baseline đã khởi chạy:

```powershell
Get-Content logs/baseline_launcher.stdout.log -Tail 20 -Wait
Get-Content logs/baseline_launcher.stderr.log -Tail 20
Get-Process -Id (Get-Content logs/baseline.pid)
```

`python test_training_smoke.py` kiểm tra train/eval/lưu checkpoint cả ba kịch bản bằng dữ liệu tổng hợp nhỏ; kết quả đặt trong `runs_smoke`, tách khỏi thực nghiệm thật.
