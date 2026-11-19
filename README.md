# ST-DSKPose — thử nghiệm WiPose: Baseline và TCN

Nhánh `experiment/wipose-baseline-tcn` chuẩn bị riêng cho việc train WiPose trên máy Windows có GPU. Nhánh này kế thừa `main` tại commit `3e7e2c8`.

Hai mô hình được train lần lượt từ đầu, cùng dữ liệu, seed, hàm loss, optimizer và cách chọn checkpoint:

- **Baseline:** backbone `HPEWiPoseModel` từ mã nguồn HPE-Li, đầu ra XY.
- **TCN:** cùng backbone/decoder, giữ 5 mẫu thời gian CSI trước khối TCN đa tỉ lệ, đầu ra XY.

WiPose công khai có 18 khớp **XY + confidence**. Thử nghiệm này đánh giá **CSI-TCN trong một frame**, chưa phải Full ST-HPE với XYZ và kinematic loss, và chưa dùng chuỗi pose giữa các file.

## Tải nhánh về máy train

Chạy trong CMD/Terminal tại nơi bạn muốn đặt code:

```bat
git clone --branch experiment/wipose-baseline-tcn --single-branch https://github.com/Lamchu1110/WiFi-based.git WiFi-based-wipose
cd WiFi-based-wipose
```

Sử dụng môi trường Python đã kiểm tra nhận GPU. Máy bên Đài hiện đã báo PyTorch 2.7.0+cu118 và RTX 3060. Cài thêm phụ thuộc rồi kiểm tra lại:

```bat
python -m pip install -r requirements-wipose-trial.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python verify_package.py
python test_wipose_trial.py
```

`test_wipose_trial.py` chạy kiểm thử CPU và dữ liệu tổng hợp, chưa cần dataset. Bằng chứng riêng của nhánh ở [validation/wipose_trial](validation/wipose_trial/README.md). Python 3.12 là phiên bản được dùng trên máy kiểm thử.

## Chuẩn bị dữ liệu

Giải nén `Wi-Pose.rar`, ví dụ:

```text
D:\Datasets\Wi-Pose\
  Train\
    bend_001-frame001.mat
    ...
  Test\
    bend_009-frame001.mat
    ...
```

Thay đường dẫn ví dụ bên dưới bằng thư mục thật trên máy của bạn. Loader hỗ trợ MAT thường và MATLAB 7.3/HDF5. Hai schema `CSI/SkeletonPoints` và `csi_serial/jointsVector` đều được hỗ trợ; vector phẳng có thứ tự không rõ ràng sẽ yêu cầu khai báo thay vì tự đoán.

## Chạy theo thứ tự

**1. Kiểm tra cấu trúc và một số mẫu dữ liệu:**

```bat
python run_wipose_comparison.py --dataset-root "D:\Datasets\Wi-Pose" --mode preflight
```

**2. Chạy pilot trên tập nhỏ để kiểm tra GPU, forward/backward, đánh giá và checkpoint:**

```bat
python run_wipose_comparison.py --dataset-root "D:\Datasets\Wi-Pose" --mode pilot
```

Mặc định: 1 epoch, tối đa 256 mẫu train + 128 validation + 128 test, batch 64, FP32. Cả hai mô hình dùng đúng cùng các mẫu. PCK thấp hoặc bằng 0 ở lượt ngắn này không phải bằng chứng chất lượng cuối cùng.

**3. Train thử 3 epoch trên toàn bộ dữ liệu của protocol đã chọn:**

```bat
python run_wipose_comparison.py --dataset-root "D:\Datasets\Wi-Pose" --mode train --epochs 3
```

Sau khi kiểm tra ổn định, bắt đầu một suite mới cho 50 epoch:

```bat
python run_wipose_comparison.py --dataset-root "D:\Datasets\Wi-Pose" --mode train --epochs 50
```

Nếu thiếu VRAM, dùng cùng `--batch-size 32 --eval-batch-size 16` cho cả cặp mô hình trong một suite mới. Ghi rõ thay đổi batch so với cấu hình 64 của script WiPose nguồn.

## Kết quả và tiếp tục khi bị gián đoạn

Mỗi lệnh tạo một thư mục riêng trong `runs_wipose_trial/`:

```text
train_<timestamp>/
  suite_config.json, environment.json, suite.log, split.json, status.json
  comparison.csv, comparison.json
  baseline/
    config.json, train.log, metrics.csv, best.pt, last.pt, test_metrics.json
  tcn/
    config.json, train.log, metrics.csv, best.pt, last.pt, test_metrics.json
```

- `comparison.csv`: MPJPE, PA-MPJPE, PCK20/PCK50 và PCK của hai cổ tay, lấy từ **best validation checkpoint** của từng mô hình trên cùng tập test.
- Với nhãn ảnh native: **MPJPE/PA-MPJPE là pixel**, không phải khoảng cách vật lý mm. Loader giữ phép nhân XY × 0.001 của upstream để train; metric đổi lại pixel.
- PCK20/PCK50 là phần trăm khớp trong ngưỡng 0.2/0.5 chiều dài torso, không phải ngưỡng 20/50 pixel.
- `comparison.json` ghi thêm chênh lệch TCN trừ Baseline và trạng thái complete/partial.
- Preflight chỉ tạo log/metadata kiểm tra; không có metric train/test hay checkpoint model.

Tiếp tục đúng suite đã chạy, thay đường dẫn bằng thư mục thực tế:

```bat
python run_wipose_comparison.py --resume-suite "D:\WiFi-based-wipose\runs_wipose_trial\train_TIMESTAMP"
```

Lệnh tự đọc cấu hình cũ, giữ kết quả mô hình đã hoàn thành và tiếp tục mô hình còn lại. Chỉ có thể thêm `--device` khi cần đổi thiết bị. Đổi epochs, batch, objective hoặc dữ liệu phải bắt đầu suite mới. Không resume checkpoint main/v2 cũ vào nhánh này.

## Phạm vi so sánh

Recipe mặc định theo `att_wipose.py`: AdamW lr=0.001, weight_decay=0.01, batch=64, loss `MSE(conf*pred, conf*target)/32`, chọn best theo PCK20. Dùng FP32 và tắt clipping để giữ gần recipe nguồn. Có `--objective normalized` nếu muốn so cả hai mô hình bằng confidence-weighted MSE chuẩn hóa; đây là thí nghiệm khác.

Đây là **so sánh có kiểm soát với backbone từ repository**, không tuyên bố đã tái lập chính xác bảng kết quả ECCV. Chia Test thành validation/test 50/50 theo seed 41 như script nguồn; sắp xếp filename ổn định; metric được tính lại mỗi epoch và có kiểm tra visibility/pose suy biến. Việc chia theo file chưa bảo đảm tách sequence/subject. TCN đồng thời thay cách pooling để giữ T=5, nên kết quả là đóng góp của cả thay đổi này và khối TCN.

Xem [chi tiết recipe và những khác biệt](docs/WIPOSE_TRIAL.md), [nguồn tác giả](docs/UPSTREAM_README.md), và [hướng dẫn v2 chung](docs/README_V2.md). Các hướng dẫn chung v2 không thay thế đơn vị và recipe riêng của nhánh thử WiPose này.
