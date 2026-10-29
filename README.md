# ST-DSKPose

**Spatio-Temporal Dual Selective Kernel Network for WiFi-based Human Pose Estimation**

Tên tạm thời của dự án nâng cấp từ HPE-Li. Gói này kế thừa implementation ST-HPE đã kiểm thử; chưa có kết luận mô hình mới cải thiện độ chính xác trên dữ liệu thật.

Mã nguồn độc lập cho thực nghiệm MM-Fi và WiPose, đóng gói ngày 29/10/2026. Source nền tảng được ghi nhận trong [README của tác giả](docs/UPSTREAM_README.md). Thư mục này chứa đủ source cần cho các entrypoint bên dưới; không cần cài/giữ repo cũ cạnh nó. Đây là snapshot của kiến trúc đã triển khai, không phải một kiến trúc temporal mới được sửa âm thầm khi đóng gói.

**Khả năng hiện có:** MM-Fi: baseline, TCN và Full XYZ; WiPose MAT theo loader gốc: baseline và TCN XY; WiPose Full XYZ: chỉ chạy khi cung cấp XYZ thật qua schema NPZ. Cột confidence của WiPose không được dùng làm Z.

## 1. Cài đặt trên máy mới

Đã kiểm thử bằng Python 3.12; có thể tạo môi trường độc lập:

```powershell
git clone https://github.com/Lamchu1110/WiFi-based.git
cd WiFi-based
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS dùng: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -m pip install -r requirements-wipose.txt
```

Lệnh torch/torchvision trên dùng package index mặc định. Nếu máy cần bản CUDA theo driver/GPU, cài cặp torch/torchvision phù hợp với môi trường máy đó trước khi cài requirements. Không chép `.venv` hay thư viện CUDA từ máy cũ. `requirements-wipose.txt` chỉ cần cho MAT v7.3/HDF5.

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Phiên bản thư viện đã test ở máy nguồn được lưu tại `validation/environment_reference.json` để đối chiếu, không phải lockfile bảo đảm mọi OS/GPU. Không ép cài build CUDA máy cũ lên máy mới.

## 2. Kiểm tra sau khi chuyển máy

```powershell
# Dùng standard library, chưa cần torch:
python verify_package.py

# Kiểm tra forward/backward, loss, Params/MACs:
python test_sanity.py

# Train/eval/checkpoint trên dữ liệu giả, CPU, không tải dataset:
python validate_portable.py
```

`validate_portable.py` chạy lại ba bài test, ghi `validation/summary.json` và log riêng. Các test tạo checkpoint giả trong `runs_smoke`; không dùng metric giả làm kết quả nghiên cứu. Kiểm tra checksum trước khi sửa source; sau khi bạn chỉnh source, việc checksum báo Changed là dự kiến.

## 3. MM-Fi thật

Nếu đã giải nén, thư mục gốc cần dạng:

```text
MMFi_Dataset/
  E01/S01/A01/ground_truth.npy
  E01/S01/A01/wifi-csi/frame001.mat
  ... E01–E04, S01–S40, A01–A27 ...
```

CSI `CSIamp`: `[3,114,10]`; GT mỗi action `[297,17,3]`. YAML chia tập ở `dataset_lib/config.yaml` đi kèm snapshot, không có đường dẫn dataset cũ. Loader này vẫn giả định297frame/action và đầy đủ split theo YAML; dataset giảm bớt hoặc cấu trúc khác cần sửa config/loader tương ứng.

```powershell
# Thay /path/to bằng đường dẫn thật trên máy mới. Windows dùng D:/Data/...
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario baseline --preflight-only
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario baseline
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario tcn
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario full

# Hoặc tự chạy cả ba tuần tự:
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario all

# Nếu chỉ có ZIP, không cần giải nén:
python train_mmfi.py --dataset-zip /path/to/MMFi_Dataset.zip --scenario all
```

Mặc định50epochs, batch32, SGDM lr0.001/momentum0.9, seed0. Ghi config và split; giữ protocol3/random_split trong YAML, heldout chia validation/test random_state41. Chạy `python train_mmfi.py --help` để thay setting. Đọc ZIP ngẫu nhiên có thể chậm; nếu máy mới đủ dung lượng, có thể dùng dữ liệu giải nén. Không tự động tải dataset.

## 4. WiPose MAT theo loader gốc: XY + confidence

```text
WiPose/
  Train/*.mat
  Test/*.mat
```

MAT có biến `CSI`, `SkeletonPoints`. Đảo trục CSI như repo gốc rồi reshape `[9,30,5]`, chuẩn hóa bằng mean/std9kênh gốc. `SkeletonPoints` reshape `[18,3]`; hai cột đầu mm→m, cột3 là confidence theo cách dùng trong repo. File được sort để split có thể lặp lại qua các máy. Test chia đôi validation/test seed41 như code cũ.

```powershell
python train_wipose.py --dataset-root /path/to/WiPose --scenario baseline --preflight-only
python train_wipose.py --dataset-root /path/to/WiPose --scenario baseline
python train_wipose.py --dataset-root /path/to/WiPose --scenario tcn
```

Hai lệnh này mặc định50epochs/batch32/SGDM theo setting đã yêu cầu cho gói, không tuyên bố đó là setting WiPose gốc tối ưu. Có `--device cpu` hoặc `--device cuda:0` và `--epochs`, `--batch-size`, `--lr` để thay.

**Giới hạn kiến trúc chưa sửa:** WiPose trước TCN chỉ cònT1 do pooling. TCN vẫn chạy nhưng chưa học tương tác nhiều thời điểm. Gói giữ snapshot này để không lẫn thay đổi kiến trúc với thao tác đóng gói; cần thí nghiệm sửa pooling/vị trí TCN trước khi kết luận temporal modeling hiệu quả trên WiPose.

## 5. WiPose Full 3D khi thực sự có nhãn XYZ

Không chạy Full bằng cách đổi tên confidence thành Z hoặc tự gán Z0. Phải có nhãn XYZ đúng nguồn và đúng thứ tự18joint. Đọc [DATA_FORMAT.md](docs/DATA_FORMAT.md) để chuẩn bị `train.npz`, `val.npz`, `test.npz`, gồm CSI đã tiền xử lý và window pose liên tiếp.

```powershell
python train_wipose.py --npz-dir /path/to/WiPose_XYZ_windows --scenario full --preflight-only
python train_wipose.py --npz-dir /path/to/WiPose_XYZ_windows --scenario full
```

Schema kiểm tra shape và khai báo định dạng; không thể tự xác thực rằng nhãn thật sự là XYZ hay window thật sự đúng thứ tự. Người chuẩn bị dữ liệu chịu trách nhiệm xác minh thông tin này. Chưa test MAT v7.3 hay WiPose thật trong phiên đóng gói; đã test MAT tổng hợp và NPZ tổng hợp.

## 6. Output

MM-Fi: `runs/<timestamp>/<scenario>/`; WiPose: `runs_wipose/<timestamp>/<scenario>/`.

- `train.log`: tiến độ, metric, traceback.
- `metrics.csv`: validation metric từng epoch, loss, lr.
- `best.pt`, `last.pt`: checkpoint sau epoch.
- `config.json`: setting, schema và giới hạn.
- `test_metrics.json`: đánh giá test từ best checkpoint khi xong epochs.
- `status.json`: complete/failed/preflight_pass.

PCK20/PCK50 là torso-normalized threshold0.2/0.5 theo repo, đơn vị%, không phải20/50mm. Metric XY chung, XYZ chỉ ghi thêm khi output XYZ. Nhãn tọa độ phải tính bằng mét trước khi đổi metric sangmm. MM-Fi báo R.Hand13/L.Hand16; WiPose báo R.Wrist4/L.Wrist7.

Baseline/TCN MM-Fi giữ Z-weightedXY loss gốc `/32`; WiPose native giữ confidence-weightedXY loss `/32`. Full dùng lossXYZ với warm-up. So sánh Full thay đồng thời nhiều thành phần; batch32windows cũng khác batch32frames. Không gọi đó là phép ablation chỉ thay TCN. Chưa có resumeCLI hoặc metric jitter; checkpoint chỉ ghi sau epoch.

## 7. Thành phần cải tiến

| Thành phần | File |
|---|---|
| Backbone baseline MM-Fi và WiPose | `model/HPE_no_denoiser.py`, `model/HPE_Wipose.py` |
| TCN depthwise đa nhánh dilation 1/2/4 | `model/tcn_block.py` |
| Tích hợp backbone, TCN và decoder XYZ | `model/st_hpe.py` |
| MSE, bone, velocity, symmetry và warm-up | `utils/losses.py` |
| Thực nghiệm MM-Fi baseline/TCN/Full | `train_mmfi.py`, `run_experiments.py` |
| Thực nghiệm WiPose | `train_wipose.py` |
| Đọc trực tiếp MM-Fi ZIP | `dataset_lib/mmfi_zip.py` |

## 8. Code cũ và code mới

Đọc [SO_SANH_CODE_CU_MOI.md](docs/SO_SANH_CODE_CU_MOI.md) để xem toàn bộ bản đồ. Baseline backbone được chép nguyên vẹn; module mới tái sử dụng backbone. `MANIFEST.json` có SHA256 và ghi nguồn từng file. Không kèm dataset, checkpoint thật, log train thật, môi trườngPython hay `.git`.

Tài liệu `docs/EXPERIMENTS.md`, `docs/ERROR_LOG.md` là lịch sử từ repo nguồn, có đường dẫn máy cũ. Dùng README này cho máy mới. README tác giả được giữ ở `docs/UPSTREAM_README.md` để ghi nhận nguồn; gói portable không phải bản phát hành chính thức của tác giả.


## 9. Phạm vi phiên bản này

Repository này chứa source của gói `ST-HPE-Portable` tại thư mục gốc để clone và chạy độc lập. Tài liệu lịch sử có thể còn dùng tên thư mục cũ; không cần có thư mục HPE-Li-ECCV2024 trên máy chạy. Dữ liệu và trọng số thật phải được chuẩn bị riêng.

Kiểm thử kỹ thuật đã PASS trên dữ liệu giả cho cả MM-Fi/WiPose. Chưa khẳng định hiệu quả temporal trên WiPose (TCN nhận T=1), chưa có nhãn XYZ WiPose thật đã xác nhận, và chưa có metric jitter hoặc resume CLI. Chi tiết giới hạn được giữ trong hướng dẫn để tránh nhầm PASS kỹ thuật với kết quả nghiên cứu.
