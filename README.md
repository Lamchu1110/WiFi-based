# ST-DSKPose — training v2

**Spatio-Temporal Dual Selective Kernel Network for WiFi-based Human Pose Estimation**

Phiên bản cải thiện độ tin cậy ngày 29/10/2026, kế thừa HPE-Li. Xem [nguồn tác giả](docs/UPSTREAM_README.md). Đây là implementation thực nghiệm; kiểm thử kỹ thuật PASS chưa chứng minh độ chính xác tốt hơn. **V2 thay kiến trúc temporal, loss mặc định và protocol chia tập: bắt đầu run mới, không dùng checkpoint v1 để resume v2.**

## Cài đặt

Chạy từ thư mục gốc repository, Python 3.12 là phiên bản đã kiểm thử:

```powershell
git clone https://github.com/Lamchu1110/WiFi-based.git
cd WiFi-based
python -m venv .venv
# PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
# WiPose MAT v7.3/HDF5:
python -m pip install -r requirements-wipose.txt
python verify_package.py
python validate_portable.py
```

Cài build PyTorch phù hợp driver/GPU của máy mới nếu cần CUDA. Đã bỏ import torchvision thừa khỏi decoder; train và bộ test hiện không cần torchvision. Không kèm dataset, môi trường Python hoặc checkpoint thật.

## V2 đã thay đổi gì?

- **WiPose CSI-TCN giữ T=5** trước TCN, rồi mới pool về kích thước decoder. Không còn T=1 trước TCN ở chế độ này.
- **Pose-TCN chạy theo chuỗi frame W thật**. Kiểm thử ở eval xác nhận thay frame lân cận làm thay đổi dự đoán của frame đang xét. Đây là mô hình window hai chiều, có dùng thông tin tương lai trong window; chưa phải mô hình causal/online.
- Các kịch bản MM-Fi cùng dùng window, batch, stride và split. Có thêm các ablation XYZ để tách tác động TCN và kinematic loss.
- XY MSE mặc định không dùng tọa độ Z làm confidence và không chia mean loss thêm 32 lần. WiPose confidence được dùng làm trọng số không âm và chuẩn hóa theo tổng trọng số.
- Symmetry loss mặc định khớp **độ chênh lệch chiều dài trái/phải của GT**, tránh ép đối xứng tuyệt đối khi GT bất đối xứng. `--symmetry-mode prior` khôi phục prior đối xứng tuyệt đối.
- Cache GT MM-Fi theo sequence; đọc thư mục hoặc ZIP, không còn tải lại toàn bộ GT mỗi frame. Đọc số frame từ dữ liệu, kiểm tra index thay vì giả định mọi action có đúng 297 frame.
- Chia validation/test MM-Fi theo **sequence** mặc định; window không vượt sequence hoặc khoảng frame bị thiếu. PCK/PA-MPJPE xử lý torso/pose suy biến và thống kê số mẫu không dùng được.
- Checkpoint atomic giữa epoch, lưu optimizer/scheduler/RNG/vị trí batch/best model; `--resume` kiểm tra cấu hình và fingerprint dữ liệu. Có khóa OS chống hai tiến trình ghi cùng run.
- Log riêng từng thành phần loss, clipping gradient, tùy chọn deterministic và AMP. Mặc định FP32; AMP BF16 chỉ bật khi yêu cầu và GPU hỗ trợ.

## Chạy MM-Fi

Input một frame `[3,114,10]`; nhãn `[P,3]`, P=17, tọa độ mét. Cấu trúc: `E01/S01/A01/ground_truth.npy` và `wifi-csi/frame001.mat`. Giữ YAML protocol3/random_split; v2 tách heldout theo subject/action sequence để validation/test không chia lẫn frame của cùng sequence. Đây chưa phải một benchmark cross-subject mới được xác lập.

```powershell
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario full --preflight-only
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario baseline
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario tcn
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario full
# Hoặc 3 kịch bản trên tuần tự:
python train_mmfi.py --dataset-zip /path/to/MMFi_Dataset.zip --scenario all
```

Mặc định: 50 epochs, batch 32 **window**, W=8, stride=8, SGDM lr=0.001, momentum=0.9, gradient clip=1, seed=0. Cùng batch32/W8 tương đương 256 frame mỗi optimizer update ở mọi kịch bản MM-Fi. W=8 cho phép nhánh dilation4 nhìn được frame lân cận thật; với W<=4 các tap ngoài của nhánh này chỉ thấy padding. Window cuối không đủ W bị bỏ; metric tính trên các window đã chọn. Stride nhỏ hơn W làm lặp frame/đoạn chuyển động và được ghi vào contract dữ liệu; để so sánh hãy dùng cùng stride.

| Kịch bản | Output | TCN | Kinematic loss |
|---|---|---|---|
| baseline | XY | Tắt | Tắt |
| tcn | XY | Bật | Tắt |
| xyz | XYZ | Tắt | Tắt |
| xyz_tcn | XYZ | Bật | Tắt |
| xyz_kinematic | XYZ | Tắt | Bật |
| full | XYZ | Bật | Bật |

Để tách đóng góp trên cùng objective XYZ và cùng ngân sách:

```powershell
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario xyz_all --select-metric MPJPE_xyz_mm
```

`xyz_all` chạy xyz/xyz_tcn/xyz_kinematic/full. `--temporal-axis auto` dùng pose-TCN khi W>1; chọn `--temporal-axis csi` nếu muốn nghiên cứu CSI trong frame. Tắt TCN và direct3D vẫn khớp backbone baseline ở eval khi dùng cùng trọng số. Không gọi đây là bản tái lập toàn bộ setting bài báo.

Tùy chọn legacy để nghiên cứu lại objective XY cũ:

```powershell
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario baseline --window 1 --split-mode frame --legacy-objective
```

Lệnh legacy chỉ giữ công thức loss XY cũ và kiểu chia frame; các sửa metric, engine và scheduler v2 vẫn áp dụng. Không so lẫn kết quả legacy/v1 với protocol v2 mà không ghi chú.

## Chạy WiPose

MAT native theo cấu trúc `WiPose/Train/*.mat`, `WiPose/Test/*.mat`; loader giữ cách đổi trục CSI/mean/std và XY mm→m từ repo gốc. Cột thứ ba đang được xử lý là confidence. Thứ tự thời gian giữa file native chưa được xác minh, vì vậy không tự tạo window dựa trên filename.

```powershell
python train_wipose.py --dataset-root /path/to/WiPose --scenario baseline --preflight-only
python train_wipose.py --dataset-root /path/to/WiPose --scenario baseline
python train_wipose.py --dataset-root /path/to/WiPose --scenario tcn
```

Native dùng frame và CSI-TCN, T=5. Không có velocity/gia tốc theo pose khi chưa có window được xác minh. Validation/test native vẫn chia theo file (seed41), vì chưa có thông tin sequence đáng tin cậy; chưa tuyên bố hết leakage giữa các file liên quan.

Các kịch bản XYZ cần **nhãn XYZ thật**, đơn vị mét, đúng thứ tự joint18 và window liên tiếp trong từng sequence. Chuẩn bị `train.npz`, `val.npz`, `test.npz` theo [DATA_FORMAT.md](docs/DATA_FORMAT.md):

```powershell
python train_wipose.py --npz-dir /path/to/WiPose_XYZ_windows --scenario full --preflight-only
python train_wipose.py --npz-dir /path/to/WiPose_XYZ_windows --scenario xyz_all --select-metric MPJPE_xyz_mm
```

Không dùng confidence làm Z. Schema chỉ xác thực khai báo/shape, không xác thực bản chất annotation. MAT v7.3 và XYZ WiPose thật vẫn cần kiểm tra trên dữ liệu máy đích. NPZ được nạp vào RAM.

## Pilot, lưu và tiếp tục huấn luyện

```powershell
# Thử 20 optimizer update trên dữ liệu thật, không train toàn bộ ngay:
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --scenario full --max-steps 20

# Giữ cùng dataset/config/window/batch/seed/lr/epochs, bỏ giới hạn pilot:
python train_mmfi.py --dataset-root /path/to/MMFi_Dataset --resume /path/to/run/full/last.pt
```

Resume suy ra kịch bản từ checkpoint, nhưng các tham số ảnh hưởng training phải khớp contract. Nếu run dùng flag khác mặc định, truyền lại các flag đó. `--checkpoint-every 200` lưu giữa epoch mỗi 200 update; Ctrl+C ở ranh giới bước sẽ lưu ngay, nếu đang chạy optimizer thì giữ checkpoint atomic gần nhất. Mất điện/kill có thể mất các bước sau lần lưu gần nhất. V1 checkpoint và dữ liệu/cấu hình khác sẽ bị từ chối.

Đã kiểm thử resume giữa epoch cho trọng số, CSV và test metric trùng khít với run liên tục trên CPU. Không bảo đảm bit-for-bit giữa các GPU, phiên bản thư viện hoặc precision khác. Fingerprint MM-Fi dùng nhãn, inventory/CRC ZIP hoặc metadata file thư mục; không phải SHA toàn bộ CSI đã giải nén. Không chạy hai tiến trình cùng output run.

FP32 là mặc định. Nếu GPU hỗ trợ, có thể pilot `--amp --amp-dtype bfloat16`; loss vẫn tính FP32. BF16 có thể được mô phỏng trên một số GPU, vì vậy không suy ra tốc độ tăng từ việc chạy được. FP16 trên GPU nguồn từng tạo NaN ở convolution trong test nên không dùng làm mặc định. Gradient/loss/parameter không hữu hạn sẽ dừng và giữ checkpoint trước đó. Không tự giảm batch hoặc đổi precision giữa run. Batch32/W8 có thể cần nhiều VRAM; pilot trước, giảm batch khi thiếu VRAM và áp dụng cùng setting cho ablation tương ứng.

## Metric và output

`runs/<timestamp>/<scenario>/` cho MM-Fi, `runs_wipose/...` cho WiPose:

- `config.json`, `split.json`: protocol, fingerprint, cấu hình và số mẫu/window.
- `train.log`, `metrics.csv`: loss thành phần, validation metric theo epoch và tiến độ.
- `last.pt`: checkpoint resume; `best.pt`: trọng số tốt nhất theo validation.
- `test_metrics.json`: test sau khi train xong, từ best validation; không dùng test để chọn checkpoint.
- `status.json`: running/interrupted/failed/complete hoặc preflight_pass.

Mặc định chọn best theo MPJPE XY nhỏ nhất. Dùng `--select-metric MPJPE_xyz_mm` cho controlled XYZ; nếu chọn PCK phải thêm `--select-mode max`. PCK20/50 là phần trăm joint dưới ngưỡng 0.2/0.5 lần torso như repo, không phải mm. MPJPE/PA-MPJPE XY và XYZ được ghi riêng. Confidence=0 bị loại khỏi metric XY. Torso không hợp lệ/GT suy biến được thống kê, các metric không tính được ghi null.

`velocity_*_mm_per_frame`: sai số vector chênh lệch hai pose; `acceleration_*_mm_per_frame2`: sai số chênh lệch bậc2, chỉ có khi W>=3. Đây là proxy về độ ổn định chuyển động, không tự chứng minh mọi loại jitter đã giảm. Không nối frame qua batch/sequence, không quy đổi thành mm/s khi chưa có dt.

## Kiểm chứng và giới hạn

`validate_portable.py` chạy sanity, train/eval/checkpoint MM-Fi/WiPose tổng hợp và `test_reliability.py` (13 regression test). `test_cuda_pilot.py` kiểm tra riêng hai update BF16 và resume trên CUDA nếu có hỗ trợ. `validation/` giữ bằng chứng; metric ngẫu nhiên không phải kết quả nghiên cứu. `MANIFEST.json` và `verify_package.py` kiểm tra source đã chuyển đủ.

Params không tăng thêm so với ST-HPE v1: MM-Fi full 1.308899M, WiPose full 6.424934M (gồm module gốc đăng ký nhưng không gọi). WiPose vẫn chưa đạt mục tiêu 2.5M. CSI-TCN WiPose giữ nhiều vị trí nên FLOPs cao hơn v1; THOP vẫn là phép đo một phần toán tử. Pose-TCN cần profile theo window, không lấy FLOPs của frame chia tùy ý thành FPS.

Chưa tinh chỉnh siêu tham số hay chạy nhiều seed trên dữ liệu thật, chưa bảo đảm mô hình v2 tốt hơn baseline. Hệ số loss, lr, clipping và W là điểm khởi đầu; hãy chọn bằng validation và giữ test cuối cùng. Ghi riêng kết quả từng seed. Xem [V2_CHANGELOG.md](docs/V2_CHANGELOG.md) để biết thay đổi và vấn đề phát hiện trong kiểm thử. Tài liệu lịch sử v1 được giữ để truy vết, không dùng thay hướng dẫn v2 này.
