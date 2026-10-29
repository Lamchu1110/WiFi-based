> Historical V1 documentation. For current behavior see [training v2](README_V2.md).

# Code cũ có gì, gói mới có gì?

## 1. Hai thư mục nằm cạnh nhau

```text
Đồ Án 3/
  HPE-Li-ECCV2024/    # repo làm việc hiện tại, còn log/thực nghiệm và code cũ
  ST-HPE-Portable/    # gói độc lập mang sang máy mới
```

Không di chuyển/xóa repo nguồn và không tác động tiến trình train đang có. Lưu ý repo nguồn hiện đã có các file ST-HPE được thêm từ những lần làm việc trước; nó không còn là một bản archive baseline sạch. “Code cũ” dưới đây nói về thành phần gốc của repo.

## 2. Thành phần gốc

| File/thư mục trong repo cũ | Vai trò |
|---|---|
| `main.py` | Train OriginalHPE và các nhánh denoiser/filter |
| `att_mmfi.py`, `att_wipose.py` | Các entrypoint thử nghiệm gốc; cấu hình khácmain |
| `constant.py` | Đường dẫn và setting gốc, có đường dẫnLinux cố định |
| `model/HPE_no_denoiser.py` | OriginalHPE MM-Fi outputXY |
| `model/HPE_Wipose.py` | WiPose outputXY |
| `model/HPE_basic_cnn.py` | Biến thể BasicCNN |
| `model/HPE_*_denoiser.py` | Các biến thể denoiser/autoencoder |
| `model/sknet_trans_*.py` | Biến thể SK/Transformer |
| `model/utils/` | SKConv/SKUnit, regression, attention và Transformer |
| `dataset_lib/mmfi.py`, `config.yaml` | Đọc MM-Fi và chia tập |
| `wipose/wipose_dataset.py` | Đọc MAT WiPose, nhãnXY+confidence |
| `utils/eval.py`, `noise.py` | Metric và tạo nhiễu |
| `traditional_filter/`, `denoiser_training.py` | Lọc và train denoiser |
| `calculate_model.py`, `comlexity.py` | THOP từ checkpoint/đường dẫn cũ |
| `Figure/`, ảnh, README | Tài liệu/hình bài báo |

Các thành phần đã thêm trước khi đóng gói: `st_hpe.py`, `tcn_block.py`, `losses.py`, `mmfi_zip.py`, `run_experiments.py`, script test, báo cáo và log.

## 3. Gói mới

```text
ST-HPE-Portable/
  README.md
  requirements.txt
  requirements-wipose.txt
  MANIFEST.json
  verify_package.py
  train_mmfi.py
  run_experiments.py
  train_wipose.py
  test_sanity.py
  test_training_smoke.py
  test_wipose_smoke.py
  validate_portable.py
  model/
    HPE_no_denoiser.py
    HPE_Wipose.py
    st_hpe.py
    tcn_block.py
    utils/{SKNet.py,regression.py,utils.py,__init__.py}
  dataset_lib/{mmfi.py,mmfi_zip.py,config.yaml,__init__.py}
  wipose/{wipose_dataset.py,__init__.py}
  utils/{losses.py,eval.py,noise.py,__init__.py}
  docs/
    SO_SANH_CODE_CU_MOI.md
    DATA_FORMAT.md
    EXPERIMENTS.md
    ERROR_LOG.md
    UPSTREAM_README.md
  validation/             # bằng chứng kiểm thử gói
```

Backbone MM-Fi/WiPose, SKNet, regression và loss/TCN/STHPE được copy nguyên vẹn từ repo làm việc. `model/__init__.py` và `model/utils/__init__.py` được rút gọn để chỉ import module được đóng gói; không đổi phép tính model.

## 4. Các bổ sung riêng cho chuyển máy

- `train_mmfi.py`: entrypoint ngắn gọi runner MM-Fi đã có; nhận đường dẫn máy mới quaCLI.
- `train_wipose.py`: runner mới riêng cho WiPose, train/eval/checkpoint/log, cặp baseline/TCNXY và FullXYZ có kiểm tra schema.
- LoaderWiPose: giữ mean/std và cách đổi trục/đơn vịXY của repo, sort file, kiểm tra dữ liệu hữu hạn; thêm scipyMAT và lazy mat73 choHDF5, bỏ đoạn thống kê mẫu có đường dẫn máy cũ.
- `requirements*.txt`: phụ thuộc cần cài, torch/torchvision cài theo máy mới.
- `verify_package.py`, `MANIFEST.json`: kiểm tra source được chuyển đủ và không đổi.
- `validate_portable.py`, testWiPose: kiểm tra thực tế từ thư mục độc lập.

## 5. Gói mới không mang gì?

Không kèm datasets, checkpoint huấn luyện thật, run/log thật, `.git`, cache, `.venv` hay CUDA cũ. Không kèm các biến thể denoiser, BasicCNN, Transformer và script gốc không nằm trên đường chạy ST-HPE. Các baseline tối thiểu vẫn có trong `model/` để train ablation.

Các giá trị đường dẫn máy cũ trong tài liệu lịch sử chỉ mô tả lịch sử; entrypoint mới nhận đường dẫn quaCLI. Không sao chép góiPortable đè lên repo cũ; chạy trong thư mục riêng.

## 6. Những giới hạn giữ nguyên trong snapshot

- TCN chạy theoCSI time trongframe, chưa convolution dọc chuỗi poseW.
- WiPose trướcTCN cònT1; cần sửa kiến trúc nếu nghiên cứu temporal modeling có ý nghĩa.
- FullWiPose cầnXYZ thật; native loader chỉXY+confidence nên chưa dùng cho Full.
- ParamsWiPose khoảng6.42M gồm module đăng ký không chạy; chưa đạt mục tiêu2.5M.
- FLOPs là ước tínhTHOP, chưa đủ mọi toán tử; không tương đương tốc độ thực.
- Chưa có resumeCLI/jitter metric; không tuyên bố đã nâng độ chính xác khi chưa train/so sánh thật.

Ngày đóng gói không phải lần huấn luyện mới. Các lệnh train trongREADME cần được bạn chạy ở máy đích sau khi kiểm tra dữ liệu.
