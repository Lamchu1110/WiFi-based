# Nhật ký sửa lỗi và kiểm thử

## 2026-11-23 — Protocol theo paper và cache WiPose

- Đối chiếu lại mục 4/Bảng 1: paper ghi SGDM/batch 32/50 epoch, WiPose 70/30. Script công khai AdamW/batch 64 là recipe khác. Nhận định trước đây “SGDM chỉ áp dụng MM-Fi” không đúng với mô tả paper.
- Đính chính nhận định cũ về metric cộng qua epoch: `att_wipose.py` có reset `metric` trong từng epoch. Những khác biệt đã xác minh là aggregation/visibility, split và optimizer, không phải lỗi cộng lẫn epoch.
- Thêm `run_wipose_paper.py` với SGD lr 0.001, momentum 0.9, batch 32, MSE XY, 70/30. Theo lựa chọn của người dùng, chỉ đánh giá epoch cuối, không dùng test chọn checkpoint. LR cố định, weight decay 0 và split seed 41 là lựa chọn được công bố rõ, không giả định chúng là thông số tác giả đã xác nhận.
- Thêm cache float32 mmap, checksum, ghi hoàn thành atomic, kiểm tra nguyên vẹn raw/cache, persistent workers. Pilot chỉ cache mẫu nhỏ. Gom finite checks và số liệu loss để giảm đồng bộ CPU/GPU; thêm timing theo từng công đoạn.
- Timer `monotonic` trên máy Windows trả các khoảng ngắn bằng 0; đã chuyển phép đo công đoạn sang `perf_counter` để đo chi tiết hơn. Không ảnh hưởng cập nhật mô hình.
- Checkpoint từ nhánh AdamW cũ không được resume vào protocol mới. Giữ riêng `final.pt`, không gọi là best theo test.
- Bằng chứng kiểm thử và phạm vi đo tốc độ được lưu trong `validation/wipose_paper/`. Không kết luận cải thiện độ chính xác từ pilot ngắn.

## 2026-11-19 — Nhánh thử WiPose Baseline/CSI-TCN

- Thêm `run_wipose_comparison.py`: preflight, pilot nhỏ, train hai mô hình cùng recipe, bảng CSV/JSON và resume suite. Lỗi runtime được lưu cùng log của suite và model.
- Đối chiếu `att_wipose.py`: dùng AdamW lr=0.001, batch=64, weight_decay=0.01 và objective upstream cho cả cặp. Không áp recipe SGDM của MM-Fi sang WiPose. Chặn learning rate âm; full run mặc định 50 epoch, khác hằng số 60 epoch của snapshot nguồn.
- Đọc MAT v7.3 thật từ Wi-Pose.rar: xác nhận `CSI` shape `[5,30,3,3]`, `SkeletonPoints` phẳng gồm 18 X, 18 Y, 18 confidence. Loader hỗ trợ thêm schema công khai, từ chối vector không rõ thứ tự và lỗi đảo trục, báo tên file khi lỗi.
- Sửa nhãn đơn vị metric của native WiPose: XY ảnh sau khi nhân 0.001 vẫn là tọa độ ảnh; xuất MPJPE/PA-MPJPE theo pixel, không gọi là mm. Confidence không được dùng làm Z.
- Fixture của test loader ban đầu tạo sai số phần tử trước reshape; đã sửa theo tích `5*30*3*3`. Bài test đọc MAT v5/v7.3 và đối chiếu normalization đã PASS.
- Pilot dữ liệu thật đã hoàn thành trên GTX 1650 Ti 4 GB: 256 train / 128 val / 128 test, batch 64, 1 epoch cho Baseline và TCN. Đây chỉ là 4 update mỗi model, không chứng minh cải thiện độ chính xác.
- Kiểm thử hồi quy `test_reliability.py`, `test_training_smoke.py`, `test_wipose_smoke.py` đã PASS. Bằng chứng, môi trường và kết quả tương thích bổ sung ở [validation/wipose_trial](../validation/wipose_trial/README.md).
- Không đưa dataset, checkpoint, môi trường Python hay log run cá nhân lên GitHub. Các log kiểm thử đã chọn trong `validation/` là bằng chứng kiểm tra kỹ thuật.

## 2026-10-29 — Training v2

- Sửa WiPose CSI-TCN còn T=1; nay giữ T=5. Thêm pose-TCN theo window thật, mặc định W=8 cho MM-Fi.
- Bỏ Z-weighted XY loss mặc định, có legacy flag. Thêm symmetry theo GT, confidence mask và các ablation XYZ chung initialization/window/split.
- Thêm GT cache, sequence split, atomic checkpoint/resume giữa epoch, RNG/cursor/optimizer/scaler, OS lock, gradient clipping và các metric vận tốc/gia tốc.
- Phát hiện FP16 NaN trong Conv2d trên GPU nguồn; giữ FP32 mặc định. Pilot BF16 hai update và CUDA resume đã PASS; không kết luận tốc độ từ BF16.
- Sửa smoke test chọn nhầm thư mục WiPose khi chạy lại MM-Fi; output mỗi dataset được tách riêng.
- Kiểm thử CPU và hồi quy có log trong validation; preflight MM-Fi thật W=8 PASS: 27972 train /5994 val /5994 test windows. Không chạy train dài mới trong lần sửa này.
- Chi tiết thay đổi và các hạn chế còn lại: [V2_CHANGELOG.md](V2_CHANGELOG.md). Các mục dưới đây là lịch sử trước v2.

Ghi thêm mục mới sau mỗi lần sửa và kiểm thử; giữ lại lịch sử cũ. Chỉ đánh dấu đã sửa khi có kết quả kiểm chứng.

## Cách lưu kết quả

- Chạy `python test_sanity.py` tại thư mục dự án.
- Script tự ghi nối tiếp stdout, stderr, traceback và mã kết thúc vào `logs/sanity.log`, đồng thời hiển thị trên terminal.
- Mỗi lượt chạy có thời gian UTC+07:00. Mã kết thúc 0 là thành công, khác 0 là còn lỗi hoặc chưa kiểm tra đầy đủ.
- Các lệnh sửa/test khác cần được ghi bổ sung vào nhật ký này; việc ghi tự động hiện áp dụng cho `test_sanity.py`.

## 2026-10-21 — Kiểm thử ban đầu

### ERR-001 — MM-Fi Baseline lệch chiều subcarriers

- Trạng thái: **Chưa sửa**.
- Tái hiện: `python test_sanity.py`, input `[4,3,314,10]`.
- Vị trí: `model/utils/SKNet.py`, reshape attention trong `SKConv.forward`; cấu hình từ `model/HPE_no_denoiser.py`.
- Lỗi: `RuntimeError: shape '[4, 4, 64, 114, 1]' is invalid for input of size 321536`.
- Nguyên nhân đã quan sát: mô hình cấu hình `dim1=114`, input yêu cầu có 314 subcarriers.
- Ảnh hưởng: forward thất bại, chưa chạy được backward và đo THOP cho input này.
- Hướng xử lý: đối chiếu preprocessing/dataset và thiết kế kích thước mô hình trước khi sửa, bảo toàn baseline.

### ERR-002 — Thiếu ST-HPE và KinematicPoseLoss

- Trạng thái: **Chưa triển khai**.
- Chưa có mô hình tích hợp TCN, direct 3D và kinematic loss; chưa có `model/tcn_block.py` và `utils/losses.py`.
- Ảnh hưởng: chưa kiểm tra được forward/backward ST-HPE, bốn thành phần loss và chênh lệch Params/FLOPs.
- Hướng xử lý: triển khai các module và cờ cấu hình theo kế hoạch, sau đó bổ sung kiểm thử thực sự vào script. Script hiện báo thiếu implementation, chưa tự kiểm thử ST-HPE khi các file được thêm vào.

### ERR-003 — Terminal Windows không in được đường dẫn tiếng Việt

- Trạng thái: **Đã sửa và xác nhận** bằng `python test_sanity.py`; traceback có đường dẫn tiếng Việt được ghi thành công.
- Lỗi ban đầu: `UnicodeEncodeError: 'charmap' codec can't encode characters` khi in traceback qua encoding cp1252.
- Sửa: cấu hình stdout/stderr UTF-8; mở file log bằng UTF-8.

### Kết quả và điểm cần đối chiếu

- WiPose Baseline: forward/backward thành công, output `[4,18,2]`; MSE `1.08099949`; 95 tensor gradient tham số hữu hạn và có gradient khác 0; gradient input hữu hạn.
- Params MM-Fi Baseline: `1.058834M`; WiPose Baseline: `6.174484M`.
- WiPose THOP: `0.058419G MACs`, tương đương ước tính `0.116838G FLOPs` theo quy ước 2 FLOPs/MAC, batch 1.
- Params WiPose hiện vượt mục tiêu 2.5M trong kế hoạch; cần đối chiếu kiến trúc trước khi kết luận về mục tiêu ST-HPE.
- Không dùng các mẫu ngẫu nhiên trong batch làm frame liên tiếp để tính velocity loss.

## 2026-10-21 — Sửa input và triển khai ST-HPE

- ERR-001: **Đã sửa và kiểm thử đạt**. Dummy MM-Fi đổi sang `[4,3,114,10]`. YAML chọn frame, không khai báo số subcarriers; loader đọc nguyên CSIamp. Shape này dựa trên cấu hình mô hình gốc và script complexity; chưa đọc file MAT thật để xác nhận dữ liệu cục bộ.
- ERR-002: **Đã triển khai và kiểm thử dummy đạt**. Thêm `model/tcn_block.py`, `model/st_hpe.py`, `utils/losses.py`; export `STHPE` từ `model`.
- TCN có ba nhánh depthwise separable dilation 1/2/4, fusion BN/GELU/1x1 và residual. Chạy sau DSK cuối, trước decoder; giữ trục CSI time, tổng hợp theo frequency rồi broadcast phần residual về feature map.
- ST-HPE hỗ trợ `use_tcn`, `direct_3d` ở constructor, đầu vào frame hoặc window. Tắt cả hai khớp baseline khi dùng cùng trọng số; chưa thay các entrypoint huấn luyện dữ liệu thật.
- Loss có MSE/bone/velocity/symmetry; warm-up tuyến tính: epoch 0 chỉ MSE, epoch 5 nửa trọng số phụ, epoch 10 đủ trọng số (mặc định). Frame đơn có velocity loss bằng 0.
- Lệnh: `python -m py_compile model/tcn_block.py model/st_hpe.py utils/losses.py test_sanity.py`; `python test_sanity.py`.
- Kết quả: syntax PASS; forward/backward cả hai baseline và ST-HPE PASS; window `[4,4,P,3]` PASS; từng loss và gradient hữu hạn, khác 0 trên sequence; warm-up và ablation compatibility PASS. `RESULT: PASS; issues=[]`, exit code 0.
- MM-Fi Params: baseline 1.058834M; ST-HPE 1.308899M; tăng 0.250065M. THOP MACs: 0.343540G / 0.344795G; FLOPs ước tính: 0.687080G / 0.689589G.
- WiPose Params: baseline 6.174484M; ST-HPE 6.424934M; tăng 0.250450M. THOP MACs: 0.058419G / 0.058670G; FLOPs ước tính: 0.116838G / 0.117340G.
- Params đếm mọi tham số đăng ký (bao gồm `skunit4` không được gọi ở WiPose); THOP đo batch 1 mỗi pose, chưa bao phủ đầy đủ functional operations; FLOPs ước tính theo 2 FLOPs/MAC, không phải phép đếm tuyệt đối mọi toán tử.
- Output chi tiết: `logs/sanity.log` (ghi nối tiếp, các lỗi cũ vẫn giữ làm lịch sử).

### ERR-004 — Nhãn WiPose hiện là XY + confidence

- Trạng thái: **Cần xác minh trước huấn luyện 3D thật**.
- `wipose/wipose_dataset.py` nhân 0.001 cho XY nhưng giữ cột thứ ba làm confidence; không được dùng cột này làm nhãn Z.
- Các kiểm thử mới dùng XYZ giả lập, chưa chứng minh loader WiPose có nhãn 3D thật. Cần xác minh nguồn nhãn trước khi nối loss 3D vào training.

## 2026-10-24 — Bộ chạy ablation và khởi chạy Baseline thật

- Thêm `run_experiments.py`: baseline/tcn/full, SGDM lr=0.001 momentum=0.9, 50 epochs, batch=32; CSV metric, log, cấu hình, split và best/last checkpoint riêng.
- Smoke test train/eval/checkpoint cả ba kịch bản trên dữ liệu tổng hợp: **PASS**, output `runs_smoke/20261024_232851_336444`. Đã sửa lỗi stdout cp1252 ở dòng thông báo cuối của smoke test.
- Lần tìm dataset mặc định ban đầu thất bại vì đường dẫn Linux trong mã gốc không tồn tại; traceback lưu ở `runs/20261024_232619_306606/baseline/train.log`.
- Tìm thấy archive thật `D:/Downloads/MMFi_Dataset.zip`, 83.07 GB, 1080 file GT và 320760 file CSI. Mẫu thật CSI `[3,114,10]`, GT `[297,17,3]`, hữu hạn; xác nhận ERR-001 bằng dữ liệu thật.
- CSI + nhãn cần 17.70 GB nhưng ổ C còn 17.28 GB: thêm `dataset_lib/mmfi_zip.py` để đọc trực tiếp ZIP, không giải nén. Kiểm tra danh mục, tất cả GT, CSI đầu/cuối mỗi split trước train; các frame khác kiểm tra khi đọc.
- Baseline thật đã khởi chạy nền PID 39156, 50 epochs/batch32/SGD momentum0.9; CUDA; train 224532, heldout 96228, train batches/epoch 7016.
- Xác nhận batch đầu epoch 1 đã optimizer step thành công: loss 0.10603277. Chưa có kết quả cả epoch tại thời điểm ghi mục này.
- Log thật: `runs/20261024_232918_004573/baseline/train.log`; stdout/stderr launcher trong `logs/baseline_launcher.*.log`; PID ở `logs/baseline.pid`.
- Quy ước so sánh: XY chung cả ba kịch bản, XYZ bổ sung cho Full; PCK chuẩn hóa theo torso như repo, không phải ngưỡng mm. Baseline/TCN giữ loss Z-weighted XY gốc; Full dùng XYZ/windows. Chi tiết tại `EXPERIMENTS.md`.

## 2026-10-24 — Rà soát code phục vụ báo cáo tiến độ

- Tạo `reports/BAO_CAO_TIEN_DO_2026-10-24.md` và `reports/GIAI_THICH_CODE_ST_HPE.md`. Chỉ thêm tài liệu, không thay kiến trúc hay dừng/restart tiến trình train.
- Kiểm tra shape bằng forward hooks ở eval trên CPU: MM-Fi SKUnit1 `[1,64,114,10]`, SKUnit2 `[1,128,57,5]`, TCN `[1,128,5]`; WiPose SKUnit3 `[1,256,7,1]`, TCN `[1,256,1]`.
- Phát hiện giới hạn thiết kế cần công bố: WiPose TCN nhận T=1 nên dilation không học được tương tác giữa thời điểm thực; sequence W hiện được flatten vào batch, chưa có convolution dọc W. Velocity loss vẫn tính theo W đúng.
- Rà soát baseline: `OriginalHPE` theo `main.py`, khác `DSKNetTransMMFI` của `att_mmfi.py`. `constant.py` chưa có vector ghép Z hằng như giả định kế hoạch. SKUnit trong nhánh đang dùng truyền M=4/G=1 cố định; có module đăng ký nhưng forward không dùng.
- Không thay objective trong run đang chạy: baseline/TCN giữ Z-weighted XY loss từ code cũ. Đây là hạn chế cần chốt trước kết luận; không gọi Z là confidence MM-Fi.
- Tại mốc kiểm tra khoảng 23:36 UTC+07 ngày24/10, PID39156 tồn tại, log mới nhất epoch1 step700/7016 loss0.02741695; chưa có validation metric hoặc checkpoint sau epoch.
- Các phần chưa có: metric jitter/velocity trong evaluation, resume CLI, huấn luyện thật TCN/Full, xác minh nhãn XYZ WiPose và đối chiếu protocol đầy đủ với bài báo.

## Mẫu ghi cho lần sửa tiếp theo

- Thời gian:
- Mã lỗi liên quan:
- File và thay đổi:
- Lệnh kiểm thử / input:
- Kết quả thực tế / đường dẫn log:
- Trạng thái sau kiểm thử:
- Việc còn lại:
