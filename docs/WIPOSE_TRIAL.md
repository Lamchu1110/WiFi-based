# Recipe và giới hạn của thử nghiệm WiPose

**Lịch sử nhánh AdamW 19/11/2026.** Protocol SGDM/70–30 mới nằm ở [WIPOSE_PAPER_PROTOCOL.md](WIPOSE_PAPER_PROTOCOL.md), dùng `run_wipose_paper.py`. Ngày 23/11 đã đối chiếu lại paper: SGDM/batch 32 được mô tả chung cho các thí nghiệm, không chỉ riêng MM-Fi. Cũng đính chính rằng `att_wipose.py` đặt lại danh sách metric mỗi epoch; không có bằng chứng nó cộng lẫn metric qua epoch.

Nhánh: `experiment/wipose-baseline-tcn`, bắt đầu từ `main` commit `3e7e2c8`, ngày 19/11/2026.

## Cơ sở đối chiếu

- [Repository HPE-Li](https://github.com/Toandinh1/HPE-Li-ECCV2024), cụ thể `att_wipose.py`, `constant.py`, `HPE_Wipose.py` và loader WiPose.
- Snapshot Git cục bộ để đối chiếu backbone: `854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7`.
- [Wi-Pose Dataset](https://github.com/NjtechCVLab/Wi-PoseDataset): nhãn ảnh XY/confidence cho 18 khớp; 5 mẫu CSI/frame, 30 subcarrier và 3×3 anten.
- Archive Wi-Pose.rar cục bộ đã kiểm tra có 132847 file Train và 33753 file Test. Mẫu thật có field `CSI` và `SkeletonPoints`, MAT v7.3. Đây là schema mà code HPE-Li đọc; README dataset còn mô tả tên field khác nên loader hỗ trợ cả hai.

## Những gì được giữ và thay đổi

| Thành phần | Baseline và TCN trong nhánh này |
|---|---|
| Backbone/decoder | `HPEWiPoseModel`, XY, cùng khởi tạo backbone khi seed giống nhau |
| Input | `[B,9,30,5]`, chuẩn hóa bằng mean/std upstream, giữ đúng phép transpose anten/thời gian |
| Nhãn | 18 bộ XY/confidence, XY nhân 0.001 để train như upstream |
| Optimizer | AdamW, lr 0.001, betas mặc định, weight_decay 0.01 |
| Batch | 64; giữ batch cuối nếu có ít nhất 2 mẫu |
| Objective mặc định | `MSE(conf * pred, conf * target) / 32`, áp dụng giống nhau cho cả hai mô hình |
| Lịch learning rate | `max(0, 1 - max(0, e+1-20)/31)`, với `e` là chỉ số scheduler |
| Epoch | Pilot 1; train mặc định 50; có thể chạy thử 3 |
| Precision / clipping | FP32; clipping tắt, vẫn kiểm tra gradient/parameter hữu hạn |
| Chọn checkpoint | PCK20 validation cao nhất; không chọn bằng test |
| Chia tập | Train nguyên gốc; Test tách 50/50 validation/test với seed 41, filename được sort |
| TCN | Baseline giữ pooling gốc; TCN giữ T=5 qua backbone, TCN theo CSI time rồi pool về decoder |

`constant.py` của snapshot nguồn đặt **60 epochs**, trong khi lịch learning rate của script dùng mốc 20 và 30, có thể thành âm khi chạy quá dài. Nhánh thử cũ đặt **50 epochs** cho run đầy đủ và chặn learning rate âm; đây là recipe từ script công khai, khác SGDM/batch 32 mô tả trong bài báo.

Bản gốc không cố định thứ tự `os.listdir`; nhánh này sort tên file để split lặp lại được. Vì vậy cùng seed 41 không bảo đảm đúng từng mẫu của một lần chạy upstream không lưu inventory. Các tập có cùng dữ liệu/seed trong một suite của nhánh này thì giống nhau hoàn toàn.

## Đọc metric đúng đơn vị

Nhãn công khai được suy từ ảnh bằng AlphaPose; không có phép hiệu chuẩn biến pixel thành khoảng cách vật lý trong loader. Vì vậy nhánh này mặc định khai báo `pixel` và ghi `MPJPE_xy_px`, `PA_MPJPE_xy_px`. Nhân tọa độ với 0.001 để tối ưu không biến nhãn ảnh thành mét. `--wipose-coordinate-unit mm` chỉ dùng cho dữ liệu đã được xác minh có nhãn mm thật; metric khi đó mang tên `_mm`.

PCK dùng torso giữa joint 5 và joint 8 theo code nguồn. Ngưỡng 0.2/0.5 là tỉ lệ chiều dài torso. Joint confidence bằng 0 bị loại; pose có torso không hợp lệ được thống kê và loại khỏi PCK. PA alignment tính riêng từng pose. Khác biệt với evaluator nguồn gồm aggregation/visibility; cả hai đều đặt lại metric mỗi epoch.

Các cột `R.Wrist_*`, `L.Wrist_*` là khớp cổ tay phải/trái (joint 4/7). Đây là khớp tay mà ta theo dõi trong thử nghiệm WiPose.

## Schema loader

- Hỗ trợ MAT v5 bằng scipy và MAT v7.3 bằng mat73/h5py; field `CSI/SkeletonPoints` hoặc `csi_serial/jointsVector`.
- CSI phải đúng `[5,30,3,3]`; CSI phức được lấy biên độ. Không chấp nhận nhầm trục chỉ vì cùng số phần tử.
- `SkeletonPoints` phẳng 54 phần tử dùng thứ tự coordinate-major đã đối chiếu trực tiếp archive thật: 18 X, 18 Y, 18 confidence.
- `jointsVector` dạng `[18,3/4]` hoặc `[3/4,18]` nhận diện được; dạng phẳng cần `--joint-vector-layout coordinate-major` hoặc `joint-major`. Với `(x,y,c,c)`, hai cột confidence phải trùng nhau.
- Loader kiểm tra finite và confidence không âm, báo tên file gây lỗi. Không tự xem confidence là Z.
- Preflight chỉ lấy một số mẫu, không tuyên bố đã quét mọi file; khi train, mỗi mẫu tiếp tục được kiểm tra lúc đọc.

## Cách đánh giá công bằng

Cùng suite dùng cùng split, seed, thứ tự sample và recipe. Bảng so sánh từ chối ghép hai run có contract khác nhau. Bảng tự động chỉ xuất sau khi có test metric của model hoàn thành. Nếu mới có một model thì trạng thái là `partial`.

Pilot lấy một tập con cố định để kiểm tra luồng thực thi, chưa đủ đánh giá độ chính xác. Run 3 epoch cho thấy tiến độ học ban đầu; kết luận đóng góp cải tiến cần ngân sách đủ dài, nhiều seed và kiểm tra protocol. Không dùng chỉ số pilot để so với bảng trong bài báo.

Đường CSI-TCN kết hợp 5 mẫu CSI nằm trong một frame, không phải 5 ảnh/frame tư thế. Việc thay pooling giữ T=5 là một phần của biến thể. Nhánh này không chạy pose-TCN theo các file nối tiếp vì thứ tự/metadata chuỗi chưa được xác minh đầy đủ; không bật decoder XYZ hoặc kinematic loss cho nhãn 2D.

## Checkpoint và log

`run_wipose_comparison.py --resume-suite PATH` đọc lại cấu hình đã lưu, giữ nguyên kết quả của model hoàn thành và resume model bị gián đoạn. Resume không cho đổi epoch/batch/loss. Có thể đổi device nhưng không hứa bit-exact qua GPU/precision khác nhau.

Checkpoint lưu optimizer AdamW, scheduler, RNG, model tốt nhất và vị trí batch. Contract bổ sung optimizer/scheduler/unit/drop-last; checkpoint ở main/v2 cũ cần run mới. Mọi lỗi trong chạy thử được ghi vào `suite.log`, `train.log` và `status.json`.
