# Nhật ký triển khai WiPose v3

- 23/11/2026: Đóng băng code/kết quả trước thay đổi tại kho `ablation_archive/20261123_pre_ablation_v1` của worktree paper; nhánh mới bắt đầu từ `f3af7ad`.
- DSK kế thừa: kiểm tra attention đều xác nhận phần tử bị đổi vị trí kênh/tần số dù shape đúng. Giữ nhánh `legacy` và thêm nhánh `corrected` có kiểm thử vị trí và trọng số nhánh. Chưa kết luận hiệu quả accuracy khi sửa.
- Lượt kiểm thử đầu bằng Python mặc định không chạy được vì thiếu `h5py`. Lỗi đầy đủ giữ trong `validation/v3/test_wipose_ablation.log`. Chuyển sang môi trường có đủ requirements; kiểm tra CPU với PyTorch 2.7 và GPU local với PyTorch CUDA.
- Lượt unit test đầu trên PyTorch 2.7 có 2 lỗi ở fixture: 89 mẫu chia batch 8 để lại batch cuối 1 mẫu, được guard BatchNorm từ chối đúng. Đổi batch của fixture final thành 16, giữ nguyên guard và thêm test riêng cho trường hợp singleton. Log ban đầu giữ tại `validation/v3/test_wipose_ablation_torch27.log`.

- Xác nhận cuối: 10 test mới và 8 test paper regression PASS trên PyTorch 2.7 CPU; smoke train MM-Fi/XYZ/kinematic PASS; ba cấu hình chính hoàn tất pilot trên dữ liệu WiPose thật với GPU local. Chi tiết và giới hạn ở `validation/v3/README.md`.
- Khi đóng gói, Git chưa cấu hình author nên lần commit đầu bị từ chối. Dùng danh tính `Codex <codex@local.invalid>` đã có trong ba commit trước, chỉ áp dụng cho lệnh commit này; không đổi cấu hình Git toàn máy.

Kết quả kiểm thử tiếp theo được lưu ở `validation/v3/`; các run thực tế ở `runs_wipose_v3/` tự lưu trạng thái, traceback, source snapshot và config. Không xem lỗi môi trường ban đầu là lỗi training đã chạy, và không xem pilot là kết quả benchmark.


## 26/11/2026 — V4, temporal trainability

- Snapshot v3 trước sửa: `ablation_archive/20261126_before_tcn_trainability_v4` trong worktree Paper. Tạo worktree v4/nhánh riêng; baseline giữ nguyên forward tác giả.
- Thêm ShortResidualTCN, đối chứng DSK×TCN, nhật ký gradient/cập nhật/residual từng bước được lấy mẫu. Không thay loss, optimizer hoặc chuẩn hóa dữ liệu.
- 11 test temporal/resume/isolation và 10 test ablation regression PASS. Đối chiếu baseline với commit tác giả cố định PASS 4 trường hợp train/eval ở seed 0/17.
- Probe thật `probe_20261126_131711_798907`: 256 ID train từ split remote, 128 bước SGD mỗi cấu hình, cả năm hoàn tất. Không đọc tensor validation/test; toàn bộ checkpoint/log giữ ở `runs_trainability/`. Đây không phải kết quả accuracy benchmark.
- Smoke CLI `pilot_develop_20261126_132139_690542`: năm model hoàn tất nhưng lần cập nhật cuối `comparison.json` gặp `PermissionError [WinError 5]` ở os.replace. Suite giữ trạng thái failed cùng traceback/source/checkpoint. Không xác nhận tiến trình nào đã chặn file.
- Sửa bằng retry có giới hạn cho Windows error 5/32/33 ở atomic replace (tối đa 6 lần, tổng chờ 1,55 giây). Không xóa file đích hoặc thay quyền; lỗi kéo dài vẫn được báo, file đích cũ và file tạm còn nguyên. Áp dụng cho JSON/checkpoint/CSV comparison/predictions của runner v4. Ba test giả lập lỗi tạm/lỗi kéo dài/lỗi khác đều PASS.
- `.gitattributes` giữ nguyên byte với `* -text`, nên kiểm tra whitespace của Git dùng `cr-at-eol` để chấp nhận CRLF. Không đổi byte các file backbone gốc chỉ để chuẩn hóa xuống dòng.

- Kiểm tra sau sửa I/O: suite `pilot_develop_20261126_132319_526591` hoàn tất cả năm cấu hình; comparison đủ năm dòng, không có test_metrics. Kết quả kiểm thử giữ trong `validation/20261126_trainability/RESULTS.json`.
