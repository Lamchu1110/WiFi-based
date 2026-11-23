# Nhật ký triển khai WiPose v3

- 23/11/2026: Đóng băng code/kết quả trước thay đổi tại kho `ablation_archive/20261123_pre_ablation_v1` của worktree paper; nhánh mới bắt đầu từ `f3af7ad`.
- DSK kế thừa: kiểm tra attention đều xác nhận phần tử bị đổi vị trí kênh/tần số dù shape đúng. Giữ nhánh `legacy` và thêm nhánh `corrected` có kiểm thử vị trí và trọng số nhánh. Chưa kết luận hiệu quả accuracy khi sửa.
- Lượt kiểm thử đầu bằng Python mặc định không chạy được vì thiếu `h5py`. Lỗi đầy đủ giữ trong `validation/v3/test_wipose_ablation.log`. Chuyển sang môi trường có đủ requirements; kiểm tra CPU với PyTorch 2.7 và GPU local với PyTorch CUDA.
- Lượt unit test đầu trên PyTorch 2.7 có 2 lỗi ở fixture: 89 mẫu chia batch 8 để lại batch cuối 1 mẫu, được guard BatchNorm từ chối đúng. Đổi batch của fixture final thành 16, giữ nguyên guard và thêm test riêng cho trường hợp singleton. Log ban đầu giữ tại `validation/v3/test_wipose_ablation_torch27.log`.

- Xác nhận cuối: 10 test mới và 8 test paper regression PASS trên PyTorch 2.7 CPU; smoke train MM-Fi/XYZ/kinematic PASS; ba cấu hình chính hoàn tất pilot trên dữ liệu WiPose thật với GPU local. Chi tiết và giới hạn ở `validation/v3/README.md`.
- Khi đóng gói, Git chưa cấu hình author nên lần commit đầu bị từ chối. Dùng danh tính `Codex <codex@local.invalid>` đã có trong ba commit trước, chỉ áp dụng cho lệnh commit này; không đổi cấu hình Git toàn máy.

Kết quả kiểm thử tiếp theo được lưu ở `validation/v3/`; các run thực tế ở `runs_wipose_v3/` tự lưu trạng thái, traceback, source snapshot và config. Không xem lỗi môi trường ban đầu là lỗi training đã chạy, và không xem pilot là kết quả benchmark.
