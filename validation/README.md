# Bằng chứng kiểm thử gói portable

**Cập nhật nhánh thử WiPose ngày 19/11/2026:** đã đọc MAT v7.3 thật và hoàn thành pilot nhỏ cho Baseline/CSI-TCN. Xem [bằng chứng riêng của nhánh](wipose_trial/README.md). Các kết quả và giới hạn bên dưới là lịch sử kiểm thử v2 ngày 29/10/2026, không phải trạng thái mới nhất của nhánh.

**V2 ngày 29/10/2026:** summary/log đã được cập nhật cho engine mới. Xem `test_reliability.py.log` (13 test), `cuda_pilot.json` (BF16/resume trên GPU) và `mmfi_preflight.json` (chỉ kiểm tra dữ liệu thật, không train). Bằng chứng archive được cập nhật theo gói v2.

Kiểm thử ngày 29/10/2026 bằng Python3.12 trên CPU, cwd là chính thư mục ST-HPE-Portable, không import repo nguồn.

- `summary.json`: kiểm tra cú pháp và mã kết thúc của ba bài test, đều PASS.
- `test_sanity.py.log`: MM-Fi/WiPose forward/backward, loss thành phần, warm-up, Params/MACs và tắt nâng cấp.
- `test_training_smoke.py.log`: một epoch giả cho baseline/TCN/Full MM-Fi, train/validation/test và best/last checkpoint.
- `test_wipose_smoke.py.log`: MAT tổng hợp cho baseline/TCN; XYZ-windowNPZ tổng hợp cho Full; đối chiếu normalization gốc; xác nhận confidence bị từ chối làmZ.
- `environment_reference.json`: phiên bản thư viện máy nguồn, chỉ để tham khảo môi trường đã test.

Chưa huấn luyện WiPose thật, chưa kiểm thử đọcMATv7.3/HDF5 trong môi trường này (chưa cài mat73/h5py), chưa xác minh nhãnXYZ thật của WiPose. Kiểm thử dummy không chứng minh cải thiện độ chính xác.

Checkpoint/data tạm của các bài smoke test không nằm trong fileZIP phát hành. Nếu tự chạy lại test, chúng sẽ được tạo trong runs_smoke và log sẽ cập nhật. Checksum manifest chỉ bảo vệsource/tài liệu, không bao gồm bằng chứngvalidation có thể tái sinh.
