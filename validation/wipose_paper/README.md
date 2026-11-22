# Kiểm chứng nhánh paper-protocol — 23/11/2026

Nhánh bắt đầu từ `6c0a57d`. Đây là bằng chứng kiểm tra phần mềm và pilot ngắn, chưa có kết quả train hội tụ 50 epoch. Chưa chạy trực tiếp trên RTX 3060 bên Đài.

| Kiểm tra | Môi trường | Kết quả |
|---|---|---|
| `test_wipose_paper.py` | Python 3.12.8, PyTorch 2.7.0+cpu, Windows | 8 test PASS |
| `test_reliability.py` | PyTorch 2.7.0+cpu | 13 test PASS |
| `test_wipose_trial.py` | PyTorch 2.7.0+cpu | 8 test PASS |
| `test_training_smoke.py` / `test_wipose_smoke.py` | PyTorch 2.7.0+cpu, dữ liệu tổng hợp | PASS |
| Pilot hai mô hình bằng MAT thật | PyTorch 2.13.0+cu126, GTX 1650 Ti 4 GB, FP32 | Complete |

## Những yêu cầu đã kiểm tra

1. SGD lr 0.001, momentum 0.9, batch 32, plain MSE không phụ thuộc confidence hoặc phép chia 32 của nhánh cũ.
2. Split 70/30 không trùng file, không có validation; đổi seed mô hình không làm đổi split.
3. Tensor raw/cache giống nhau từng phần tử, gồm MAT v5/v7.3; sửa tensor trả về không làm đổi cache.
4. Cache bị thay byte phải bị từ chối; source đổi mtime tạo fingerprint khác.
5. Worker Windows 0/2 trả dữ liệu theo cùng thứ tự.
6. Train không evaluate test trong các epoch; chỉ evaluate sau khi đủ số epoch.
7. SGD resume giữa epoch, đổi 0 sang 2 worker, đi qua ranh giới 2 epoch cho cùng model/scheduler/cursor trên CPU như chạy liên tục. Final checkpoint mang `evaluated_epoch`, không mang nhãn best theo test.
8. Suite dừng/resume hoàn thành cả hai model, xuất CSV, không train lại model đã xong và từ chối so sánh contract khác nhau.

Các kiểm thử hồi quy xác nhận đường validation/AdamW cũ và các smoke test MM-Fi/XYZ tổng hợp vẫn hoạt động trong engine đã sửa. Checkpoint cũ từ nhánh trước vẫn cần lượt mới vì contract đã bổ sung chính sách đánh giá.

## Pilot dữ liệu thật

Pool cục bộ gồm 512 MAT thật đã lấy từ archive Wi-Pose.rar trong lần kiểm tra trước, không phải toàn bộ dataset. Chia pool 70/30 tạo 358/154; pilot chọn 256 train và 128 test, cache 384 mẫu được dùng. Mỗi model chạy 1 epoch / 8 update, SGD, batch 32; không có validation. Cả hai hoàn thành train, final checkpoint và test.

Xem `real_pilot/suite.log`, `comparison.csv`, `comparison.json`, `protocol.json` và `environment.json`. Sai số/PCK của 8 update không dùng để suy ra chất lượng cuối hoặc so với bài báo. Timing pilot có khởi tạo kernel và tác vụ kiểm thử CPU đồng thời; không dùng để xếp hạng tốc độ Baseline/TCN.

## Phép đo I/O

`io_benchmark.json` đo đọc + collate CPU sau warm-up, cùng 256 mẫu thật, 3 lần lặp, worker 0, PyTorch 2.7 CPU. Đã kiểm tra exact tensor equality trước khi đo. Thứ tự MAT/cache được đảo giữa các lần để giảm thiên lệch.

- MAT: trung vị khoảng **0.449 s** / 256 mẫu.
- Mmap: trung vị khoảng **0.00316 s** / 256 mẫu.

Phép đo không bao gồm xây cache, forward/backward, checkpoint hoặc test. **Không được gọi hệ số I/O này là tăng tốc train 142 lần.** Cần đo toàn bộ epoch trên máy RTX 3060 để biết mức cải thiện thực tế.

## Giới hạn

PyTorch 2.7 đã kiểm tra trên CPU, không phải build CUDA 11.8 của máy bên Đài. GPU pilot dùng môi trường local 2.13/CUDA 12.6. BF16 không được kiểm chứng trong pilot này; default FP32. Không cam kết mọi seed, GPU, driver hoặc bản dataset sẽ cho cùng số đo. Không có dataset/checkpoint được commit.
