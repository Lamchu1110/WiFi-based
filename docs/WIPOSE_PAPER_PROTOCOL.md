# Protocol WiPose theo thông số công bố — 23/11/2026

Mục tiêu: đối chiếu Baseline từ snapshot repository tác giả và biến thể CSI-TCN trong cùng điều kiện, ưu tiên các thông số được báo cáo trong ECCV 2024. Người dùng đã chọn **70/30, train đủ 50 epoch và đánh giá checkpoint cuối**, không tách validation.

## Nguồn và mức độ tái lập

- [HPE-Li ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04496.pdf): mục 4, trang 8 ghi 50 epoch, SGDM, batch 32, lr 0.001, momentum 0.9; Bảng 1 và phần WiPose trang 9 ghi train/test 70/30, chia ngẫu nhiên; mục 3.4 ghi MSE.
- [Script WiPose công khai, snapshot 854ca8e](https://github.com/Toandinh1/HPE-Li-ECCV2024/blob/854ca8eebc2dd7c5ec7bb1cb9a12da0f1f850cd7/att_wipose.py) lại dùng AdamW/batch 64 và giữ Train, chia đôi Test. Đây là cơ sở nhánh thử cũ, không phải bằng chứng rằng paper dùng AdamW.
- Code của nhánh mới bắt đầu từ `6c0a57d`. Backbone/decoder và CSI-TCN không đổi trong lần cập nhật này. Tính tương đương với snapshot đã được kiểm thử, nhưng chưa chứng minh mọi chi tiết kiến trúc giống bản dùng để tạo bảng ECCV.

## Những thông số có nguồn và những lựa chọn của lần triển khai này

| Nội dung | Thiết lập | Cơ sở |
|---|---|---|
| Optimizer, batch, lr, momentum, epochs | SGD / 32 / 0.001 / 0.9 / 50 | Paper mục 4 |
| Train/test | 70/30 ngẫu nhiên trên toàn bộ các file MAT trong hai thư mục | Paper Bảng 1 và mô tả WiPose |
| Loss | MSE XY; reduction mean | Loại loss từ paper; reduction cụ thể chưa nêu rõ |
| Confidence | Không nhân confidence vào loss; đánh giá mọi khớp | MSE trong paper không ghi confidence; evaluator repo công khai không mask confidence |
| LR schedule | Cố định 0.001 | Lựa chọn triển khai vì paper không mô tả decay; không mặc định dùng scheduler của script AdamW |
| Weight decay, Nesterov | 0, tắt | Lựa chọn triển khai, không phải thông số paper đã xác nhận |
| Seed chia tập | 41, file-list lưu rõ | Seed của chúng ta; không phải seed được tác giả xác nhận |
| Seed model/shuffle | 0 mặc định, có thể đổi cùng cặp | Lựa chọn triển khai |
| Checkpoint đánh giá | Sau epoch 50, không dùng validation/test để chọn epoch | Người dùng chọn; paper chưa mô tả chi tiết này |
| Normalize CSI, scale nhãn | Mean/std upstream; XY nhân 0.001 | Kế thừa loader công khai, giữ ổn định số học |
| Precision | FP32, TF32 tắt, không AMP mặc định | Lựa chọn triển khai chung cho cả hai mô hình |

Không gọi đây là “tái lập chính xác 100%”. Không có file-list/seed thí nghiệm của tác giả thì cùng tỉ lệ không có nghĩa cùng từng mẫu. Random frame split cũng chưa đánh giá khả năng khái quát qua người hoặc chuỗi khác nhau.

## Split và tránh dùng test chọn model

Các tên file mang tiền tố `Train/` hoặc `Test/`, được sắp xếp ổn định rồi dùng `train_test_split(test_size=0.3, random_state=41)`. Hai thư mục gốc chỉ là nơi lưu file; không giữ ranh giới train/test của archive. Bộ 166600 file tạo 116620 train và 49980 test. Ghi đầy đủ tên, kích thước và mtime trong manifest; dataset thay đổi sẽ có fingerprint khác và bị từ chối resume.

Không có validation. `fit` không gọi evaluator trong vòng epoch khi `selection_policy=final`; chỉ evaluate test sau khi hoàn thành ngân sách epoch. `final.pt` không có nghĩa “tốt nhất”. `last.pt` lưu SGD/momentum, scheduler, RNG và cursor để tiếp tục chính xác trong cùng điều kiện.

Với pilot, lấy tập nhỏ cố định từ các phần 70/30; pilot chỉ kiểm tra pipeline. Kết quả pilot không được dùng chọn seed hoặc điều chỉnh kiến trúc nhằm tối ưu test. Nếu sau này cần tuning, thiết kế validation từ phần train và công bố protocol đó như thí nghiệm khác.

## Đơn vị và metric

Đầu ra là 18 XY, confidence vẫn lưu trong nhãn để giữ schema nhưng không dùng làm Z. Mục 3.3 của paper mô tả tọa độ pixel; mục metric dùng tên đơn vị mm, trong khi loader công khai chỉ có phép nhân 0.001. Không có hiệu chuẩn chuyển ảnh sang khoảng cách vật lý, nên nhánh này ghi MPJPE/PA-MPJPE **pixel**. Không đổi tên pixel thành mm để khớp bảng.

PCK20/50 dùng ngưỡng 0.2/0.5 chiều dài torso giữa joint 5 và 8, không phải 20/50 pixel. PA căn chỉnh riêng từng pose. Torso không hợp lệ được thống kê và loại khỏi PCK; pose GT suy biến được thống kê khi không tính được PA. Confidence bằng 0 không bị mask trong protocol mới, khác metric mặc định của nhánh AdamW cũ.

## Tăng tốc và giữ điều kiện đối chứng

Cache lưu đúng tensor float32 sau chuẩn hóa, không fit thống kê trên train/test và không sửa nhãn. Dữ liệu MAT/cache đã được kiểm tra bằng nhau từng phần tử. Các mảng dùng mmap, worker Windows mở riêng, không pickle toàn bộ dataset vào mỗi worker. Cache chỉ được công nhận khi ghi xong cả hai mảng và manifest; checksum được xác minh khi dùng lại. Cache pilot chỉ chứa các mẫu pilot, tránh đọc toàn bộ dataset cho kiểm thử nhỏ.

DataLoader giữ cùng permutation sample theo seed/epoch; có persistent workers và prefetch khi workers > 0. Không dùng augmentation ngẫu nhiên trong loader này. Kiểm thử resume đổi từ 0 sang 2 worker phải cho cùng trọng số trên CPU. Không bảo đảm bit-exact khi đổi GPU hoặc phiên bản PyTorch.

Gom kiểm tra finite của parameter thành một kết quả và gom số liệu loss trước khi chuyển CPU làm giảm các điểm đồng bộ. Không bỏ kiểm tra gradient/NaN, không giảm số batch hoặc số mẫu để báo tốc độ. Pinned memory và copy nonblocking được dùng trên CUDA. Batch train giữ 32; eval batch có thể lớn hơn do không ảnh hưởng cập nhật optimizer.

`performance.jsonl` ghi thời gian chờ loader, cập nhật train, validation (gần 0 ở protocol này) và checkpoint theo epoch/đoạn resume. `test_timing.json` ghi thời gian đánh giá cuối. Các lần đo đầu có thể bao gồm khởi tạo kernel/worker nên không dùng chúng để khẳng định TCN nhanh hơn Baseline.

`benchmark_wipose_io.py` kiểm tra raw/cache bằng nhau trước khi đo đọc + collate CPU, warm-up rồi đảo thứ tự phép đo. Nó không đo forward/backward và không bao gồm thời gian tạo cache. Không suy từ hệ số tăng tốc I/O sang hệ số tăng tốc toàn bộ train.

## Bắt đầu lại và bảo toàn kết quả

Checkpoint 3 epoch trước dùng AdamW, objective khác và split khác. Protocol mới phải train từ đầu. Giữ kết quả cũ trong nhánh/lượt chạy cũ, không ghép chúng vào bảng mới. Có thể resume một lượt mới bị gián đoạn bằng `--resume-suite`; không đổi tổng epoch/split/objective khi resume.

Hai mô hình dùng cùng input, split, MSE, optimizer, số epoch và seed. TCN kèm thay đổi pooling giữ 5 mẫu CSI/frame, nên kết quả không tách riêng tác động của khối TCN khỏi pooling. Chưa bật mô hình XYZ/kinematic hoặc pose windows trên native WiPose.
