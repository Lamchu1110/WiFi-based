# Hợp đồng dữ liệu WiPose NPZ

Chỉ cần định dạng này khi bạn có XYZ thật hoặc đã tự chuẩn bị dữ liệu theo split có kiểm soát. Không có script suy ra Z từ nhãn XY/confidence.

Thư mục chứa ba file `train.npz`, `val.npz`, `test.npz`. Mỗi file dùng `allow_pickle=False` và chứa:

| Key | Kiểu / shape | Ý nghĩa |
|---|---|---|
| `x` | float32 `[N,9,30,5]` hoặc `[N,W,9,30,5]` | CSI đã normalize; runner không normalize NPZ thêm lần nữa |
| `y` | float32 `[N,18,3]` hoặc `[N,W,18,3]` | Phải khớp chiềuN/W của x |
| `target_format` | scalar string `xy_conf` hoặc `xyz` | Cột cuối là confidence hay tọa độ Z |
| `coordinate_unit` | scalar string `m` | XY hoặc XYZ đã quy về mét |
| `windows_ordered` | scalar bool | Full cầnTrue: frame cùng sequence, liên tiếp, đúng thứ tự |

Full yêu cầu `target_format='xyz'`, input windowW>=2, `windows_ordered=True`. Baseline/TCN dùng frame riêng4D; nếu schemaXYZ thì lossXY không lấyZ làm confidence. Các split phải khai báo cùngtarget_format. NPZ được nạp vàoRAM, cần đủ bộ nhớ cho ba split.

Ví dụ lưu **các mảng đã được chuẩn bị và xác minh** (không phải tạo nhãn giả):

```python
np.savez('train.npz', x=csi_windows.astype('float32'),
         y=xyz_windows_in_meters.astype('float32'),
         target_format=np.array('xyz'), coordinate_unit=np.array('m'),
         windows_ordered=np.array(True))
```

Thứ tựjoint18 theo loader/nhãn repo: Nose,Neck,R.Shoulder,R.Elbow,R.Wrist,L.Shoulder,L.Elbow,L.Wrist,R.Hip,R.Knee,R.Ankle,L.Hip,L.Knee,L.Ankle,R.Eye,L.Eye,R.Ear,L.Ear. Nhãn khác convention phải map trước; XYZ theo convention khác không thể chỉ reshape.

Chia train/val/test trước khi tạo window. Không để frame cùngwindow vượtsplit, subject/action/sequence boundary; không suy ra thứ tự frame bằng danh sách filename ngẫu nhiên. Với sampling không đều, losschênh lệch/frame không tương đương vận tốc vật lý theo giây.

Schema chỉ xác thực khai báo và shape, không chứng minh sự đúng đắn khoa học của nguồnnhãn. Khai báo `xyz` cho confidence vẫn là dữ liệu sai. File `test_wipose_smoke.py` có dữ liệu ngẫu nhiên chỉ để test chương trình; tuyệt đối không dùng làm bộ dữ liệu nghiên cứu.
