# Cam kết chỉ số — T05, cố định trước đánh giá chính thức

> Giữ các nhóm trường của bản mẫu; bộ chấm chuẩn tại `src/metrics/tracks.py`
> không thay đổi. Chưa có GT giảng viên và chưa công bố điểm chính thức.
> Không đổi tập chấm, ngưỡng hoặc trọng số sau khi xem kết quả chính thức.

| Mục | Khai báo đã điền |
|---|---|
| Bài thực hiện | T05 · Leakage & Near-Duplicate Guardian |
| Chỉ số chính, đúng tên evaluator | **Leakage Detection F1 (pair)** |
| Chiều tốt | Tăng là tốt (`higher`) |
| Đơn vị đánh giá | Cặp ảnh không thứ tự, hai `image_id` khác nhau, chuẩn hóa thứ tự ID |
| GT dương | Hai ảnh cùng `dup_group_id`; mở rộng tất cả cặp trong mỗi nhóm GT |
| Dự đoán dương | Nếu có `pred_dup`, chỉ lấy dòng `pred_dup=1`; pipeline luôn xuất cờ 0/1 |
| Ghép dự đoán với GT | Khớp chính xác cặp ID chuẩn hóa; không IoU/dung sai hình học |
| Precision | TP/(TP+FP): cặp dương đúng / số cặp dự đoán dương |
| Recall | TP/(TP+FN): cặp GT dương tìm được / số cặp GT dương |
| F1 | 2PR/(P+R); helper chuẩn trả F1=0 khi P/R bằng 0 hoặc không xác định; giữ xử lý mẫu số rỗng |
| Tập ứng viên cố định | Bộ công khai 120 ảnh tạo C(120,2)=7.140 cặp cho cả baseline/PoC; số cặp GT dương chưa biết |
| Ngân sách duyệt B | PoC tối đa 20 cặp; baseline SHA-256 không có lượt duyệt, không báo số lượt đã dùng bằng nhau |
| Seed | 42 cho pipeline/chia tập; 2027/9173 chỉ là seed sinh nguồn ảnh xác nhận tổng hợp |
| Chọn/cố định ngưỡng | Từ thiết kế và dữ liệu tổng hợp nội bộ; `configs/tracks/t05.yaml`, không dùng GT giảng viên |
| Tập chấm không dùng để tune | Toàn bộ bộ GT giảng viên khi được cung cấp; T05 không có lọc split tổng quát, không tự bỏ mẫu chấm |
| Sai số cần kiểm soát | FN ảnh biến đổi; FP/gộp nhầm cảnh; rò rỉ sau chia tập và mất dữ liệu |
| Lát cắt cần báo cáo | `transform` của giảng viên khi có GT; nội bộ: giống byte, JPEG, độ sáng, crop, cùng cảnh không trùng, khác cảnh trông giống |
| Bất định khi mẫu nhỏ | Bootstrap theo nguồn/cảnh, không coi các cặp là độc lập; báo N, tử số/mẫu số. Tái hiện báo cáo đặt số lượt bootstrap=0 nên không báo CI chưa tính |
| Ghi nhận giao thức | V2 bổ sung ngày 15/11/2026 trước khi có GT; tham số trong commit `69301c20612c98130438607c1d81ece81457fb3c` |
| Chủ sở hữu / phê duyệt | `Nguyen Hai Dang` theo cấu hình và xác nhận người dùng; chưa có ký duyệt chỉ số của người thật. Giờ commit không được coi là giờ ký duyệt |

## Chỉ số phụ từ bộ chấm giữ nguyên

| Chỉ số, giữ nguyên tên kỹ thuật | Tử số / mẫu số | Chiều tốt |
|---|---|---|
| Pair precision | TP / số cặp dự đoán dương | Tăng |
| Pair recall | TP / số cặp GT dương | Tăng |
| False merge rate | Cặp dự đoán dương có hai `scene_id` khác nhau / số cặp dự đoán dương | Giảm |
| Residual cross-split leak (near-dup) | Cặp GT trùng có hai ảnh được giữ nhưng khác split / số cặp GT trùng có cả hai ảnh được giữ | Giảm |
| Residual cross-split leak (scene) | Cặp cùng cảnh GT có hai ảnh được giữ nhưng khác split / số cặp cùng cảnh GT có cả hai ảnh được giữ | Giảm |
| % data kept | Ảnh GT không thiếu split, không rỗng và không `drop` / tổng ảnh GT; evaluator dùng thang 0–1 | Tăng |

Đặt `split_manifest.csv` cạnh `suspect_pairs.csv`. Hai phương pháp dùng cùng bộ
công khai, seed và biểu diễn cặp. Bộ chấm cần GT để tính mẫu số; tỷ lệ giữ ảnh
quan sát được không thay thế kết quả chấm chính thức.

## Chính sách triển khai cố định

- pHash/dHash 8×8, Hamming tối đa 8/8; histogram 16 bin/kênh, L1 tối đa 0.08.
- Ảnh thu nhỏ xám 32×18, MAE chuẩn hóa tối đa 0.10.
- Điểm thô: `0.30*pHash + 0.30*dHash + 0.30*thumbnail + 0.10*histogram`, dùng tương
  đồng theo ngưỡng. Điểm heuristic, không phải xác suất đã hiệu chỉnh.
- Cổng thô: điểm tối thiểu 0.75 và ít nhất hai tín hiệu không phải histogram;
  cùng SHA-256 luôn dương. Khoảng bất định 0.55–0.85, B=20.
- Burst tối đa 60 giây/cạnh và 60 giây toàn chuỗi; nhóm quá 20 ảnh cần HOLD.
- Chia nguyên nhóm, test mục tiêu 0.30; mặc định không chấp nhận drop hay review
  chưa giải quyết. GO chỉ chứng nhận cấu trúc quan sát được.

### Bổ sung v2 trước đánh giá chính thức

V2 giảm lỗi trên ví dụ tổng hợp đã xem, không đổi ngưỡng thô/seed/ngân sách.
Ví dụ đã xem là phát triển/hồi quy, không còn là holdout chưa quan sát.

| Tham số / quy tắc | Giá trị cố định |
|---|---|
| `verification_enabled` | 1; 0 chỉ để đối chứng phiên bản cũ trong thí nghiệm |
| Kích thước RGB | 48×32 |
| Crop tâm mỗi phía | 0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12 |
| Hệ số sáng chung RGB | [0.625,1.6], không khớp màu riêng từng kênh |
| Sai số RGB tương đối đối xứng tối đa | 0.05 |
| Độ biến thiên không gian tối thiểu | 0.02; ảnh phẳng không giống byte chờ duyệt |
| Kết luận tự động | Cặp không giống byte phải xác minh RGB; crop có cấu trúc đã xác minh có thể thay cổng hash toàn ảnh |
| Điểm sau xác minh | Lấy lớn hơn giữa điểm thô và tương đồng RGB đã xác minh; lưu điểm thô ở `coarse_duplicate_score` |
| Chọn mục duyệt | Cân bằng cặp đã nhận, cặp bất định, cạnh cầu bị chặn, bẫy màu; đa dạng quan hệ nhóm, B=20 |

Hai bộ nguồn seed 2027/9173, crop 7%, JPEG 55, độ sáng 0.9 được thiết kế sau khi
cố định thông số. Đã tái hiện nhưng không chỉnh lại ngưỡng. Thành công trên bộ
nhỏ không chứng minh chất lượng cảnh thật. Không ép dùng lại review v1 bằng cách
sửa fingerprint.

## Minh chứng nội bộ và điểm chính thức

`tools/t05_experiments.py` gắn `internal_development_only_not_official`. Không
chép số tổng hợp vào hai tệp metrics chính thức; không gán GO thành F1 hoặc
khẳng định rò rỉ theo GT bằng 0.

Chỉ khi có GT giảng viên mới chạy:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --track t05 --pred out\baseline\suspect_pairs.csv --run-name baseline --out reports\metrics_baseline.json
.\.venv\Scripts\python.exe -m src.evaluate --track t05 --pred out\poc_v2\suspect_pairs.csv --run-name poc --out reports\metrics_poc.json
.\.venv\Scripts\python.exe -m src.compare reports\metrics_baseline.json reports\metrics_poc.json --out reports\table1_baseline_vs_poc.md
```

Nếu baseline F1=0, ghi `baseline = 0`, không chia cho 0 để báo phần trăm cải thiện.
Giữ cả kết quả không cải thiện hoặc xấu hơn.

## Tình trạng hiện tại

- GT giảng viên: **chưa cài**, chưa có điểm/lát cắt chính thức.
- Hai metrics và bảng so sánh: **chưa sinh**, không tạo bản giả.
- Tham số v2: đã có trong commit thật `69301c...`, không đồng nghĩa có ký duyệt.
- Ký duyệt con người và tag mốc nộp: **chưa xác nhận/chưa có tag cục bộ**.
- Tác vụ báo cáo không thay evaluator, ngưỡng hoặc mã pipeline.
