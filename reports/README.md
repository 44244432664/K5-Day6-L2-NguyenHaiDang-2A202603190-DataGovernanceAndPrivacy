# reports/ — Các tệp báo cáo T05

Giữ danh sách tệp và ba cột của mẫu kho nguồn; điền cho kho Nguyen Hai Dang.
Các lệnh dùng `python` của `.venv` đã cài `requirements.txt`.

| File | Ai tạo | Cách tạo |
|---|---|---|
| `group_report.md` | Nhóm | Điền tiếng Việt theo bố cục mẫu, từ cấu hình, lần chạy, kiểm thử và Git; tách điểm chính thức khỏi nội bộ |
| `metric_contract.md` | Nhóm | Khai báo tên/đơn vị chỉ số, công thức, ứng viên, ngân sách và ngưỡng trước đánh giá chính thức |
| `metrics_baseline.json` | **Script** | Khi có GT: `python -m src.evaluate --track t05 --pred out/baseline/suspect_pairs.csv --run-name baseline --out reports/metrics_baseline.json` |
| `metrics_poc.json` | **Script** | Khi có GT: `python -m src.evaluate --track t05 --pred out/poc_v2/suspect_pairs.csv --run-name poc --out reports/metrics_poc.json` |
| `table1_baseline_vs_poc.md` | **Script** | Sau khi có hai metrics: `python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json --out reports/table1_baseline_vs_poc.md` |
| `review_queue.csv` | Pipeline và người duyệt thật | Chỉ sao chép bản đã duyệt hợp lệ tương ứng `out/poc_v2/review_queue.csv`; giữ người duyệt/quyết định, không đổi bot thành người thật |
| `decision_log.csv` | Pipeline | Sao chép từ `out/poc_v2/decision_log.csv`, giữ lịch sử và ngữ cảnh quyết định cấu trúc |
| `team_contributions.csv` | Nhóm | Giữ tên cột CSV; đối chiếu tên/email/SHA bằng `git shortlog -sne --all`, `git show --stat <SHA>`; không tự thêm người/commit |

Metrics/bảng so sánh **không được sửa tay**; giảng viên sẽ chạy lại để đối chiếu.
Không tạo điểm giả khi chưa có GT. Đặt `split_manifest.csv` cạnh dự đoán để chấm
rò rỉ sau chia tập và tỷ lệ giữ ảnh.

## Trạng thái ngày 15/11/2026

- Chủ sở hữu: `Nguyen Hai Dang`; nhóm G01, T05, seed 42.
- Có commit triển khai `69301c20612c98130438607c1d81ece81457fb3c`, tác giả
  `44244432664 <haidang453@gmail.com>`. Các chỉnh sửa báo cáo hiện tại chưa tự được commit.
- Kiểm thử mới cho báo cáo: **217 passed, 39 skipped**; 39 mục cần GT chưa cài.
- Baseline/PoC có kết quả chạy và quyết định cấu trúc; không coi GO là điểm F1.
- Chưa có GT, metrics/bảng chính thức hoặc ký duyệt người thật; chưa sao chép
  hàng đợi thành minh chứng duyệt hoàn tất.
- Chưa có tag `demo-180m`, `final-24h` cục bộ; danh sách nhóm và người trình bày
  còn cần xác nhận bổ sung.
- `DOC/`, `PLAN/`, `TASK/`, `docs/T05_*.md` và bản khóa môi trường bổ sung chỉ lưu
  local theo `.gitignore` đã chọn.