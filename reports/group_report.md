# Báo cáo nhóm — T05: Phát hiện rò rỉ dữ liệu và ảnh gần trùng lặp

> Cập nhật ngày 15/11/2026. Theo bố cục bản mẫu trong kho nguồn; điền bằng minh
> chứng của kho Nguyen Hai Dang. Chưa có nhãn chuẩn (GT) giảng viên: số liệu tổng
> hợp và kiểm tra cấu trúc không thay thế điểm chính thức.

## 00 · Thông tin chung

| Mục | Giá trị |
|---|---|
| Mã nhóm | G01, theo `configs/lab.yaml` |
| Bài thực hiện | T05 · Leakage & Near-Duplicate Guardian |
| Thành viên đã xác nhận và vai trò | Nguyen Hai Dang · MSSV 2A202603190 theo tên kho; chủ sở hữu cấu hình, tích hợp, chuyển kho và kiểm chứng |
| Thành viên khác | Chưa được cung cấp; cần xác nhận danh sách và quy mô nhóm theo yêu cầu lớp, không tự thêm tên |
| Kho Git | <https://github.com/44244432664/K5-Day6-L2-NguyenHaiDang-2A202603190-DataGovernanceAndPrivacy> |
| Commit chứa triển khai T05 | `69301c20612c98130438607c1d81ece81457fb3c`; tác giả `44244432664 <haidang453@gmail.com>`; 15/11/2026 lúc 13:31:01, múi giờ +07:00 |
| Tag `demo-180m` | Chưa có trong danh sách tag cục bộ |
| Tag `final-24h` | Chưa có trong danh sách tag cục bộ |
| Hạn nộp theo giờ Việt Nam | Chưa được cung cấp; không suy ra hạn nộp từ thời điểm commit |

Mã T05 được chuyển từ working tree kho DangQuocThai. Đổi chủ sở hữu cấu hình
không gán lại toàn bộ tác quyền mã đã nhập. Commit trên có thật, chứa 32 tệp thay
đổi; xem mục 10. Các `run_info.json` hiện có trong `out/baseline/` và `out/poc_v2/`
được tạo trước commit này nên còn ghi `git_sha=b0ce8bf`. Đây là SHA nền upstream,
không phải commit triển khai; không sửa dữ liệu lịch sử để thay SHA.

## 01 · Phát biểu bài toán

- **Ai chịu ảnh hưởng:** người phát triển mô hình, chủ dữ liệu và người đánh giá.
  Ảnh trùng ở cả train/test làm điểm cao giả tạo và gây quyết định phát hành sai.
- **Điểm nghẽn:** SHA-256 chỉ bắt tệp giống byte; nén lại, đổi sáng hoặc crop có
  thể bị bỏ sót. Gom theo cạnh yếu/màu giống nhau có thể nhập nhầm cảnh.
- **Đầu vào → đầu ra:** ảnh, `images.csv`, manifest và cấu hình → dự đoán cặp,
  nhóm trùng, nhóm ràng buộc chia tập, split, hàng đợi duyệt, audit và quyết định.
- **Hai quan hệ riêng:** trùng lặp tạo cặp dương; cùng cảnh/chia tập chỉ buộc ảnh
  ở một tập, không tự biến thành cặp trùng.
- **Cần chủ dữ liệu xác nhận:** ý nghĩa camera/thời gian, chi phí bỏ sót/gộp nhầm,
  quyền sử dụng và người có thẩm quyền phê duyệt.

**GO:** kiểm tra cấu trúc đạt. **HOLD:** còn mâu thuẫn, review chưa giải quyết,
nhóm quá lớn hoặc không thể chia hai tập. **REWORK:** có lỗi cấu trúc có thể sửa,
như nhóm/cặp dương vượt train/test. Baseline luôn HOLD. GO không chứng minh F1 cao
hoặc không còn rò rỉ thật; pháp chế/DPO phải xác nhận các giả định pháp lý.

## 02 · Dữ liệu và nguồn gốc

| Tập/mục đích | Số lượng | Nguồn | Giấy phép | Tổng hợp? |
|---|---:|---|---|---|
| Bộ mẫu công khai | 120 ảnh, 4 camera | `fixtures/t05_near_duplicate/`; nguồn sinh và seed 42 trong manifest | CC0-1.0 theo manifest | Có |
| Bộ hồi quy nội bộ | 20 ảnh từ 4 nguồn | Bộ sinh trong `tests/test_t05_quality_mitigation.py` và `tools/t05_experiments.py` | Ảnh tự sinh phục vụ kiểm thử, không dùng ảnh người thật | Có |
| Hai bộ xác nhận nội bộ | Mỗi bộ 30 ảnh từ 6 nguồn; seed nguồn 2027 hoặc 9173 | Bố cục/kích thước ảnh khác bộ hồi quy | Ảnh tự sinh phục vụ kiểm thử | Có |
| Đánh giá chính thức | Chưa biết số nhãn/cặp | GT riêng giảng viên chưa cài | Chờ manifest của giảng viên | Chưa xác minh |

Schema công khai: `image_id,path,camera_id,captured_at`, không có `scene_id` hay
`dup_group_id`. Manifest có 121 dòng dữ liệu: 120 ảnh và `images.csv`. Đề bài mô
tả 20 cảnh nhưng pipeline không có nhãn cảnh để kiểm chứng. Kiểm manifest bằng
`python -m tools.verify_manifests fixtures/t05_near_duplicate`: **1 manifest, 0 lỗi**.
Không tạo lại manifest để che lỗi hash.

Baseline chia từng ảnh; PoC chia nguyên nhóm. Không bỏ ảnh hợp lệ để làm đẹp số
rò rỉ. Không dùng dữ liệu thật, mô hình tải ngoài hoặc GT giảng viên trong
pipeline; bật chặn mạng và chặn đọc GT khi chạy.

## 03 · Nghiên cứu và thiết kế

```text
Ảnh + metadata + manifest + cấu hình
 → kiểm nguồn gốc/đầu vào
 → SHA-256, pHash, dHash, histogram, ảnh thu nhỏ
 → kiểm chứng RGB dưới crop/độ sáng có giới hạn (v2)
 → phân loại cặp, nhóm trùng có kiểm soát cạnh nối yếu
 → thêm ràng buộc cảnh/burst, chia nguyên nhóm
 → xếp hàng duyệt theo bất định và ảnh hưởng lên kết quả gom nhóm
 → áp quan hệ duyệt, ghi audit chỉ nối thêm, dựng lại nhóm/split
 → kiểm đầu ra, quyết định GO/HOLD/REWORK
```

1. pHash/dHash và ảnh thu nhỏ phản ánh cấu trúc. Histogram chỉ có trọng số 0.10,
   không đủ để kết luận trùng khi đứng một mình.
2. Kiểm các cặp chéo trước khi nhập nhóm; cạnh yếu không tạo nhóm trùng theo
   chuỗi tùy tiện. Nhóm m ảnh xuất đủ m(m−1)/2 cặp theo hợp đồng evaluator.
3. Camera/thời gian chỉ hỗ trợ cảnh/chia tập. Cạnh burst và tổng khoảng thời
   gian burst đều tối đa 60 giây, tránh nối chuỗi không giới hạn.
4. V2 so khớp RGB 48×32 với crop tâm 0–12% mỗi phía; một hệ số sáng chung trong
   [0.625,1.6], không đổi riêng màu từng kênh. Sai số tương đối tối đa 0.05;
   độ biến thiên không gian tối thiểu 0.02. Ảnh ít chi tiết không giống byte
   phải chờ duyệt, không tự kết luận bằng ảnh thu nhỏ phẳng.
5. Hàng đợi cân bằng cặp đã được nhóm nhận, cặp bất định, cạnh cầu bị chặn và bẫy
   màu; đa dạng quan hệ giữa các nhóm thay vì lặp một nhóm cạnh cầu.
6. Đánh đổi: độ lệch tỷ lệ test được chấp nhận để không phá nhóm; RGB tăng thời
   gian chạy nhưng không cần tải weights. Chưa có đối chứng embedding cục bộ.

API/quy tắc chi tiết ở docstring `src/t05/`; ngưỡng tại `configs/tracks/t05.yaml`.
Tài liệu bổ sung ngoài danh sách nộp chỉ lưu local theo `.gitignore`.

## 04 · So sánh baseline và PoC

**Baseline:** SHA-256 bằng nhau + chia ngẫu nhiên từng ảnh, seed 42.
**PoC v2:** nhiều đặc trưng + xác minh RGB + tách nhóm trùng/cảnh + chia theo nhóm
và duyệt/audit. Cùng 120 ảnh, 7.140 cặp ứng viên và schema đầu ra. PoC có B=20;
baseline không có bước duyệt, không báo số lượt duyệt đã sử dụng là bằng nhau.

### Bảng 1 · Chỉ số chính thức — chờ GT

| Chỉ số của bộ chấm | Baseline | PoC | Giải thích |
|---|---|---|---|
| Leakage Detection F1 (pair) | Chưa đánh giá | Chưa đánh giá | Tăng là tốt; cần GT nhóm trùng |
| Pair precision / Pair recall | Chưa đánh giá | Chưa đánh giá | Cần TP/FP/FN theo GT |
| False merge rate | Chưa đánh giá | Chưa đánh giá | Cần nhãn cảnh thật |
| Residual cross-split leak (near-dup) | Chưa đánh giá | Chưa đánh giá | Cần GT nhóm trùng |
| Residual cross-split leak (scene) | Chưa đánh giá | Chưa đánh giá | Cần GT cùng cảnh |
| % data kept | Chưa chấm chính thức | Chưa chấm chính thức | Tỷ lệ giữ ảnh quan sát được được báo riêng |

Chưa sinh `reports/metrics_baseline.json`, `reports/metrics_poc.json` và
`reports/table1_baseline_vs_poc.md`; không tạo số/bảng chính thức giả.

### Chẩn đoán quan sát được — không thay thế Bảng 1

Nguồn: các `diagnostics.json` và `run_info.json` tại `out/baseline/`, `out/poc_v2/`
của kho đích; đây là lần chạy đã lưu trước commit `69301c...`.

| Chẩn đoán | Baseline | PoC v2 |
|---|---:|---:|
| Ảnh / dòng cặp | 120 / 7.140 | 120 / 7.140 |
| Cặp dự đoán dương | 20 cặp giống byte | 128 cặp; chưa biết số đúng theo GT |
| Train / test / drop | 84 / 36 / 0 | 84 / 36 / 0 |
| Cặp dương vượt train/test | 7 cặp giống byte | 0 cặp dự đoán dương |
| Nhóm trùng / nhóm chia tập | Không áp dụng | 59 / 33 |
| Tỷ lệ giữ ảnh | 1.0 | 1.0 |
| Thời gian toàn lần chạy | 3.5191 giây | 15.1477 giây |
| Quyết định cấu trúc | HOLD | GO |

Không suy ra 128 cặp đều đúng hoặc rò rỉ theo cảnh thật bằng 0 từ bảng này.

## 05 · Giao thức đánh giá

- **Đơn vị:** cặp không thứ tự, hai ID khác nhau; ghép chính xác ID, không IoU.
- **GT dương:** hai ảnh cùng `dup_group_id`; mở rộng đủ các cặp trong nhóm.
- **Dự đoán dương:** dòng `pred_dup=1`; các dòng 0 không được tính là dương.
- **Công thức:** P=TP/(TP+FP), R=TP/(TP+FN), F1=2PR/(P+R); giữ nguyên xử lý mẫu số
  rỗng. Khai báo đầy đủ ở `reports/metric_contract.md`.
- **Cố định:** seed 42, điểm thô 0.75, pHash/dHash tối đa 8 bit khác nhau, B=20,
  mục tiêu test 0.30; xác minh v2 theo mục 03 và cấu hình hiện tại.
- **Không tune theo điểm chính thức.** Ví dụ đã xem là phát triển/hồi quy, không
  còn là holdout chưa quan sát. Hai bộ tổng hợp xác nhận không phải hidden eval.
- **Bất định:** công cụ hỗ trợ bootstrap theo nguồn/cảnh, không coi các cặp tương
  quan là độc lập. Lần tái hiện số cho báo cáo đặt `bootstrap_iterations=0`,
  nên không công bố khoảng tin cậy chưa tính.

Chỉ chạy sau khi có GT; đặt split manifest cạnh tệp dự đoán:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --track t05 --pred out\baseline\suspect_pairs.csv --run-name baseline --out reports\metrics_baseline.json
.\.venv\Scripts\python.exe -m src.evaluate --track t05 --pred out\poc_v2\suspect_pairs.csv --run-name poc --out reports\metrics_poc.json
.\.venv\Scripts\python.exe -m src.compare reports\metrics_baseline.json reports\metrics_poc.json --out reports\table1_baseline_vs_poc.md
```

Nếu F1 baseline=0 phải ghi `baseline = 0`, không chia để báo phần trăm cải thiện.

## 06 · Lát cắt kết quả, lỗi và tài nguyên

Đã tái hiện bằng script ở kho đích tại `out/reports_vi_20261115T064455262838Z/`.
Đây là **nhãn tổng hợp nội bộ**, không ghi vào metrics chính thức. Bộ hồi quy có
115 cặp đủ điều kiện; mỗi bộ xác nhận có 235 cặp. Công cụ loại cặp chéo partition.
Đối chiếu ID với ảnh qua `regression/dataset/development_images.csv`; các bảng
`regression/evidence_v1.csv`, `regression/evidence_v2.csv` và `comparison.json`
trong thư mục trên lưu minh chứng do script sinh.

| Bộ/lát cắt | Phiên bản cũ | V2 | Ghi chú |
|---|---|---|---|
| Hồi quy: TP / FP / FN | 24 / 17 / 16 | 40 / 0 / 0 | 40 cặp dương thật tổng hợp |
| Recall cặp liên quan crop | 0/16 | 16/16 | Chỉ bộ hồi quy |
| F1 hồi quy | 0.5925925925925926 | 1.0 | Không phải điểm chính thức |
| Nguồn seed 2027: TP / FP / FN | 36 / 0 / 24 | 60 / 0 / 0 | Crop 7%, JPEG 55, hệ số sáng 0.9 |
| Nguồn seed 9173: TP / FP / FN | 36 / 0 / 24 | 60 / 0 / 0 | Nguồn khác bộ hồi quy |
| F1 hai bộ xác nhận | Xấp xỉ 0.75 mỗi bộ | 1.0 mỗi bộ | Bộ nhỏ, không phải điểm trên dữ liệu thật |

Các lỗi có nguyên nhân cụ thể:

1. **Gộp nhầm bố cục:** `dev:1:b:original`–`dev:1:c:original` là ảnh tổng hợp khác
   cảnh nhưng nhiều hash vẫn đồng ý. Nhiều hash không chứng minh cùng nội dung;
   v2 yêu cầu bố cục RGB tương thích, loại 17 FP hồi quy. Không quy toàn bộ các FP
   này thành lỗi histogram đơn thuần; histogram chỉ đóng vai trò hỗ trợ.
2. **Bỏ sót crop:** `dev:1:a:crop`–`dev:1:a:original` bị biến đổi hash toàn ảnh dù
   còn nội dung tương ứng. Kiểm crop tâm phục hồi 16/16 cặp liên quan trên bộ hồi
   quy, nhưng chưa bao quát crop lệch tâm/phối cảnh.
3. **Rò rỉ sau chia tập:** baseline để 7/20 cặp giống byte ở hai tập. Giữ nguyên
   nhóm suy luận khắc phục giao cắt dự đoán, chưa chứng minh không rò rỉ cảnh thật.

Giữ nguyên 33 lỗi/dự đoán/nhóm của bản cũ, chỉ thay xếp hàng duyệt thì độ phủ lỗi
tăng từ 0/33 lịch sử lên **5/33**, B=20. Còn 28 lỗi không trong hàng đợi; không
tuyên bố duyệt đã giải quyết mọi lỗi.

Máy: Windows 11 `10.0.26300`, Python 3.12.0, Intel64 Family 6 Model 165 Stepping 2.
Từ JSON hiện có, quy đổi baseline khoảng 2.932583 giây/100 ảnh, PoC khoảng
12.623083 giây/100 ảnh. Đây là quy đổi một lần chạy, không phải benchmark cô lập
tải CPU hay trung bình nhiều lần.

## 07 · Con người, nhật ký và quyết định

`out/poc_v2/review_queue.csv` có 20 mục; HTML nhúng hai ảnh để duyệt ngoại tuyến.
Xuất bản đã duyệt thành `review_queue_reviewed.csv` rồi chạy lại cùng cấu hình.
Fingerprint ràng buộc dữ liệu/cấu hình/hàng đợi; không sửa để ép dùng bản cũ.

| Thao tác | Ý nghĩa |
|---|---|
| `approve` | Xác nhận `duplicate` |
| `reject` | Bắt buộc chọn `same_scene_only` hoặc `different_scene` |
| `edit` | Gán rõ một trong ba quan hệ |
| `escalate`, `skip` | Chưa giải quyết; mặc định chặn GO |

Thay đổi cần người duyệt và lý do. `audit_log.jsonl` chỉ nối thêm trạng thái
trước/sau; giữ `automatic_pairs.csv`, dựng lại nhóm/cặp/split/chẩn đoán sau duyệt.
Bộ kiểm thử xác nhận override thực sự đổi đầu ra, không chỉ sửa tệp hình thức.

**Hiện tại:** lần chạy tại kho đích có `n_reviewed=0`; chưa có người thật ký duyệt.
Các lượt bot thử trước chỉ kiểm cơ chế/audit, không được đổi thành phê duyệt con
người. Chủ sở hữu là `Nguyen Hai Dang`; GO mang mã `POLICY_CHECKS_PASSED`, chỉ
phản ánh cấu trúc. Chỉ sao chép hàng đợi đã được duyệt thật vào `reports/`.

## 08 · Tái lập và kiểm thử

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m tools.verify_manifests fixtures\t05_near_duplicate
.\.venv\Scripts\python.exe -m src.main --track t05 --pipeline src.t05.baseline --output out\baseline --run-name baseline --offline --seed 42
.\.venv\Scripts\python.exe -m src.main --track t05 --output out\poc_v2 --run-name poc --offline --seed 42
.\.venv\Scripts\python.exe -m pytest -o addopts='' -q -rs
```

Lần kiểm thử phục vụ báo cáo: **217 passed, 39 skipped trong 69.69 giây**. Toàn bộ
39 mục bỏ qua cần GT chưa cài, không tính thành kiểm chứng thành công. Bao gồm
kiểm provenance, chuẩn hóa cặp, cạnh cầu, giữ nhóm, crop/bẫy màu, review/audit,
fingerprint cũ, chặn mạng/GT và lỗi đầu vào. Kiểm thử tự sinh dữ liệu tạm, không
sửa fixture giảng viên. Mã hiện tại ở commit `69301c...`; các thay đổi của tác vụ
này chỉ là báo cáo. Không sửa thuật toán, ngưỡng, bộ chấm hoặc kho nguồn.

Phiên bản thư viện đã đối chiếu trong môi trường chạy: NumPy 2.5.4, pandas 3.1.1,
Pillow 13.0.0, OpenCV headless 5.0.0.93, SciPy 1.18.1 và pytest 9.2.0. Đây là
môi trường kiểm chứng cụ thể, không bảo đảm mọi tổ hợp phiên bản trong dải hỗ trợ.

## 09 · Giới hạn và đạo đức

- Chưa có chất lượng theo GT; F1=1 trên bộ tổng hợp không thay điểm chính thức.
- V2 chỉ thử crop tâm tối đa 12% mỗi phía và hệ số sáng có giới hạn; còn yếu với
  crop lệch tâm, xoay, phối cảnh, che khuất, clipping và vùng ít chi tiết.
- Cảnh rất giống vẫn có thể khớp RGB; metadata camera/thời gian có thể sai.
- Toàn cặp có chi phí bậc hai: tăng 10 lần ảnh tạo khoảng 100 lần cặp.
- Chưa có embedding cục bộ; hàng đợi B=20 chưa phủ đủ lỗi.
- Ví dụ lỗi ở mục 06 là dữ liệu tự sinh, chưa xác nhận cặp bẫy cụ thể của fixture
  theo nhãn cảnh thật. Cần người duyệt minh họa, không suy nhãn từ thứ tự ID.
- Ảnh thật cần quyền sử dụng, giấy phép, mục đích, thời gian lưu, kiểm soát truy
  cập và bảo vệ PII; CC0 của fixture tổng hợp không giải quyết mọi vấn đề dữ liệu thật.
- Không tuyên bố đáp ứng GDPR/pháp luật Việt Nam từ GO/checklist; cần DPO/pháp chế.

Tiếp theo: duyệt thật, GT chính thức, dữ liệu phát triển đại diện hơn, kiểm crop
khó, đo độ phủ lỗi theo B và truy hồi ứng viên để giảm chi phí.

## 10 · Đóng góp và minh chứng Git

| Thành viên | Tên/email Git | Phạm vi minh chứng | Commit | Trình bày? |
|---|---|---|---|---|
| Nguyen Hai Dang | `44244432664 <haidang453@gmail.com>` | Nhập/tích hợp T05: cấu hình, pipeline, module, công cụ, kiểm thử và báo cáo; không khẳng định tự viết toàn bộ mã chuyển từ kho nguồn | `69301c20612c98130438607c1d81ece81457fb3c` | Chưa xác nhận |
| Thành viên khác | Chưa cung cấp | Không tự suy diễn | Chưa có minh chứng để điền | Chưa xác nhận |

`git show --stat 69301c...` xác nhận 32 tệp thay đổi. `git shortlog -sne --all`
ghi 1 commit của `44244432664`, 6 của `loversky02`, 4 của `Tin Duong`; hai tác giả
sau là upstream, không tính thành đóng góp mới của nhóm. CSV giữ nguyên tên cột
của mẫu, mô tả bằng tiếng Việt. Cần danh sách đầy đủ trước khi kết luận mỗi thành
viên đều có commit thực chất.

## 11 · Tài liệu tham khảo

| URL | Ngày xem | Ý nghĩa đối với thiết kế |
|---|---|---|
| <https://scikit-learn.org/stable/common_pitfalls.html#data-leakage> | 15/11/2026 | Giải thích rò rỉ gây đánh giá lạc quan; cơ sở tách thông tin tập kiểm tra và không tune theo kết quả chấm. |
| <https://creativecommons.org/publicdomain/zero/1.0/> | 15/11/2026 | Đối chiếu mã giấy phép manifest; CC0 không tự loại bỏ ràng buộc quyền riêng tư hoặc thay việc xác minh quyền sử dụng. |

## 12 · Danh sách kiểm tra trước khi nộp

- [x] Điền kho đích, chủ sở hữu, T05 và commit triển khai có thật.
- [x] Có lệnh baseline/PoC ngoại tuyến, seed 42, kiểm thử và kiểm manifest.
- [x] Có cặp/split/chẩn đoán và nêu giới hạn GO cấu trúc.
- [x] Phân tích lỗi, cải thiện v2; tách nội bộ khỏi điểm chính thức.
- [x] Ghi nhận commit thực chất của thành viên đã xác nhận và SHA trong CSV.
- [ ] Xác nhận đủ danh sách nhóm, vai trò, người trình bày; đối chiếu commit từng người theo yêu cầu lớp.
- [ ] Có người duyệt thật và minh chứng phê duyệt/audit tương ứng.
- [ ] Có GT giảng viên; sinh metrics và bảng so sánh bằng script.
- [ ] Tạo/đối chiếu `demo-180m`, `final-24h` tại các commit được chấp thuận.

Mục chưa đánh dấu còn cần bổ sung trước khi xác nhận nộp hoàn tất. `DOC/`,
`PLAN/`, `TASK/`, `docs/T05_*.md` và bản khóa môi trường bổ sung chỉ lưu local theo
`.gitignore`. Nội dung bản nộp nằm ở README, báo cáo, metric contract, mã/kiểm thử.
Không thay bất kỳ trường hay nội dung mặc định nào trong kho nguồn.
