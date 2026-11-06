# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python 3.10 / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

---

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. **Chép record hạng 1:** `class_id`= 468 ; `class_name`= cab ; `rank`= 1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. **Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?**  
   - Record này nói về **cả bức ảnh**.  
   - *Vì:* Tác vụ phân loại ảnh (image classification) gán nhãn ở cấp độ toàn bức ảnh (image-level) để phản ánh nội dung tổng thể, hoàn toàn không có tọa độ bounding box (`bbox_xyxy`) hay mặt nạ (`polygon_xy`) để định vị bất kỳ vật thể riêng lẻ nào.
3. **Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?**  
   - Danh sách tên lớp **đi kèm taxonomy/checkpoint** (cụ thể ở đây là taxonomy ImageNet-1K gồm 1000 lớp cố định). Model chỉ tính toán phân phối xác suất trên danh sách lớp đã định sẵn này chứ không thể tự sinh ra lớp mới ngoài tập dữ liệu huấn luyện.
4. **Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**  
   - Cần ghi đầy đủ cả ba trường để đảm bảo tính định danh chính xác và truy nguyên nguồn gốc: cùng một `class_name` có thể có ý nghĩa hoặc mang ID khác nhau ở các bộ taxonomy khác nhau (ví dụ: COCO vs ImageNet); `taxonomy_name` xác định không gian nhãn chuẩn, còn `class_id` giúp việc lập trình và xử lý logic không bị phụ thuộc vào chuỗi ký tự hay định dạng chữ hoa/thường.
5. **Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**  
   - Guideline cần quy định rõ **tiêu chí ưu tiên chọn nhãn chính** (salient object rule): ưu tiên đối tượng chiếm diện tích lớn nhất ở tiền cảnh, đối tượng là trọng tâm hành động của bức ảnh, hoặc bảng phân cấp ưu tiên (class priority hierarchy) giữa các nhóm đối tượng.
6. **Vì sao không được lấy `score` của model làm ground truth?**  
   - Vì `score` chỉ phản ánh mức độ tự tin (confidence score) theo phỏng đoán thống kê của mô hình toán học tại thời điểm suy luận; mô hình vẫn có thể đoán sai (dương tính giả / âm tính giả). Ground truth bắt buộc phải là dữ liệu chuẩn xác được con người kiểm chứng và gắn nhãn theo quy tắc chuẩn.

---

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. **Chép một record:** `class_name`= person ; `score`= 0.91 ; `bbox_xyxy`= [383.0, 72.0, 497.0, 412.0] ; `bbox_width`= 114.0 ; `bbox_height`= 340.0
2. **Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?**  
   - Góc trên-trái của hộp bao quanh người đầu bếp nằm tại tọa độ $(x_1 = 383.0\text{ px}, y_1 = 72.0\text{ px})$, góc dưới-phải nằm tại tọa độ $(x_2 = 497.0\text{ px}, y_2 = 412.0\text{ px})$. Chiều rộng của hộp là $114.0\text{ pixel}$ và chiều cao là $340.0\text{ pixel}$.
3. **Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**  
   - Ngưỡng `0.20`: có khoảng 18 prediction (giữ lại nhiều hộp hơn, bao gồm cả các dự đoán có độ tin cậy thấp).
   - Ngưỡng `0.60`: có khoảng 7 prediction (chỉ giữ lại những đối tượng có độ tin cậy cao, loại bỏ hầu hết các box gây nhiễu).
4. **Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**  
   - Reviewer phải tốn nhiều thời gian hơn để **rà soát và xóa bỏ các dương tính giả (false positives)**, lọc bỏ các box rác hoặc box bị gán sai nhãn, đồng thời phải điều chỉnh lại ranh giới box nếu model vẽ lệch.
5. **Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này):**  
   - *"Hộp bao (bounding box) phải bao trọn các điểm cực viền ngoài cùng của vật thể, ôm sát mép ngoài (tight bounding box) và không để khoảng trống thừa quá 2 pixel; nếu vật thể bị che khuất một phần vẫn phải vẽ bao hết toàn bộ phần nhìn thấy được."*
6. **Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**  
   - *Vật thể:* Chiếc chảo đồng treo sau lưng người đầu bếp bị che khuất phần lớn bởi thân người.
   - *Xử lý:* **Viết thành quy tắc trong guideline**.
   - *Vì sao:* Hiện tượng che khuất (occlusion) và cắt mép (truncation) diễn ra rất thường xuyên và có tính quy luật trong thị giác máy tính. Guideline cần chuẩn hóa bằng quy tắc cụ thể (ví dụ: "chỉ gán nhãn nếu phần nhìn thấy được lớn hơn 20% và vẽ box bám sát phần nhìn thấy") để annotator thực hiện đồng bộ mà không cần dừng lại hỏi từng trường hợp thông thường.

---

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. **Chép một record:**  
   - `instance_id`= traffic-001 *(hoặc kitchen-001)*  
   - `class_name`= person  
   - `score`= 0.90  
   - `polygon_point_count`= 108  
   - 3–5 điểm đầu của `polygon_xy`= `[[386.0, 75.0], [384.0, 77.0], [383.0, 80.0], [383.0, 110.0]]`
2. **So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**  
   - Polygon cung cấp chi tiết đường viền thực tế và hình học chính xác (contour, ranh giới pixel) của đối tượng; loại bỏ hoàn toàn các vùng không gian trống và phần nền (background) nằm lọt trong góc hộp chữ nhật mà bounding box bắt buộc phải chứa.
3. **`instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?**  
   - `instance_id` dùng để phân biệt **từng cá thể/thực thể riêng biệt trong cùng một bức ảnh** (ví dụ: người 1 khác với người 2). Nó không phải `class_id` vì không đại diện cho danh mục loài/lớp chung; và nó không phải `tracking_id` vì nó chỉ có giá trị nội bộ trong một bức ảnh tĩnh duy nhất, không liên kết theo dõi đối tượng xuyên suốt qua chuỗi video.
4. **Viết một quy tắc vẽ biên mask để người gán nhãn làm theo:**  
   - *"Đường biên đa giác (polygon) phải bám khít theo đường viền thực tế của đối tượng, sai số cho phép không quá 2 pixel; không cắt phạm vào bên trong vật thể và không được bao gồm phần nền hoặc vật thể khác."*
5. **Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**  
   - *Vị trí:* Vùng tiếp giáp giữa mặt bàn gỗ (`dining table`) với các vật dụng đặt đè lên trên (khăn trải bàn, bát đĩa).
   - *Xử lý:* **Dùng guideline**.
   - *Vì sao:* Tình huống vật thể lồng ghép/đè lên nhau (nesting) là kịch bản chuẩn trong nhà bếp; guideline cần nêu rõ nguyên tắc tách biệt (mặt bàn chỉ lấy phần diện tích nhìn thấy được hay ước lượng toàn phần) để người dán nhãn áp dụng nhất quán.

---

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | 1 nhãn duy nhất cho toàn bức ảnh: `class_id`, `class_name` (ví dụ: `cab`). | Ảnh có nhiều xe cộ lẫn lộn (bus, car, cab); model dự đoán nhãn `cab` với score chỉ 0.510915, khó xác định đối tượng áp đảo. | Đọc guideline, xác định đối tượng nổi bật nhất theo quy tắc ưu tiên và chọn 1 nhãn lớp duy nhất. | Kiểm tra xem nhãn được chọn có nằm đúng taxonomy và tuân thủ đúng quy tắc ưu tiên đối tượng hay không. |
| **Phát hiện vật thể** | Tập hợp các Bounding Box: tọa độ `[x1, y1, x2, y2]` kèm `class_name` cho từng vật thể. | Các chảo đồng treo bị che khuất chồng chéo sau lưng người; model bỏ sót hoặc vẽ hộp lồng nhau gây nhầm lẫn class. | Vẽ box bao khít từng đối tượng riêng lẻ theo đúng class quy định; xử lý trường hợp bị che theo guideline. | Soi độ khít (tightness) của box, kiểm tra sót vật thể (FN) hoặc gán nhầm nhãn / box thừa (FP). |
| **Instance segmentation** | Tập hợp các mặt nạ đa giác (`polygon_xy`) kèm `instance_id` và `class_name` cho từng thực thể. | Vùng tiếp giáp phức tạp giữa người và các dụng cụ treo phía sau; vùng chân và bóng tối gầm bàn khó phân biệt pixel biên. | Dùng công cụ polygon vẽ bám sát viền từng đối tượng thực tế, tách rời các thực thể chạm nhau. | Phóng to kiểm tra độ chính xác từng pixel biên viền (under/over-segmentation) và xem các instance có bị dính chùm hay không. |

---

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu):**  
   - Tuyệt đối không đưa bất kỳ thông tin nhận dạng cá nhân (PII), thông tin nhạy cảm hay dữ liệu riêng tư vào trong file báo cáo, mã nguồn và dữ liệu đầu ra của bài thực hành.
2. **Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**  
   - Dừng ngay thao tác xử lý trên mẫu dữ liệu đó và thông báo ngay lập tức cho **Lab Coach / Giảng viên hướng dẫn**.

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
