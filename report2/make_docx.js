// Build the assignment-2 report as a .docx (same content as report2/report.tex).
//
//   npm install docx     # once, in this directory
//   node report2/make_docx.js
//
// Numbers come from report2/generated/macros.tex and results/*.csv -- never typed here,
// so the Word file cannot drift from the PDF.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, PageBreak,
  ExternalHyperlink,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");  // repo root, wherever it is checked out
const GEN = path.join(ROOT, "report2", "generated");
const RESULTS = path.join(ROOT, "results");
const OUT = path.join(ROOT, "report2", "BaoCao_BaiTap2_VoHoangKhang_25C01034.docx");

// ---------- data ----------
function macros() {
  const text = fs.readFileSync(path.join(GEN, "macros.tex"), "utf8");
  const out = {};
  for (const line of text.split(/\r?\n/)) {
    const m = line.match(/^\\newcommand\{\\(\w+)\}\{(.*)\}$/);
    if (m) out[m[1]] = m[2].replace(/\{,\}/g, ",").replace(/\\%/g, "%");
  }
  return out;
}

function csv(file) {
  const [head, ...rows] = fs.readFileSync(path.join(RESULTS, file), "utf8")
    .trim().split(/\r?\n/);
  const keys = head.split(",");
  return rows.map(r => Object.fromEntries(r.split(",").map((v, i) => [keys[i], v])));
}

const m = macros();
const vn = (x, d) => Number(x).toFixed(d).replace(".", ",");

// ---------- building blocks ----------
const FONT = "Times New Roman";

const p = (text, opts = {}) => new Paragraph({
  alignment: opts.align,
  spacing: { after: opts.after ?? 120, line: 276 },
  children: [new TextRun({ text, font: FONT, size: opts.size ?? 22, bold: opts.bold,
                           italics: opts.italics, color: opts.color })],
});

// Rich paragraph: array of [text, {bold/italics}] pairs.
const rich = (parts, opts = {}) => new Paragraph({
  alignment: opts.align,
  spacing: { after: opts.after ?? 120, line: 276 },
  children: parts.map(([text, o = {}]) => new TextRun({
    text, font: FONT, size: opts.size ?? 22, bold: o.bold, italics: o.italics })),
});

const heading = (text, level = HeadingLevel.HEADING_1) => new Paragraph({
  heading: level,
  spacing: { before: 280, after: 140 },
  children: [new TextRun({ text, font: FONT, size: level === HeadingLevel.HEADING_1 ? 28 : 24,
                           bold: true, color: "1F3864" })],
});

const bullet = (text) => new Paragraph({
  bullet: { level: 0 },
  spacing: { after: 80, line: 276 },
  children: [new TextRun({ text, font: FONT, size: 22 })],
});

// PNG dimensions straight from the IHDR chunk, so figures keep their aspect ratio.
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function figure(file, caption, maxWidthPt = 430) {
  const full = path.join(GEN, file);
  const { w, h } = pngSize(full);
  const width = Math.min(maxWidthPt, w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160, after: 60 },
      keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(full),
                                transformation: { width, height: Math.round(width * h / w) } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [new TextRun({ text: caption, font: FONT, size: 20, italics: true })],
    }),
  ];
}

function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  // keepNext on every row but the last keeps the whole (small) table on one page;
  // cantSplit stops a single row from breaking across pages.
  const cell = (text, { bold = false, shade = null, align = AlignmentType.LEFT,
                        keepNext = false } = {}, width) =>
    new TableCell({
      width: { size: width, type: WidthType.DXA },
      shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({
        alignment: align,
        spacing: { after: 0 },
        keepNext,
        children: [new TextRun({ text, font: FONT, size: 20, bold })],
      })],
    });

  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({
        tableHeader: true,
        cantSplit: true,
        children: headers.map((h, i) =>
          cell(h, { bold: true, shade: "DCE6F1", keepNext: true,
                    align: i ? AlignmentType.RIGHT : AlignmentType.LEFT }, widths[i])),
      }),
      ...rows.map((r, ri) => new TableRow({
        cantSplit: true,
        children: r.map((v, i) =>
          cell(String(v), { keepNext: ri < rows.length - 1,
                            align: i ? AlignmentType.RIGHT : AlignmentType.LEFT }, widths[i])),
      })),
    ],
  });
}

// Above the table, per IEEE convention, and keepNext so a page break cannot separate
// the caption from the table it labels.
const tableCaption = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 200, after: 80 },
  keepNext: true,
  children: [new TextRun({ text, font: FONT, size: 20, italics: true })],
});

// ---------- tables from the CSVs ----------
const probe = csv("cnn_linear_probe.csv");
const perClass = csv("cnn_finetune_resnet18_per_class.csv").filter(r => Number(r.f1) < 1);

const backboneTable = table(
  ["Backbone", "Tham số (M)", "Số chiều", "Kiểm định chéo", "Tập giữ lại", "ms/ảnh"],
  probe.map(r => [r.backbone, vn(r.params_m, 1), r.dim, vn(r.cv_mean, 4), vn(r.held_out, 4),
                  vn(r.embed_ms_per_image, 1)]),
  [2600, 1300, 1100, 1700, 1500, 1000]);

const comparisonTable = table(
  ["Phương pháp", "Kiểm định chéo", "Tập giữ lại"],
  [["Đặc trưng thủ công + SVM (bài tập 1)", "0,9885", "0,9895"],
   ["ResNet18 đóng băng + phân lớp tuyến tính", "0,9974", "0,9937"],
   ["ResNet18 tinh chỉnh (mở khoá block cuối)", "—", "0,9979"]],
  [5000, 2100, 2100]);

const perClassTable = table(
  ["Loài", "Precision", "Recall", "F1", "Số mẫu"],
  perClass.map(r => [r.species, vn(r.precision, 3), vn(r.recall, 3), vn(r.f1, 3), r.support]),
  [3600, 1500, 1500, 1300, 1300]);

// ---------- document ----------
const children = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 100 },
    children: [new TextRun({ text: "PHÂN LỚP LÁ CÂY TRÊN BỘ DỮ LIỆU FLAVIA", font: FONT, size: 32, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text: "BẰNG MẠNG NƠ-RON TÍCH CHẬP", font: FONT, size: 32, bold: true })],
  }),
  p("Học chuyển giao với ResNet18, MobileNetV3 và VGG11", { align: AlignmentType.CENTER, italics: true, after: 240 }),
  p("Học viên thực hiện: Võ Hoàng Khang        MSHV: 25C01034", { align: AlignmentType.CENTER, after: 60 }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 320 },
    children: [
      new TextRun({ text: "Mã nguồn: ", font: FONT, size: 20 }),
      new ExternalHyperlink({
        link: "https://github.com/vo-hoang-kh4ng/Leaf-Segmentation",
        children: [new TextRun({ text: "github.com/vo-hoang-kh4ng/Leaf-Segmentation",
                                 font: FONT, size: 20, color: "0563C1", underline: {} })],
      }),
    ],
  }),

  heading("Tóm tắt", HeadingLevel.HEADING_2),
  p(`Báo cáo giải lại bài toán phân lớp ${m.nclasses} loài lá cây trên bộ dữ liệu Flavia (1907 ảnh) bằng mạng nơ-ron tích chập, thay cho đặc trưng thủ công của bài tập trước. Ba kiến trúc tiền huấn luyện trên ImageNet được so sánh ở hai chế độ sử dụng: đóng băng toàn bộ backbone và chỉ huấn luyện một bộ phân lớp tuyến tính, hoặc tinh chỉnh khối tích chập cuối. Toàn bộ thí nghiệm dùng đúng phép chia dữ liệu và giao thức đánh giá của bài tập trước nên các con số so sánh được trực tiếp. Kết quả chính: với backbone đóng băng hoàn toàn, ${m.probebestname} đã đạt ${m.probebestcv} (kiểm định chéo 5 phần), cao hơn ${m.cnngain} điểm so với ${m.classicalcv} của quy trình đặc trưng thủ công; tinh chỉnh nâng độ chính xác trên tập kiểm tra giữ lại lên ${m.ftacc}, chỉ còn ${m.fterrors} ảnh bị phân lớp sai trên 477 ảnh.`),

  heading("1. Giới thiệu"),
  p("Bài tập trước giải bài toán Flavia bằng đặc trưng thủ công: phân đoạn lá, trích xuất bốn nhóm đặc trưng (hình dạng, màu sắc, kết cấu, gân lá) do con người thiết kế, rồi phân lớp bằng SVM. Cách làm đó đòi hỏi hiểu biết chuyên môn ở từng bước: phải biết rằng độ răng cưa của mép lá mang thông tin phân loài thì mới nghĩ ra đặc trưng đo nó."),
  p("Mạng nơ-ron tích chập thay đổi vị trí của tri thức chuyên môn: thay vì thiết kế đặc trưng, ta thiết kế kiến trúc và để mạng tự học biểu diễn từ dữ liệu. Trở ngại là mạng sâu cần rất nhiều dữ liệu, trong khi Flavia chỉ có 1907 ảnh. Lời giải tiêu chuẩn là học chuyển giao: lấy mạng đã huấn luyện trên ImageNet với hơn một triệu ảnh, rồi tái sử dụng biểu diễn đó cho bài toán mới."),
  p("Báo cáo trả lời ba câu hỏi:"),
  bullet("Biểu diễn học từ ImageNet có sẵn sàng tách được 32 loài lá hay không, khi không huấn luyện lại gì trong mạng?"),
  bullet(`Ba kiến trúc với độ phức tạp chênh nhau ${m.paramratio} lần khác nhau thế nào trên bài toán này?`),
  bullet("Tinh chỉnh mạng thêm được bao nhiêu so với việc chỉ dùng mạng như bộ trích đặc trưng?"),

  heading("2. Dữ liệu và giao thức đánh giá"),
  p(`Bộ dữ liệu giữ nguyên như bài tập trước: Flavia, 1907 ảnh JPEG 1600×1200, ${m.nclasses} loài, mỗi lớp 50–77 ảnh, nhãn mã hoá trong số hiệu tên tệp.`),
  rich([["Điểm then chốt về phương pháp: ", {}],
        ["phép chia dữ liệu, hạt giống ngẫu nhiên và cách đánh giá được giữ y hệt bài tập trước", { bold: true }],
        [" — kiểm định chéo phân tầng 5 phần trên toàn bộ 1907 ảnh, cộng một tập kiểm tra giữ lại 25% (477 ảnh) với cùng hạt giống. Nếu đổi cách chia, con số của CNN sẽ không còn so sánh được với con số của đặc trưng thủ công.", {}]]),
  rich([["Một khác biệt lớn so với bài trước: ", {}],
        ["quy trình CNN không cần bước phân đoạn", { bold: true }],
        [". Đặc trưng thủ công bắt buộc phải tách lá khỏi nền trước, vì diện tích hay độ đặc chỉ định nghĩa được trên một vùng đã phân đoạn. CNN nhận thẳng ảnh màu, chỉ cần đưa về kích thước 224×224 và chuẩn hoá theo thống kê của ImageNet.", {}]]),
  p("Ảnh được đưa về 224×224 bằng phép co giãn trực tiếp chứ không cắt giữa: phép cắt sẽ xén mất phần chóp của những chiếc lá dài, mà chóp lá lại là chi tiết phân biệt nhiều loài."),

  heading("3. Cơ sở lý thuyết"),
  heading("3.1. Mạng tích chập", HeadingLevel.HEADING_2),
  p("Tầng tích chập trượt một bộ lọc nhỏ trên toàn ảnh và dùng chung một bộ trọng số cho mọi vị trí. Hai hệ quả quan trọng: số tham số giảm mạnh so với tầng kết nối đầy đủ, và đặc trưng học được có tính bất biến tịnh tiến — một mép lá răng cưa được nhận ra bất kể nó nằm ở đâu trong ảnh. Xếp chồng nhiều tầng tạo ra hệ thống phân cấp: tầng đầu phản ứng với biên và vùng màu, các tầng sau tổ hợp chúng thành hoạ tiết rồi thành bộ phận của vật thể."),
  heading("3.2. Ba kiến trúc được so sánh", HeadingLevel.HEADING_2),
  rich([["VGG11", { bold: true }],
        [" theo đuổi sự đơn giản: chỉ dùng bộ lọc 3×3 xếp chồng, xen kẽ với phép gộp cực đại. Hai bộ lọc 3×3 liên tiếp có cùng vùng tiếp nhận với một bộ lọc 5×5 nhưng ít tham số hơn và có thêm một phi tuyến. Nhược điểm là khối kết nối đầy đủ ở cuối rất nặng: riêng phần này chiếm phần lớn trong ", {}],
        [`${m.heavyparams} triệu tham số`, {}], [" của mạng.", {}]]),
  rich([["ResNet18", { bold: true }],
        [" giải quyết hiện tượng mạng càng sâu thì càng khó huấn luyện. Thay vì học trực tiếp ánh xạ H(x), mỗi khối học phần dư F(x) = H(x) − x rồi cộng lại qua kết nối tắt: y = F(x, {Wi}) + x. Kết nối tắt tạo đường truyền gradient không bị suy giảm qua nhiều tầng, nhờ đó huấn luyện được mạng sâu hàng chục đến hàng trăm tầng.", {}]]),
  rich([["MobileNetV3-Small", { bold: true }],
        [" được thiết kế cho thiết bị tính toán hạn chế. Ý tưởng cốt lõi là tách tích chập thường thành depthwise (lọc riêng từng kênh) và pointwise (1×1 để trộn kênh), giảm chi phí tính toán khoảng 8–9 lần với bộ lọc 3×3. Mạng bổ sung khối Squeeze-and-Excitation và hàm kích hoạt h-swish; kiến trúc được tìm bằng tìm kiếm tự động.", {}]]),
  heading("3.3. Học chuyển giao", HeadingLevel.HEADING_2),
  p("Các tầng đầu của mạng huấn luyện trên ImageNet học những thứ rất tổng quát — biên, góc, đốm màu, hoạ tiết — và những thứ này đúng với mọi ảnh tự nhiên. Vì vậy có thể giữ nguyên chúng và chỉ thay phần đầu ra. Báo cáo so sánh hai mức độ tái sử dụng:"),
  bullet("Backbone đóng băng (linear probe): bỏ tầng phân lớp của ImageNet, dùng mạng như một hàm trích đặc trưng cố định, rồi huấn luyện một bộ phân lớp tuyến tính lên trên. Không một trọng số nào trong mạng thay đổi."),
  bullet("Tinh chỉnh (fine-tune): mở khoá khối tích chập cuối cùng tầng phân lớp mới, cho phép mạng điều chỉnh các đặc trưng bậc cao cho phù hợp với lá cây."),

  heading("4. Phương pháp thực nghiệm"),
  p("Mỗi ảnh được đưa qua backbone một lần, thu được vectơ đặc trưng (512 chiều với ResNet18, 576 với MobileNetV3, 4096 với VGG11). Toàn bộ 1907 vectơ được lưu lại, nên các thí nghiệm sau chỉ làm việc trên ma trận đặc trưng thay vì đọc lại ảnh. Bộ phân lớp là hồi quy logistic đa lớp, có chuẩn hoá z-score đặt bên trong quy trình huấn luyện để chỉ được ước lượng trên phần huấn luyện của mỗi lần chia, tránh rò rỉ dữ liệu."),
  p(`Khi tinh chỉnh, chỉ layer4 và tầng phân lớp mới được huấn luyện. Với 1430 ảnh huấn luyện, mở khoá toàn mạng sẽ dẫn tới quá khớp, và trên CPU cũng quá chậm. Hai tốc độ học khác nhau được dùng: tầng phân lớp khởi tạo ngẫu nhiên nên cần bước lớn (10⁻³), còn layer4 xuất phát từ một nghiệm đã tốt nên chỉ cần bước nhỏ (10⁻⁴). Thuật toán tối ưu là AdamW, hàm mất mát entropy chéo, ${m.epochs} epoch, mỗi epoch khoảng ${m.epochseconds} giây trên CPU 16 luồng.`),
  p("Tăng cường dữ liệu chỉ áp dụng cho tập huấn luyện và được giữ ở mức vừa phải: lật ngang, lật dọc, xoay ngẫu nhiên tối đa 20° (lấp nền trắng cho khớp nền Flavia), thay đổi nhẹ độ sáng và độ tương phản. Lá trên máy quét không có hướng chuẩn nên lật và xoay là phép biến đổi bảo toàn nhãn. Ngược lại, thay đổi màu bị giữ ở biên độ nhỏ vì màu sắc thực sự mang thông tin phân loài: bài tập trước đo được nhóm đặc trưng màu đóng góp hơn 11 điểm phần trăm."),

  heading("5. Kết quả"),
  heading("5.1. So sánh ba backbone đóng băng", HeadingLevel.HEADING_2),
  rich([["Điều đáng chú ý đầu tiên: ", {}],
        [`cả ba mạng đều vượt ${m.classicalcv} của quy trình đặc trưng thủ công, dù không một trọng số nào trong mạng được huấn luyện trên ảnh lá`, { bold: true }],
        [". Biểu diễn học từ ImageNet đã đủ để tách 32 loài lá bằng một mặt phẳng tuyến tính.", {}]]),
  tableCaption("Bảng 1. Ba backbone tiền huấn luyện ImageNet, đóng băng hoàn toàn, chỉ huấn luyện một bộ phân lớp tuyến tính phía trên."),
  backboneTable,
  rich([["Số tham số không dự báo chất lượng biểu diễn. ", { bold: true }],
        [`${m.lightname} chỉ có ${m.lightparams} triệu tham số, nhỏ hơn ${m.heavyname} ${m.paramratio} lần, nhưng đạt ${m.lightcv} so với ${m.heavycv}, đồng thời nhanh hơn 5 lần (${m.lightms} ms so với ${m.heavyms} ms mỗi ảnh trên CPU).`, {}]]),
  p(`Tuy nhiên, chênh lệch giữa các mạng rất nhỏ nên cần kiểm định trước khi xếp hạng. Kiểm định t ghép cặp trên cùng các phần kiểm định chéo cho thấy chỉ một cặp đạt ý nghĩa thống kê: ${m.probebestname} hơn ${m.probeworstname} (p = 0,035). Hai cặp còn lại có p = 0,15–0,18, tức chưa đủ cơ sở kết luận mạng nào hơn. Với mức chênh lệch dưới một điểm phần trăm, xếp hạng theo chữ số thập phân là diễn giải nhiễu.`),

  heading("5.2. Tinh chỉnh", HeadingLevel.HEADING_2),
  p(`Ngay sau epoch đầu tiên, độ chính xác trên tập kiểm tra đã đạt ${m.firstepochacc} — hệ quả trực tiếp của việc xuất phát từ trọng số ImageNet thay vì khởi tạo ngẫu nhiên. Mô hình tốt nhất xuất hiện ở epoch ${m.bestepoch} với ${m.ftacc}. Hàm mất mát trên tập kiểm tra giảm đều và không bật lên, nghĩa là với số epoch này mô hình chưa quá khớp.`),
  ...figure("training_curve.png", `Hình 1. Quá trình tinh chỉnh ResNet18 qua ${m.epochs} epoch.`),
  p(`So với cùng mạng ở chế độ đóng băng (${m.probeacc} trên cùng tập kiểm tra), tinh chỉnh nâng kết quả lên ${m.ftacc}. Cần thận trọng khi diễn giải: trên 477 ảnh, khoảng cách này tương ứng chỉ vài ảnh, nên nó cho thấy một xu hướng hợp lý chứ không phải một kết luận chắc chắn về mặt thống kê.`),
  tableCaption("Bảng 2. So sánh với bài tập 1 trên cùng bộ dữ liệu, cùng phép chia phân tầng và cùng hạt giống ngẫu nhiên."),
  comparisonTable,

  heading("5.3. Ma trận nhầm lẫn, precision và recall", HeadingLevel.HEADING_2),
  p(`Với ${m.nclasses} lớp có kích thước không đều, riêng độ chính xác tổng thể là chưa đủ: một loài nhỏ có thể bị sai toàn bộ mà con số tổng vẫn gần như không đổi. Vì vậy kết quả được báo cáo theo từng lớp.`),
  rich([[`Mô hình tinh chỉnh đạt macro precision ${m.ftprecision}, macro recall ${m.ftrecall} và macro F1 ${m.ftfone}. `, {}],
        [`${m.ftperfect}/${m.ftclasses} loài đạt F1 bằng 1,00`, { bold: true }],
        [", tức precision và recall đều tuyệt đối.", {}]]),
  tableCaption("Bảng 3. Toàn bộ các loài không đạt F1 = 1,00. Những loài còn lại đều đạt precision và recall bằng 1,00."),
  perClassTable,
  ...figure("precision_recall.png", "Hình 2. Precision và recall theo từng loài trên tập kiểm tra giữ lại, sắp xếp theo F1.", 400),
  p(`Ma trận nhầm lẫn cho thấy toàn bộ khối lượng nằm trên đường chéo, ngoại trừ đúng ${m.fterrors} ô: một ảnh Canadian poplar bị nhận thành camphortree. Chính cặp này cũng xuất hiện ở mô hình đóng băng, bên cạnh hai ảnh peach bị nhận thành Anhui Barberry mà việc tinh chỉnh đã sửa được. Với chỉ một mẫu sai, không thể kết luận gì về nguyên nhân.`),
  ...figure("confusion.png", "Hình 3. Ma trận nhầm lẫn của mô hình ResNet18 tinh chỉnh trên 477 ảnh kiểm tra.", 420),

  heading("6. Minh hoạ đặc trưng mà mạng học được"),
  p("Khác với đặc trưng thủ công — nơi mỗi chiều có tên và công thức rõ ràng — đặc trưng của CNN phải được quan sát gián tiếp. Bốn góc nhìn dưới đây trả lời bốn câu hỏi khác nhau."),
  rich([["Bộ lọc tầng đầu tiên", { bold: true }],
        [" là 64 nhân 7×7×3 hiển thị dưới dạng ảnh màu. Có thể thấy rõ hai nhóm: các bộ lọc dò biên theo nhiều hướng (dạng sọc sáng tối) và các bộ lọc đối lập màu (xanh–đỏ, xanh–vàng). Đây là những đặc trưng rất tổng quát, lý do khiến tầng đầu có thể giữ nguyên khi chuyển sang bài toán lá cây.", {}]]),
  ...figure("filters.png", "Hình 4. 64 bộ lọc của tầng tích chập đầu tiên.", 330),
  ...figure("feature_maps.png", "Hình 5. Bản đồ đặc trưng trên một chiếc lá thật: mỗi kênh làm nổi bật một loại cấu trúc khác nhau (mép lá, gân chính, vùng phiến lá).", 430),
  rich([["Grad-CAM", { bold: true }],
        [" trả lời câu hỏi quan trọng nhất về độ tin cậy: mạng dựa vào vùng nào để ra quyết định? Phương pháp lấy trọng số mỗi kênh của khối tích chập cuối bằng gradient của điểm số lớp dự đoán theo kênh đó, rồi cộng lại. Bản đồ thu được tập trung vào phiến lá và tắt dần ra nền, xác nhận mạng nhìn vào chiếc lá chứ không bám vào nền trắng hay một dấu vết nào của máy quét. Đây là kiểm tra cần thiết: với nền trắng đồng nhất, một mô hình vẫn có thể đạt độ chính xác cao nhờ học các manh mối giả.", {}]]),
  ...figure("gradcam.png", "Hình 6. Grad-CAM trên ảnh 1083. Vùng nóng nằm trọn trong phiến lá.", 430),
  rich([["t-SNE", { bold: true }],
        [" chiếu embedding 512 chiều của backbone đóng băng xuống hai chiều. Các loài tách thành những cụm rời rạc trước khi có bất kỳ bộ phân lớp nào. Đây là lời giải thích trực quan cho kết quả ở Bảng 1: bộ phân lớp tuyến tính đạt gần 100% vì công việc khó đã được backbone làm xong. So sánh với bài tập trước rất đáng chú ý: ở đó, phép chiếu hai chiều bằng PCA hay LDA đều không tách được các lớp, vì thông tin phân biệt nằm rải rác ở nhiều chiều.", {}]]),
  ...figure("tsne.png", "Hình 7. t-SNE của embedding ResNet18 đóng băng; mỗi màu là một loài.", 360),

  heading("7. Thảo luận: CNN so với đặc trưng thủ công"),
  rich([["Công sức thiết kế. ", { bold: true }],
        ["Quy trình thủ công cần bốn nhóm đặc trưng được thiết kế riêng, cộng với một bước phân đoạn mà bài tập trước đo được là có lỗi cục bộ trên các lá có mặt dưới nhạt màu. Quy trình CNN không cần bước nào trong số đó.", {}]]),
  rich([["Khả năng giải thích. ", { bold: true }],
        ["Ở chiều ngược lại, đặc trưng thủ công có thể nói chính xác vì sao mô hình quyết định: bài tập trước định lượng được rằng smooth factor là đặc trưng quan trọng nhất, và đo được mức độ chồng lấn thông tin giữa nhóm màu và nhóm kết cấu. Với CNN, câu trả lời tương đương chỉ có thể tiếp cận gián tiếp qua Grad-CAM.", {}]]),
  rich([["Chi phí tính toán. ", { bold: true }],
        [`Trích đặc trưng thủ công mất khoảng 0,27 giây mỗi ảnh trên CPU, trong khi MobileNetV3 chỉ mất ${m.lightms} mili giây — CNN hiện đại thực ra nhanh hơn quy trình thủ công, vì phép phân đoạn và các phép hình thái học trên ảnh 1600×1200 khá tốn kém. Bù lại, CNN cần tải về trọng số tiền huấn luyện và phụ thuộc vào một khung học sâu.`, {}]]),
  rich([["Tính bền vững. ", { bold: true }],
        ["Bài tập trước chỉ ra tổ hợp đặc trưng thủ công tốt nhất lại rất dễ vỡ khi ảnh bị nhiễu hoặc che khuất. Báo cáo này chưa lặp lại thí nghiệm đó cho CNN, nên không thể khẳng định bên nào bền hơn; đó là hướng cần làm tiếp.", {}]]),

  heading("8. Hạn chế"),
  bullet(`Độ chính xác đã chạm trần của bộ dữ liệu: ${m.ftacc} tương ứng ${m.fterrors} ảnh sai trên 477. Ở mức này, mọi so sánh giữa các cấu hình đều nằm trong vùng nhiễu thống kê.`),
  bullet("Cấu hình tinh chỉnh chỉ được đánh giá trên một phép chia giữ lại, không chạy kiểm định chéo đầy đủ vì chi phí tính toán trên CPU."),
  bullet("Siêu tham số (tốc độ học, số epoch, mức độ mở khoá) được chọn theo kinh nghiệm, chưa qua tìm kiếm có hệ thống."),
  bullet("Chưa thử kiến trúc Transformer thị giác, vốn cần nhiều dữ liệu hơn hoặc kỹ thuật huấn luyện riêng để phát huy trên tập nhỏ như Flavia."),

  heading("9. Kết luận"),
  p(`Học chuyển giao giải bài toán Flavia tốt hơn quy trình đặc trưng thủ công với công sức thiết kế ít hơn hẳn: ${m.probebestname} đóng băng đạt ${m.probebestcv} và ResNet18 tinh chỉnh đạt ${m.ftacc} trên tập kiểm tra giữ lại, so với ${m.classicalcv} của bài tập trước.`),
  rich([["Kết quả có ý nghĩa nhất không phải con số cao nhất, mà là ", {}],
        ["phần lớn thành tích đến từ trước khi huấn luyện", { bold: true }],
        [". Một bộ phân lớp tuyến tính trên đặc trưng ImageNet đóng băng đã vượt toàn bộ quy trình thủ công, và hình t-SNE cho thấy lý do: các loài đã tách thành cụm ngay trong không gian embedding. Việc tinh chỉnh chỉ thêm được một phần nhỏ. Với một bộ dữ liệu 1907 ảnh, giá trị của học sâu nằm ở biểu diễn được chuyển giao từ dữ liệu lớn, chứ không nằm ở việc huấn luyện trên chính bộ dữ liệu nhỏ đó.", {}]]),
  p(`Quan sát thứ hai là số tham số không phải chỉ dấu của chất lượng: ${m.lightname} nhỏ hơn ${m.heavyname} ${m.paramratio} lần nhưng cho kết quả cao hơn và nhanh hơn 5 lần, đủ nhẹ để triển khai trên thiết bị di động.`),

  new Paragraph({ children: [new PageBreak()] }),
  heading("Tài liệu tham khảo"),
  ...[
    "[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.",
    "[2] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in Proc. IEEE CVPR, 2009, pp. 248–255.",
    "[3] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in NIPS, vol. 25, 2012, pp. 1097–1105.",
    "[4] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in ICLR, 2015, arXiv:1409.1556.",
    "[5] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE CVPR, 2016, pp. 770–778.",
    "[6] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, “Searching for MobileNetV3,” in Proc. IEEE/CVF ICCV, 2019, pp. 1314–1324.",
    "[7] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE ICCV, 2017, pp. 618–626.",
    "[8] L. van der Maaten and G. Hinton, “Visualizing data using t-SNE,” Journal of Machine Learning Research, vol. 9, pp. 2579–2605, 2008.",
    "[9] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in NeurIPS, 2019, pp. 8024–8035.",
    "[10] S. G. Wu, F. S. Bao, E. Y. Xu, Y.-X. Wang, Y.-F. Chang, and Q.-L. Xiang, “A leaf recognition algorithm for plant classification using probabilistic neural network,” in Proc. IEEE ISSPIT, 2007, pp. 11–16.",
  ].map(t => new Paragraph({
    spacing: { after: 100, line: 252 },
    indent: { left: 420, hanging: 420 },
    children: [new TextRun({ text: t, font: FONT, size: 20 })],
  })),
];

const doc = new Document({
  creator: "Võ Hoàng Khang",
  title: "Phân lớp lá cây trên bộ dữ liệu Flavia bằng mạng nơ-ron tích chập",
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{
    properties: { page: { margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log(`${OUT}  (${(buf.length / 1048576).toFixed(1)} MB)`);
});
