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

// `topOfPage` starts the figure on a fresh page, which is how Word gets the LaTeX [t]
// behaviour: the figure sits at the top and the following text fills the page under it,
// instead of the figure landing mid-page and leaving a ragged gap.
function figure(file, caption, maxWidthPt = 430, topOfPage = false) {
  const full = path.join(GEN, file);
  const { w, h } = pngSize(full);
  const width = Math.min(maxWidthPt, w);
  return [
    ...(topOfPage ? [new Paragraph({ spacing: { after: 0 }, children: [new PageBreak()] })] : []),
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
  p(`Trong bài tập này em giải lại bài toán phân lớp ${m.nclasses} loài lá cây trên bộ Flavia (1907 ảnh), lần này bằng mạng nơ-ron tích chập thay cho đặc trưng thủ công của bài trước. Em thử ba kiến trúc đã được huấn luyện sẵn trên ImageNet, mỗi kiến trúc dùng theo hai cách: đóng băng toàn bộ mạng và chỉ huấn luyện một tầng phân lớp tuyến tính, hoặc tinh chỉnh thêm khối tích chập cuối. Phép chia dữ liệu và cách đánh giá em giữ nguyên như bài trước để hai bài so sánh được với nhau. Kết quả: khi đóng băng hoàn toàn, ${m.probebestname} đạt ${m.probebestcv} theo kiểm định chéo 5 phần, cao hơn ${m.cnngain} điểm so với ${m.classicalcv} của đặc trưng thủ công. Tinh chỉnh đưa độ chính xác trên tập kiểm tra giữ lại lên ${m.ftacc}, tức chỉ sai ${m.fterrors} ảnh trong 477 ảnh. Có hai điều em thấy thú vị khi làm: mạng nhiều tham số hơn không có nghĩa là tốt hơn, và phần lớn kết quả có được là nhờ biểu diễn sẵn có từ ImageNet chứ không phải nhờ huấn luyện trên ảnh lá.`),

  heading("1. Giới thiệu"),
  p("Ở bài tập trước em làm theo hướng cổ điển: phân đoạn lá khỏi nền, tự thiết kế bốn nhóm đặc trưng (hình dạng, màu sắc, kết cấu, gân lá) rồi đưa vào SVM. Cách đó buộc em phải hiểu khá kỹ về lá cây. Ví dụ, phải biết độ răng cưa của mép lá có ích cho việc phân loài thì mới nghĩ ra được đặc trưng đo nó. Mỗi đặc trưng đều phải tự nghĩ ra và tự cài."),
  p("Mạng tích chập làm theo cách khác. Thay vì thiết kế đặc trưng, ta thiết kế kiến trúc rồi để mạng tự học biểu diễn từ dữ liệu. Vấn đề là mạng sâu cần rất nhiều dữ liệu, mà Flavia chỉ có 1907 ảnh, quá ít để huấn luyện từ đầu. Cách thường dùng trong trường hợp này là học chuyển giao: lấy mạng đã học trên ImageNet với hơn một triệu ảnh rồi tái sử dụng biểu diễn của nó."),
  p("Em muốn trả lời ba câu hỏi:"),
  bullet("Biểu diễn học từ ImageNet đã đủ để tách 32 loài lá chưa, nếu không huấn luyện lại gì trong mạng?"),
  bullet(`Ba kiến trúc chênh nhau ${m.paramratio} lần về số tham số thì khác nhau ra sao trên bài toán này?`),
  bullet("Tinh chỉnh thêm được bao nhiêu so với chỉ dùng mạng để trích đặc trưng?"),

  heading("2. Dữ liệu và cách đánh giá"),
  p(`Dữ liệu vẫn là Flavia như bài trước: 1907 ảnh JPEG 1600×1200, ${m.nclasses} loài, mỗi loài 50–77 ảnh, nhãn nằm trong số hiệu tên tệp.`),
  p("Em giữ nguyên phép chia dữ liệu, hạt giống ngẫu nhiên và cách đánh giá của bài trước: kiểm định chéo phân tầng 5 phần trên cả 1907 ảnh, cộng một tập kiểm tra giữ lại 25% gồm 477 ảnh. Nếu đổi cách chia thì con số của CNN không còn đặt cạnh con số của đặc trưng thủ công được nữa, mà so sánh hai cách làm chính là phần em quan tâm nhất ở bài này."),
  p("Một khác biệt lớn so với bài trước là quy trình CNN không cần phân đoạn. Với đặc trưng thủ công thì bắt buộc, vì diện tích hay độ đặc chỉ tính được trên một vùng đã tách khỏi nền. CNN nhận thẳng ảnh màu, em chỉ cần đưa về 224×224 và chuẩn hoá theo thống kê của ImageNet cho khớp với dữ liệu mà mạng đã học."),
  p("Em đưa ảnh về 224×224 bằng cách co giãn trực tiếp, không cắt giữa. Cắt giữa sẽ làm mất chóp của mấy chiếc lá dài, mà chóp lá lại là chỗ phân biệt được nhiều loài."),

  heading("3. Cơ sở lý thuyết"),
  heading("3.1. Mạng tích chập", HeadingLevel.HEADING_2),
  p("Tầng tích chập trượt một bộ lọc nhỏ khắp ảnh, dùng chung một bộ trọng số cho mọi vị trí. Nhờ vậy số tham số ít hơn hẳn so với tầng kết nối đầy đủ, và đặc trưng học được có tính bất biến tịnh tiến: mạng nhận ra mép lá răng cưa dù nó nằm ở đâu trong ảnh. Khi xếp chồng nhiều tầng thì các tầng đầu phản ứng với biên và vùng màu, các tầng sau ghép chúng lại thành hoạ tiết rồi thành bộ phận của vật thể. AlexNet là công trình đưa kiểu kiến trúc này thành hướng chủ đạo của thị giác máy tính."),
  heading("3.2. Ba kiến trúc em chọn", HeadingLevel.HEADING_2),
  rich([["VGG11", { italics: true }],
        [` có thiết kế đơn giản, chỉ gồm bộ lọc 3×3 xếp chồng xen kẽ với gộp cực đại. Hai bộ lọc 3×3 liên tiếp phủ cùng vùng ảnh như một bộ lọc 5×5 nhưng ít tham số hơn và có thêm một phi tuyến. Nhược điểm nằm ở khối kết nối đầy đủ phía cuối, chiếm phần lớn trong ${m.heavyparams} triệu tham số của mạng.`, {}]]),
  rich([["ResNet18", { italics: true }],
        [" sinh ra để giải quyết chuyện mạng càng sâu càng khó huấn luyện. Mỗi khối không học thẳng ánh xạ H(x) mà học phần dư F(x) = H(x) − x, rồi cộng lại qua kết nối tắt: y = F(x, {Wi}) + x. Kết nối tắt cho gradient truyền thẳng qua nhiều tầng mà không suy giảm, nhờ đó huấn luyện được mạng sâu hàng chục tầng trở lên.", {}]]),
  rich([["MobileNetV3-Small", { italics: true }],
        [" nhắm tới thiết bị yếu. Ý chính là tách tích chập thường thành hai bước: depthwise lọc riêng từng kênh, rồi pointwise 1×1 trộn các kênh với nhau. Với bộ lọc 3×3, cách này giảm chi phí tính toán khoảng 8–9 lần. Mạng còn có thêm khối Squeeze-and-Excitation và hàm kích hoạt h-swish, và kiến trúc được tìm bằng tìm kiếm tự động.", {}]]),
  heading("3.3. Học chuyển giao", HeadingLevel.HEADING_2),
  p("Các tầng đầu của mạng học trên ImageNet toàn những thứ rất chung như biên, góc, đốm màu, hoạ tiết. Những thứ này đúng với mọi ảnh tự nhiên chứ không riêng 1000 lớp của ImageNet, nên giữ nguyên được. Em thử hai mức tái sử dụng:"),
  bullet("Đóng băng backbone: bỏ tầng phân lớp của ImageNet, coi mạng như một hàm trích đặc trưng cố định, rồi huấn luyện hồi quy logistic lên trên. Không trọng số nào trong mạng thay đổi. Cách này cho biết thông tin phân loài đã nằm sẵn trong biểu diễn ImageNet chưa."),
  bullet("Tinh chỉnh: mở khoá khối tích chập cuối cùng với tầng phân lớp mới, để mạng chỉnh lại các đặc trưng bậc cao cho hợp với lá cây."),

  heading("4. Cách làm"),
  p("Với cách thứ nhất, em cho mỗi ảnh đi qua backbone một lần và lấy vectơ đặc trưng: 512 chiều với ResNet18, 576 với MobileNetV3, 4096 với VGG11. Cả 1907 vectơ được lưu lại nên các thí nghiệm sau chỉ làm việc trên ma trận đặc trưng, không phải đọc lại ảnh. Bộ phân lớp là hồi quy logistic đa lớp, có chuẩn hoá z-score đặt bên trong quy trình huấn luyện để nó chỉ được ước lượng trên phần huấn luyện của mỗi lần chia, tránh rò rỉ dữ liệu."),
  p(`Khi tinh chỉnh, em chỉ mở khoá layer4 và tầng phân lớp mới. Với 1430 ảnh huấn luyện mà mở khoá cả mạng thì chắc chắn quá khớp, chưa kể chạy trên CPU sẽ rất lâu. Em đặt hai tốc độ học khác nhau cho hai nhóm tham số: tầng phân lớp khởi tạo ngẫu nhiên nên cần bước lớn (10⁻³), còn layer4 vốn đã ở một nghiệm tốt nên chỉ cần bước nhỏ (10⁻⁴). Nếu để chung một tốc độ học thì hoặc là phá hỏng khối đã học sẵn, hoặc là tầng phân lớp học quá chậm. Em dùng AdamW, hàm mất mát entropy chéo, chạy ${m.epochs} epoch, mỗi epoch khoảng ${m.epochseconds} giây trên CPU 16 luồng.`),
  p("Tăng cường dữ liệu em chỉ áp dụng cho tập huấn luyện và để ở mức nhẹ: lật ngang, lật dọc, xoay ngẫu nhiên tối đa 20° với nền lấp trắng cho khớp nền Flavia, thay đổi chút ít độ sáng và độ tương phản. Lá đặt trên máy quét không có hướng chuẩn nên lật với xoay không làm sai nhãn. Riêng màu thì em không dám chỉnh mạnh, vì ở bài trước nhóm đặc trưng màu đóng góp tới hơn 11 điểm phần trăm. Làm méo màu nhiều sẽ xoá mất tín hiệu chứ chẳng thêm được tính bất biến nào có ích."),

  heading("5. Kết quả"),
  heading("5.1. Ba backbone đóng băng", HeadingLevel.HEADING_2),
  p(`Kết quả ở Bảng 1. Thứ làm em bất ngờ nhất là cả ba mạng đều vượt ${m.classicalcv} của đặc trưng thủ công, trong khi không một trọng số nào trong mạng được huấn luyện trên ảnh lá. Biểu diễn học từ ImageNet đã đủ để tách 32 loài lá chỉ bằng một mặt phẳng tuyến tính.`),
  tableCaption("Bảng 1. Ba backbone tiền huấn luyện ImageNet, đóng băng hoàn toàn, chỉ huấn luyện một tầng phân lớp tuyến tính phía trên."),
  backboneTable,
  p(`Chuyện thứ hai là số tham số không nói lên chất lượng biểu diễn. ${m.lightname} chỉ có ${m.lightparams} triệu tham số, nhỏ hơn ${m.heavyname} tới ${m.paramratio} lần, vậy mà đạt ${m.lightcv} so với ${m.heavycv}, lại còn nhanh hơn 5 lần (${m.lightms} ms so với ${m.heavyms} ms mỗi ảnh trên CPU). Kiến trúc mới với tích chập tách được dùng tham số hiệu quả hơn kiểu xếp chồng kết nối đầy đủ của VGG.`),
  p(`Có điều chênh lệch giữa ba mạng nhỏ quá nên em không dám xếp hạng vội. Kiểm định t ghép cặp trên cùng các phần kiểm định chéo cho thấy chỉ một cặp đạt ý nghĩa thống kê là ${m.probebestname} hơn ${m.probeworstname} với p = 0,035. Hai cặp còn lại p khoảng 0,15–0,18, chưa đủ cơ sở kết luận. Đây là bài học em rút ra từ bài tập trước: chênh nhau dưới một điểm phần trăm thì xếp hạng theo chữ số thập phân chỉ là đang diễn giải nhiễu.`),

  heading("5.2. Tinh chỉnh", HeadingLevel.HEADING_2),
  p(`Hình 1 là quá trình huấn luyện. Ngay sau epoch đầu tiên độ chính xác trên tập kiểm tra đã là ${m.firstepochacc}, điều này đến từ việc xuất phát từ trọng số ImageNet chứ không phải khởi tạo ngẫu nhiên. Mô hình tốt nhất rơi vào epoch ${m.bestepoch} với ${m.ftacc}. Hàm mất mát trên tập kiểm tra giảm đều và không bật lên, nên với số epoch này mô hình chưa quá khớp.`),
  ...figure("training_curve.png", `Hình 1. Quá trình tinh chỉnh ResNet18 qua ${m.epochs} epoch.`, 420),
  p(`So với chính mạng đó ở chế độ đóng băng (${m.probeacc} trên cùng tập kiểm tra), tinh chỉnh nâng lên ${m.ftacc}. Nhưng trên 477 ảnh thì khoảng cách này chỉ tương ứng vài ảnh, nên em xem nó là một xu hướng hợp lý thôi, chưa phải kết luận chắc chắn.`),
  tableCaption("Bảng 2. So sánh với bài tập trước, trên cùng bộ dữ liệu và cùng phép chia."),
  comparisonTable,

  heading("5.3. Ma trận nhầm lẫn, precision và recall", HeadingLevel.HEADING_2),
  p(`Với ${m.nclasses} lớp to nhỏ không đều thì nhìn mỗi độ chính xác tổng là chưa đủ. Một loài ít ảnh có thể sai sạch mà con số tổng vẫn gần như không nhúc nhích. Vì vậy em báo cáo theo từng lớp.`),
  p(`Mô hình tinh chỉnh đạt macro precision ${m.ftprecision}, macro recall ${m.ftrecall} và macro F1 ${m.ftfone}. Có ${m.ftperfect}/${m.ftclasses} loài đạt F1 bằng 1,00, tức precision và recall đều tuyệt đối. Bảng 3 liệt kê những loài còn lại, Hình 2 vẽ precision và recall của tất cả các loài.`),
  tableCaption("Bảng 3. Những loài không đạt F1 = 1,00. Các loài còn lại đều đạt precision và recall bằng 1,00."),
  perClassTable,
  ...figure("precision_recall.png", "Hình 2. Precision và recall theo từng loài trên tập kiểm tra giữ lại, sắp theo F1.", 330),
  p(`Trong ma trận nhầm lẫn ở Hình 3, gần như toàn bộ khối lượng nằm trên đường chéo, chỉ còn đúng ${m.fterrors} ô lệch ra: một ảnh Canadian poplar bị nhận thành camphortree. Cặp này cũng xuất hiện ở mô hình đóng băng, kèm theo hai ảnh peach bị nhận thành Anhui Barberry mà tinh chỉnh đã sửa được. Với đúng một mẫu sai thì em không kết luận gì về nguyên nhân; muốn biết đây là nhầm lẫn có hệ thống hay chỉ là một ảnh biên thì cần tập kiểm tra lớn hơn.`),
  ...figure("confusion.png", "Hình 3. Ma trận nhầm lẫn của ResNet18 tinh chỉnh trên 477 ảnh kiểm tra.", 340),

  heading("6. Mạng học được những gì"),
  p("Ở bài trước mỗi đặc trưng đều có tên và công thức rõ ràng. Với CNN thì không, nên em phải nhìn gián tiếp qua bốn cách sau."),
  p("Bộ lọc tầng đầu tiên gồm 64 nhân 7×7×3, hiển thị được luôn dưới dạng ảnh màu. Nhìn vào thấy hai nhóm khá rõ: nhóm dò biên theo các hướng khác nhau, trông như những vệt sáng tối, và nhóm đối lập màu kiểu xanh–đỏ hay xanh–vàng. Toàn là thứ rất chung chung, và đó cũng là lý do tầng này giữ nguyên được khi chuyển sang bài toán lá cây."),
  ...figure("filters.png", "Hình 4. 64 bộ lọc của tầng tích chập đầu tiên.", 260),
  p("Bản đồ đặc trưng cho thấy các bộ lọc đó hoạt động ra sao trên một ảnh cụ thể. Có kênh làm nổi mép lá, có kênh làm nổi gân chính, có kênh thì phản ứng với vùng phiến lá đồng đều."),
  ...figure("feature_maps.png", "Hình 5. Các bộ lọc tầng 1 phản ứng trên một chiếc lá thật.", 400),
  p("Grad-CAM là thứ em thấy cần nhất để yên tâm về mô hình, vì nó chỉ ra mạng dựa vào vùng nào khi quyết định. Cách tính là lấy gradient của điểm số lớp dự đoán theo từng kênh của khối tích chập cuối, dùng nó làm trọng số cho kênh đó rồi cộng lại. Bản đồ thu được nằm gọn trong phiến lá và nhạt dần ra nền, tức mạng nhìn vào lá thật. Em kiểm tra điều này vì nền Flavia trắng trơn và đồng nhất, mô hình hoàn toàn có thể đạt điểm cao nhờ bám vào một manh mối giả nào đó của máy quét thay vì nhìn lá."),
  ...figure("gradcam.png", "Hình 6. Grad-CAM trên ảnh 1083. Vùng nóng nằm trọn trong phiến lá.", 390),
  p("t-SNE chiếu embedding 512 chiều của backbone đóng băng xuống hai chiều. Các loài đã tách thành từng cụm riêng trước khi có bất kỳ bộ phân lớp nào. Đây chính là lý do bộ phân lớp tuyến tính ở Bảng 1 đạt gần 100%: phần khó đã được backbone làm xong rồi."),
  p("Chỗ này so với bài trước khá thú vị. Ở bài trước, chiếu hai chiều bằng PCA hay LDA đều không tách được các lớp, vì thông tin phân biệt nằm rải ở nhiều chiều. Còn với embedding của CNN thì cấu trúc cụm hiện ra ngay trên hai chiều."),
  ...figure("tsne.png", "Hình 7. t-SNE của embedding ResNet18 đóng băng, mỗi màu là một loài.", 300),

  heading("7. So sánh với cách làm ở bài trước"),
  p("Bảng 2 đặt hai cách cạnh nhau. CNN thắng về độ chính xác, nhưng nếu nhìn rộng hơn một con số thì mỗi bên có chỗ mạnh riêng."),
  p("Về công sức, cách thủ công cần bốn nhóm đặc trưng tự thiết kế, cộng thêm bước phân đoạn mà ở bài trước em đo được là có lỗi cục bộ với những chiếc lá có mặt dưới nhạt màu. Quy trình CNN bỏ được hết các bước đó."),
  p("Đổi lại, đặc trưng thủ công giải thích được rõ ràng hơn nhiều. Ở bài trước em chỉ ra được smooth factor là đặc trưng quan trọng nhất, và đo được mức chồng lấn thông tin giữa nhóm màu với nhóm kết cấu. Với CNN thì câu trả lời tương đương chỉ có thể tiếp cận gián tiếp qua Grad-CAM."),
  p(`Về tốc độ, em hơi bất ngờ: trích đặc trưng thủ công mất khoảng 0,27 giây mỗi ảnh trên CPU, còn MobileNetV3 chỉ mất ${m.lightms} mili giây. Hoá ra CNN hiện đại lại nhanh hơn, vì phân đoạn và các phép hình thái học trên ảnh 1600×1200 khá tốn kém. Bù lại CNN phải tải trọng số tiền huấn luyện và phụ thuộc vào một khung học sâu.`),
  p("Còn về độ bền thì em chưa trả lời được. Bài trước cho thấy tổ hợp đặc trưng thủ công tốt nhất lại rất dễ vỡ khi ảnh bị nhiễu hay che khuất, nhưng em chưa kịp lặp lại thí nghiệm đó cho CNN, nên không khẳng định được bên nào bền hơn."),

  heading("8. Hạn chế"),
  bullet(`Độ chính xác đã chạm trần của bộ dữ liệu. ${m.ftacc} nghĩa là sai ${m.fterrors} ảnh trên 477, ở mức này mọi so sánh giữa các cấu hình đều nằm trong vùng nhiễu, và Flavia không còn đủ khó để phân biệt các phương pháp.`),
  bullet("Phần tinh chỉnh em chỉ đánh giá trên một phép chia giữ lại, chưa chạy kiểm định chéo đầy đủ vì chạy trên CPU khá lâu."),
  bullet("Các siêu tham số như tốc độ học, số epoch, mức mở khoá đều do em chọn theo kinh nghiệm, chưa tìm kiếm có hệ thống."),
  bullet("Em chưa thử Transformer thị giác. Kiến trúc này cần nhiều dữ liệu hơn hoặc kỹ thuật huấn luyện riêng mới phát huy được trên tập nhỏ như Flavia."),

  heading("9. Kết luận"),
  p(`Học chuyển giao giải bài toán Flavia tốt hơn cách làm thủ công mà công sức bỏ ra lại ít hơn nhiều: ${m.probebestname} đóng băng đạt ${m.probebestcv}, ResNet18 tinh chỉnh đạt ${m.ftacc} trên tập kiểm tra giữ lại, so với ${m.classicalcv} của bài trước.`),
  p("Nhưng điều em thấy đáng nhớ nhất không phải con số cao nhất, mà là phần lớn kết quả đã có từ trước khi huấn luyện. Chỉ một bộ phân lớp tuyến tính đặt trên đặc trưng ImageNet đóng băng đã vượt cả quy trình thủ công, và hình t-SNE cho thấy lý do: các loài đã tụ thành cụm sẵn trong không gian embedding. Tinh chỉnh chỉ thêm được một phần nhỏ. Với bộ dữ liệu 1907 ảnh, cái quý của học sâu nằm ở biểu diễn chuyển giao từ dữ liệu lớn, chứ không nằm ở việc huấn luyện trên chính bộ dữ liệu nhỏ này."),
  p(`Điều thứ hai là số tham số không phải chỉ dấu của chất lượng. ${m.lightname} nhỏ hơn ${m.heavyname} ${m.paramratio} lần nhưng cho kết quả cao hơn và chạy nhanh hơn 5 lần, đủ nhẹ để đưa lên điện thoại.`),

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
