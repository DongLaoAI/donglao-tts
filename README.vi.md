<div align="center">
  <p><a href="README.md">English</a> · <strong>Tiếng Việt</strong></p>

  <img src="assets/donglao-tts-logo.png" alt="donglao-tts — logo cá sấu hát nằm ngang, hình học góc cạnh" width="432" />

  <h1>donglao-tts</h1>

  <p><strong>Text-to-Speech theo giọng tham chiếu với Python API đơn giản.</strong></p>

  <p>
    <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
    <img alt="Phiên bản 0.1.6" src="https://img.shields.io/badge/version-0.1.6-00C7B7">
    <img alt="Giấy phép Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-7C3AED">
    <a href="https://huggingface.co/DongLao/DongLao-TTS"><img alt="Model Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-DongLao--TTS-FFD21E?logo=huggingface&logoColor=black"></a>
  </p>
</div>

> [!IMPORTANT]
> `donglao-tts` đang ở giai đoạn research/pre-1.0. API có thể thay đổi giữa các phiên bản. Chỉ
> sử dụng giọng tham chiếu khi có sự đồng ý của người nói.

## Python API khuyến nghị

Cài package:

```bash
uv add donglao-tts

# Hoặc dùng pip
python -m pip install donglao-tts
```

Tải model đã phát hành và tổng hợp giọng nói:

```python
from donglao_tts import DongLaoTTS

tts = DongLaoTTS.from_pretrained("DongLao/DongLao-TTS")

waveform = tts.generate(
    "Xin chào mọi người, đây là Đông Lào TTS.",
    reference_audio="reference.wav",
    reference_text="Transcript chính xác của nội dung trong reference.wav.",
    output_path="output.wav",
    leading_silence_ms=20,
    sentence_pause_ms=180,
    trailing_silence_ms=20,
)

print("Model revision:", tts.revision)
print("Sample rate:", tts.sample_rate)
print("Waveform shape:", tuple(waveform.shape))
```

`from_pretrained()` tự sử dụng CUDA nếu khả dụng, nếu không sẽ chạy trên CPU. Branch hoặc tag
được resolve thành commit bất biến trước khi tải. Lần gọi đầu tải model, tokenizer và MOSS audio
codec đã đóng gói; những lần sau sử dụng Hugging Face cache.

Giữ đối tượng `tts` trong bộ nhớ và dùng batch API cho các target có cùng reference:

```python
texts = [
    "Xin chào mọi người.",
    "The model is loaded only once.",
    "Bạn có thể tổng hợp nhiều câu liên tiếp.",
]

waveforms = tts.generate_batch(
    texts,
    reference_audio="reference.wav",
    reference_text="Transcript chính xác của nội dung trong reference.wav.",
    output_paths=[f"outputs/result-{index}.wav" for index in range(len(texts))],
)
```

Batch inference chỉ phonemize và encode reference dùng chung một lần; phần sinh AR/NAR vẫn độc lập
cho từng target. Với codec-token streaming, xử lý từng RVQ chunk sau khi decode:

```python
for audio_chunk in tts.generate_stream(
    "Đây là câu thứ nhất. Đây là câu thứ hai.",
    reference_audio="reference.wav",
    reference_text="Transcript chính xác của nội dung trong reference.wav.",
    chunk_frames=5,
):
    send_audio(audio_chunk, sample_rate=tts.sample_rate)
```

AR giữ nguyên KV-cache và sinh mỗi lần `chunk_frames` token RVQ0. NAR điền các tầng RVQ còn lại
cho nhóm token đó, sau đó MOSS decode và trả waveform trong khi AR tiếp tục sinh. Nhóm cuối của
mỗi câu có thể ít frame hơn. Với codec 12.5 Hz, `chunk_frames=5` tương ứng khoảng 400 ms audio.
Streaming trả silence thành chunk riêng ở đầu, giữa các câu tách theo dấu chấm và ở cuối.

Trong production, nên khóa commit model đã kiểm thử và chỉ định device rõ ràng:

```python
tts = DongLaoTTS.from_pretrained(
    "DongLao/DongLao-TTS",
    revision="6ba3003ccb8d938c2a725a4117084492909c9419",
    device="cuda",
)
```

### Tùy chọn generation

```python
waveform = tts.generate(
    "Nội dung cần tổng hợp.",
    reference_audio="reference.wav",
    reference_text="Transcript chính xác của audio tham chiếu.",
    output_path="output.wav",  # không bắt buộc
    max_frames=200,
    temperature=0.8,
    top_k=10,
    leading_silence_ms=20,
    sentence_pause_ms=180,
    trailing_silence_ms=20,
)
```

| Tham số | Ý nghĩa |
|---|---|
| `text` | Nội dung không rỗng cần tổng hợp |
| `reference_audio` | Đường dẫn WAV/FLAC tham chiếu được phép sử dụng |
| `reference_text` | Transcript chính xác của bản thu tham chiếu |
| `output_path` | Đường dẫn audio đầu ra, không bắt buộc |
| `max_frames` | Số codec frame sinh tối đa |
| `temperature` | Nhiệt độ sampling; `0` dùng greedy decoding |
| `top_k` | Ngưỡng sampling; `0` tắt top-k truncation |
| `leading_silence_ms` | Khoảng lặng trước câu đầu; mặc định `20` ms |
| `sentence_pause_ms` | Khoảng lặng giữa các câu tách theo dấu chấm; mặc định `180` ms |
| `trailing_silence_ms` | Khoảng lặng sau câu cuối; mặc định `20` ms |

Sau bước G2P, văn bản đích được tách theo dấu chấm và từng câu được sinh riêng. Các waveform của
từng câu được nối theo đúng thứ tự, mặc định có silence `20 / 180 / 20` ms ở đầu / giữa các câu /
cuối. Đặt thời lượng bất kỳ về `0` để tắt phần padding đó. Hàm trả về `torch.Tensor` trên CPU có
shape `[channels, samples]`, kể cả khi không truyền `output_path`.

## Cài đặt

Runtime được hỗ trợ:

- Python `>=3.10,<3.11`
- Linux x86-64
- PyTorch và TorchAudio `>=2.8.0,<3`
- Khuyến nghị GPU hỗ trợ CUDA; vẫn có thể inference bằng CPU

Tạo môi trường riêng với uv:

```bash
uv venv --python 3.10
uv pip install donglao-tts
```

Cài từ source khi cần sử dụng phiên bản chưa phát hành:

```bash
git clone https://github.com/DongLaoAI/donglao-tts.git
cd donglao-tts
uv sync --locked
```

## Cách hoạt động

```mermaid
flowchart LR
    RA[Audio tham chiếu] --> ENC[MOSS encoder]
    RT[Transcript tham chiếu] --> G2P[donglao-g2p]
    TT[Văn bản đích] --> G2P
    G2P --> SPM[SentencePiece]
    ENC --> AR[AR model]
    SPM --> AR
    AR --> NAR[NAR model]
    NAR --> DEC[MOSS decoder]
    DEC --> WAV[Audio tổng hợp]
```

AR sinh lớp residual-vector-quantization đầu tiên cùng hidden state. NAR điền các lớp codec còn
lại, sau đó MOSS codec trong model bundle chuyển chúng thành audio.

Chỉ trong quá trình training, một bộ upsample thời gian 2x có tham số và CTC head căn chỉnh AR
hidden theo từng target frame với chuỗi SentencePiece đích. EOS auxiliary loss cân bằng tăng cường
class EOS hiện có. Các objective phụ cập nhật representation của AR nhưng không được đưa vào bundle
inference native, ONNX hoặc Hugging Face.

## Huấn luyện và export model

Cấu hình đường dẫn dataset và tokenizer trong `configs/base.yaml`, sau đó bắt đầu hoặc tiếp tục:

```bash
sh train.sh                 # mặc định --resume auto
sh train.sh --resume none   # chủ động bắt đầu run mới
```

Cấu hình training mặc định bật các objective chỉ dùng lúc huấn luyện:

```yaml
train:
  ctc_loss_weight: 0.1
  ctc_upsample_factor: 2
  ctc_warmup_steps: 5000
  eos_aux_loss_weight: 0.1

sample:
  leading_silence_ms: 20
  sentence_pause_ms: 180
  trailing_silence_ms: 20
```

Checkpoint chứa optimizer và trạng thái CTC để resume chính xác. Công cụ publish Hugging Face chỉ
load và export trọng số AR/NAR; dataset, optimizer, CTC head, đường dẫn training và sample config
cục bộ đều bị loại khỏi bundle:

```bash
uv sync --locked --group dev --all-extras
sh scripts/push_to_hub.sh
```

Mặc định script publish `checkpoints/run_02/step_90000.pt` lên `DongLao/DongLao-TTS` và không yêu
cầu console entry point `donglao-push-to-hub` đã được cài. Có thể ghi đè giá trị mặc định bằng các
argument đứng sau, ví dụ:

```bash
sh scripts/push_to_hub.sh \
  --checkpoint checkpoints/run_02/step_100000.pt \
  --repo-id DongLao/DongLao-TTS-Test \
  --private
```

Lệnh tương đương khi package đã được cài là:

```bash
uv run donglao-push-to-hub \
  --config configs/base.yaml \
  --checkpoint checkpoints/run_02/step_100000.pt \
  --repo-id DongLao/DongLao-TTS \
  --out-dir release-bundle
```

Xác thực phải được cung cấp rõ ràng qua `HF_TOKEN` hoặc đăng nhập Hugging Face CLI trước đó. Hãy
kiểm tra model card, kết quả đánh giá, điều khoản dataset và bundle sinh ra trước khi publish.

## Đóng góp

Issue và pull request có phạm vi rõ ràng đều được chào đón. Trước khi mở pull request, chạy:

```bash
uv sync --locked --group dev --all-extras
uv run --locked ruff check src scripts tests
uv run --locked python -m pytest -q
uv pip check
```

Không commit dataset, checkpoint, credential, audio riêng tư hoặc model artifact đã sinh.

## Roadmap

- [x] Streaming hoặc chunked synthesis
- [ ] Inference server production và observability hook
- [ ] Bổ sung kết quả đánh giá và ví dụ model card
- [ ] Hỗ trợ thêm phiên bản Python và nền tảng
- [x] Python API đơn giản với `DongLaoTTS.from_pretrained()`
- [x] Model bundle hoàn chỉnh gồm AR + NAR + MOSS

Roadmap thể hiện định hướng, không phải cam kết thời hạn.

## Ghi nhận

Dự án xây dựng trên [PyTorch](https://pytorch.org/),
[MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano),
[Hugging Face](https://huggingface.co/), [SentencePiece](https://github.com/google/sentencepiece)
và [donglao-g2p](https://pypi.org/project/donglao-g2p/).

## Giấy phép

Mã nguồn được phát hành theo [Apache License 2.0](LICENSE). Model, codec, dataset, bản thu và danh
tính người nói của bên thứ ba có thể có điều khoản riêng. Người dùng chịu trách nhiệm xác minh
các quyền cần thiết.
