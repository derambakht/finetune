# 🦥 Unsloth FineTune Studio

<div dir="rtl">

یک پروژه‌ی کامل و حرفه‌ای برای **فاین‌تیون مدل‌های زبانی بزرگ (LLM)** با کتابخانه‌ی [Unsloth](https://github.com/unslothai/unsloth) و پایتون، همراه با یک **رابط کاربری وب (Gradio)**. در این رابط، مدل، نوع کوانتیزه‌سازی، دیتاست و همه‌ی هایپرپارامترهای مهم را انتخاب می‌کنید، آموزش را به‌صورت زنده (نمودار loss، لاگ، پیشرفت و ETA) دنبال می‌کنید، خروجی را به GGUF یا مدل merge‌شده تبدیل می‌کنید و در همان‌جا با مدل آموزش‌دیده چت می‌کنید.

</div>

---

## ✨ امکانات / Features

<div dir="rtl">

| بخش | امکانات |
|---|---|
| 🧠 **مدل و کوانتیزه‌سازی** | کاتالوگ مدل‌های محبوب (Qwen3، Llama 3.x، Gemma 3، Mistral، Phi-4، gpt-oss، DeepSeek-R1) یا هر مدل دلخواه از Hugging Face / مسیر محلی · حالت‌های **QLoRA 4-bit**، **LoRA 8-bit**، **LoRA 16-bit** و **Full fine-tuning** · انتخاب dtype، طول دنباله، قالب چت · **تخمین زنده‌ی VRAM** و پیشنهاد GPU |
| 📚 **دیتاست** | Hugging Face Hub یا فایل محلی (json، jsonl، csv، tsv، parquet، txt) با آپلود مستقیم · تشخیص خودکار فرمت: Alpaca، ShareGPT، Messages/ChatML، Prompt/Completion، متن خام، نگاشت ستون دلخواه · system prompt، محدودیت تعداد نمونه، shuffle، تقسیم eval · **پیش‌نمایش** متن نهایی‌ای که مدل می‌بیند + آمار توکن |
| 🔧 **LoRA** | rank، alpha، dropout، target modules (به‌همراه embed_tokens/lm_head)، bias، rsLoRA، LoftQ، gradient checkpointing مخصوص Unsloth |
| ⚙️ **آموزش** | epoch یا max steps، batch size، gradient accumulation، learning rate، scheduler، warmup، optimizer (adamw_8bit و ...)، weight decay، grad clipping، seed، packing، **train on responses only**، NEFTune، checkpoint، eval، TensorBoard/W&B، ادامه از checkpoint · **پریست‌های آماده** |
| 🚀 **اجرا و مانیتورینگ** | اجرای آموزش در پروسه‌ی جداگانه (UI همیشه روان می‌ماند و VRAM کامل آزاد می‌شود) · نمودار زنده‌ی loss / eval loss / learning rate · نوار پیشرفت، ETA، مصرف VRAM · لاگ زنده · توقف امن (ذخیره‌ی checkpoint) یا kill |
| 📦 **خروجی** | LoRA adapter، مدل merge‌شده‌ی 16-bit و 4-bit، **GGUF** با چند نوع کوانتیزه‌سازی (q4_k_m، q5_k_m، q8_0، ...) برای Ollama / llama.cpp / LM Studio · ارسال به Hugging Face Hub · خروجی خودکار بعد از آموزش |
| 💬 **Playground** | چت استریم‌شده با مدل آموزش‌دیده یا هر checkpoint با کنترل temperature، top-p، top-k و ... |
| 🗂️ **تاریخچه و پیکربندی** | جدول همه‌ی اجراها با loss نهایی · ذخیره/بارگذاری کامل تنظیمات به‌صورت YAML · اجرای بدون UI با CLI |

</div>

---

## 📦 نصب / Installation

<div dir="rtl">

پیش‌نیاز آموزش: **GPU انویدیا با CUDA** (لینوکس یا WSL2؛ Google Colab / Kaggle هم کار می‌کند) و پایتون 3.10 تا 3.12.
رابط کاربری بدون GPU هم اجرا می‌شود (برای ساخت کانفیگ و پیش‌نمایش دیتاست).

</div>

```bash
git clone https://github.com/derambakht/finetune.git
cd finetune

python -m venv .venv && source .venv/bin/activate

# 1) Unsloth (installs matching torch / transformers / trl / peft / bitsandbytes)
pip install unsloth unsloth_zoo

# 2) This project (UI + CLI)
pip install -e ".[dev]"
```

<div dir="rtl">

> برای نصب Unsloth روی نسخه‌های خاص CUDA یا ویندوز، به [راهنمای نصب رسمی Unsloth](https://docs.unsloth.ai/get-started/installing-+-updating) مراجعه کنید.

</div>

---

## 🚀 اجرا / Quick start

```bash
python app.py                 # or: ftstudio ui
# open http://localhost:7860

python app.py --share         # public link (Colab, remote servers)
python app.py --port 8080 --output-root /data/runs
```

<div dir="rtl">

### روند کار در رابط کاربری

1. **🧠 Model & Quantization**: مدل را انتخاب کنید (یا شناسه‌ی دلخواه تایپ کنید) و نوع کوانتیزه‌سازی را تعیین کنید. قالب چت به‌طور خودکار انتخاب می‌شود و تخمین VRAM نمایش داده می‌شود. برای مدل‌های gated مثل Llama توکن Hugging Face را وارد کنید.
2. **📚 Dataset**: شناسه‌ی دیتاست یا فایل محلی را وارد/آپلود کنید، فرمت را روی `auto` بگذارید یا دستی انتخاب کنید و روی **Load & preview** بزنید تا دقیقاً متنی را که مدل روی آن آموزش می‌بیند مشاهده کنید.
3. **🔧 LoRA / PEFT** و **⚙️ Training**: هایپرپارامترها را تنظیم کنید یا یک **Preset** اعمال کنید.
4. **📦 Export**: (اختیاری) فرمت‌های خروجی را انتخاب و گزینه‌ی خروجی خودکار بعد از آموزش را فعال کنید.
5. **🚀 Train & Monitor**: ابتدا **Validate** و سپس **Start training**. نمودارها و لاگ هر ۲ ثانیه به‌روز می‌شوند.
6. **💬 Playground**: مدل آموزش‌دیده را بارگذاری کنید و با آن چت کنید.

</div>

---

## 🖥️ استفاده از CLI / Headless usage

```bash
ftstudio init-config my_run.yaml             # write a default config
ftstudio validate --config my_run.yaml       # check it
ftstudio train --config configs/qwen3_4b_alpaca_qlora.yaml
ftstudio train --config configs/local_jsonl_example.yaml --max-steps 20

# export an existing run
ftstudio export --source outputs/<run>/final_model --methods gguf merged_16bit --gguf q4_k_m q8_0
ftstudio export --source outputs/<run>/final_model --methods lora gguf --push username/my-model

# chat in the terminal
ftstudio chat --model outputs/<run>/final_model
```

<div dir="rtl">

همه‌ی اجراها در `outputs/<run_name>-<timestamp>/` ذخیره می‌شوند:

</div>

```
outputs/my-finetune-20260928-101500/
├── config.yaml        # exact config of the run (token redacted)
├── status.json        # live state / progress
├── metrics.jsonl      # loss, lr, grad-norm, eval loss, VRAM per log step
├── train.log          # full log
├── checkpoint-*/      # intermediate checkpoints (resume with resume_from_checkpoint)
├── final_model/       # LoRA adapter (or full model) + tokenizer + ftstudio_meta.json
├── summary.json
└── export/            # lora/, merged_16bit/, gguf/ ...
```

---

## 📚 فرمت‌های دیتاست / Dataset formats

| Format | Example row |
|---|---|
| `alpaca` | `{"instruction": "...", "input": "...", "output": "..."}` |
| `sharegpt` | `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}` |
| `messages` | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` |
| `prompt_completion` | `{"prompt": "...", "completion": "..."}` (also `question`/`answer`, `input`/`output` ...) |
| `text` | `{"text": "raw document ..."}` – continued pre-training |
| `custom` | any two columns mapped in the UI (+ optional system column) |

<div dir="rtl">

همه‌ی فرمت‌های گفتگو ابتدا به لیست پیام‌ها تبدیل و سپس با قالب چت همان مدل رندر می‌شوند؛ بنابراین مدل خروجی مستقیماً در Ollama، vLLM و llama.cpp با قالب درست کار می‌کند. گزینه‌ی **Train on responses only** بخش‌های کاربر/سیستم را از loss حذف می‌کند تا مدل فقط روی پاسخ‌ها آموزش ببیند.

یک فایل نمونه در `data/sample_chat.jsonl` قرار دارد.

</div>

---

## 🧮 راهنمای انتخاب تنظیمات / Tuning guide

<div dir="rtl">

| سناریو | پیشنهاد |
|---|---|
| GPU کم‌حافظه (۸ گیگ) | QLoRA 4-bit، مدل ≤ 8B، batch=1، accumulation=8، seq=1024، پریست **Low VRAM** |
| کیفیت بالاتر | LoRA 16-bit، r=32 و alpha=32، lr=1e-4، scheduler cosine، ۲ تا ۳ epoch، eval ratio ≈ 0.05 |
| آموزش زبان/دامنه‌ی جدید با متن خام | پریست **Continued pre-training**، شامل `embed_tokens` و `lm_head`، rsLoRA |
| Full fine-tuning | فقط مدل‌های کوچک یا GPU بزرگ، lr ≈ 2e-5 |
| overfit (افزایش eval loss) | کاهش epoch یا lr، افزایش داده، dropout کم |

learning rate معمول برای LoRA بین `1e-4` تا `3e-4` و برای full fine-tuning بین `1e-5` تا `5e-5` است. **effective batch size** = batch × gradient accumulation.

</div>

---

## 🏗️ معماری / Architecture

```
app.py                     # launcher (python app.py)
src/ftstudio/
├── config.py              # typed dataclass config + YAML + validation
├── models.py              # model catalog, chat templates, response markers, VRAM estimator
├── data.py                # dataset loading, format detection, conversion, preview
├── trainer.py             # Unsloth FastLanguageModel + LoRA + TRL SFTTrainer + callbacks
├── export.py              # LoRA / merged 16-bit / 4-bit / GGUF / Hub export
├── inference.py           # streaming chat engine
├── runner.py              # background subprocess job manager (train / export)
├── reporting.py           # status.json / metrics.jsonl protocol between job and UI
├── system.py              # GPU / package introspection
├── cli.py                 # `ftstudio` command line
└── ui/
    ├── app.py             # Gradio Blocks interface
    └── logic.py           # UI logic (field registry, presets, plots, runs table)
tests/                     # unit tests (no GPU required)
configs/                   # example YAML configurations
```

<div dir="rtl">

- **جداسازی پروسه‌ها:** آموزش و خروجی‌گیری در یک subprocess اجرا می‌شوند. رابط کاربری از طریق فایل‌های `status.json` و `metrics.jsonl` وضعیت را می‌خواند و برای توقف امن فایل `STOP` را ایجاد می‌کند؛ callback آموزش در پایان همان step یک checkpoint ذخیره و آموزش را متوقف می‌کند.
- **امنیت توکن:** توکن Hugging Face فقط از طریق متغیر محیطی به پروسه‌ی آموزش داده می‌شود و هرگز در YAML ذخیره نمی‌شود.
- **سازگاری نسخه‌ها:** پارامترهای `SFTConfig` / `SFTTrainer` بر اساس امضای نسخه‌ی نصب‌شده‌ی TRL فیلتر می‌شوند، پس تغییرات API (مثل `max_seq_length` ← `max_length` یا `tokenizer` ← `processing_class`) مشکلی ایجاد نمی‌کند.

</div>

---

## 🧪 تست / Tests

```bash
pip install -e ".[dev]"
pytest -q
```

<div dir="rtl">

تست‌ها بدون GPU اجرا می‌شوند و کانفیگ، تبدیل دیتاست، منطق UI، پروتکل گزارش‌دهی و تخمین VRAM را پوشش می‌دهند.

</div>

---

## ❓ عیب‌یابی / Troubleshooting

<div dir="rtl">

| خطا | راه‌حل |
|---|---|
| `CUDA out of memory` | 4-bit، batch کوچک‌تر با accumulation بیشتر، `max_seq_length` کمتر یا مدل کوچک‌تر |
| `401` / gated model | توکن Hugging Face با دسترسی به مدل را وارد کنید (یا `HF_TOKEN` را export کنید) |
| دیتاست خالی بعد از فرمت‌دهی | فرمت یا نگاشت ستون‌ها را در تب Dataset بررسی و Preview بگیرید |
| خطای GGUF | اولین تبدیل llama.cpp را build می‌کند؛ `cmake` و `build-essential` باید نصب باشند |
| مدل پاسخ‌های نامرتبط می‌دهد | در Playground همان chat template آموزش را انتخاب کنید |

</div>

## License

MIT
