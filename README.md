# 🤖 Vietnamese AI Keyboard System

Hệ thống bàn phím AI tiếng Việt sử dụng deep learning để gợi ý từ thông minh.

## 🎯 Tính năng chính

- **🧠 AI-Powered**: Sử dụng mô hình GPT để dự đoán từ tiếng Việt
- **📝 Non-accented Input**: Nhập kiểu "xinchao" → gợi ý "xin chào"
- **🔄 Context-Aware**: Học từ ngữ cảnh để cải thiện độ chính xác
- **⚡ Real-time**: Phản hồi nhanh với caching thông minh
- **🎨 Modern UI**: Giao diện đẹp với theme tối hiện đại

## 📁 Cấu trúc dự án

```
New_version/
├── 📄 README.md              # Tài liệu này
├── 📄 requirements.txt       # Dependencies
├── 📄 run_ai_keyboard.py     # 🚀 Launcher chính
├── 📄 train_model.py         # 🏋️ Training script
├── 📁 core/                  # Core modules
│   ├── ai_recommender.py     # AI recommendation engine
│   └── text_processor.py     # Text processing utilities
├── 📁 ui/                    # User interface
│   └── ai_keyboard_ui.py     # Modern AI keyboard UI
├── 📁 ml/                    # Machine learning
│   ├── data_preprocessor.py  # Corpus preprocessing
│   ├── tokenizer.py          # Vietnamese tokenizer
│   ├── inference.py          # Real-time inference
│   ├── models/               # GPT model architecture
│   └── training/             # Training infrastructure
├── 📁 data/                  # Training data
├── └── 📁 checkpoints/       # Saved models
```

## 🚀 Cài đặt nhanh

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt file `corpus-full.txt` vào thư mục `data/`:

```
data/
└── corpus-full.txt    # File chứa câu tiếng Việt để training
```

### 3. Training mô hình (tùy chọn)

```bash
python train_model.py --corpus_file data/corpus-full.txt --epochs 10
```

### 4. Chạy ứng dụng

```bash
python run_ai_keyboard.py
```

## 💡 Cách sử dụng

1. **Khởi động**: Chạy `python run_ai_keyboard.py`
2. **Nhập text**: Gõ Vietnamese không dấu như "xinchao", "tiengviet"
3. **Chọn gợi ý**: Dùng số 1-8 hoặc click chuột để chọn
4. **Context learning**: Hệ thống sẽ học và cải thiện từ lựa chọn của bạn

## 🧠 Kiến trúc AI

### GPT Model

- **Layers**: 8 transformer layers
- **Attention Heads**: 8 heads
- **Embedding Dim**: 256
- **Vocabulary**: 50,000 từ tiếng Việt
- **Context Length**: 32 tokens

### Training Process

1. **Preprocessing**: Xử lý corpus tiếng Việt
2. **Tokenization**: Tạo mappings từ không dấu sang có dấu
3. **Model Training**: Train GPT với context prediction
4. **Inference**: Real-time prediction với caching

## 📊 Performance

- **Response Time**: ~10-20ms
- **Memory Usage**: ~80MB RAM
- **Accuracy**: ~85-90% top-1 prediction
- **Fallback**: Frequency-based khi model không khả dụng

## ⚙️ Configuration

### Training Parameters

```python
# train_model.py
--corpus_file     # Path to corpus file
--epochs          # Training epochs (default: 5)
--batch_size      # Batch size (default: 32)
--learning_rate   # Learning rate (default: 1e-4)
--max_vocab_size  # Max vocabulary (default: 50000)
```

### Model Settings

```python
# ml/models/gpt_model.py
n_layers = 8        # Transformer layers
n_heads = 8         # Attention heads
n_embd = 256        # Embedding dimension
context_length = 32 # Context window
```

## 🔧 Troubleshooting

### Lỗi thường gặp

**1. ModuleNotFoundError**: Chạy `pip install -r requirements.txt`

**2. Không tìm thấy corpus**: Đặt file `corpus-full.txt` vào folder `data/`

**3. Out of memory**: Giảm batch_size trong training

**4. Model không load**: Chạy training trước hoặc kiểm tra checkpoint

### Performance tuning

- **Tăng accuracy**: Increase training epochs và vocabulary size
- **Tăng speed**: Enable caching và reduce model size
- **Reduce memory**: Decrease batch_size và embedding dimension

## 📈 Development

### Extend system

1. **Custom Model**: Modify `ml/models/gpt_model.py`
2. **New UI**: Create new UI in `ui/` folder
3. **Data Processing**: Extend `ml/data_preprocessor.py`
4. **Recommender Logic**: Modify `core/ai_recommender.py`

### Debug mode

```bash
python run_ai_keyboard.py --debug
```

## 📝 Changelog

### v2.0 (Current)

- ✅ Pure AI-powered system
- ✅ GPT-based recommendations
- ✅ Context-aware predictions
- ✅ Modern UI với statistics
- ✅ Cleaned up codebase

### v1.0 (Legacy)

- ❌ Rule-based recommendations (removed)
- ❌ Dictionary lookups (removed)
- ❌ Multiple UI versions (consolidated)

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

MIT License - See LICENSE file for details

---

**🎉 Happy coding với Vietnamese AI Keyboard!**

> _Hệ thống này được thiết kế để giúp người dùng gõ tiếng Việt nhanh và chính xác hơn bằng AI._
