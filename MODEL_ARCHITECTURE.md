# Kiến Trúc Mô Hình AI Vietnamese Keyboard

## 🎯 Tổng Quan Hệ Thống

Vietnamese AI Keyboard là một hệ thống thông minh sử dụng mô hình GPT tùy chỉnh để dự đoán và gợi ý từ tiếng Việt từ input không dấu. Hệ thống kết hợp hai phương pháp chính:

1. **Frequency-based prediction** (Dự đoán dựa trên tần suất)
2. **Context-aware neural prediction** (Dự đoán neural nhận biết ngữ cảnh)

## 🏗️ Kiến Trúc Tổng Thể

```
Input không dấu → Text Processor → AI Recommender → Output có dấu
                                        ↓
                 ┌─────────────────────────────────────┐
                 │          AI Engine                  │
                 ├─────────────────────────────────────┤
                 │  Tokenizer  │  GPT Model  │ Cache   │
                 └─────────────────────────────────────┘
```

## 📚 Các Thành Phần Chính

### 1. Vietnamese Non-Accented GPT Model

**Vị trí**: `ml/models/gpt_model.py`

#### 🧠 Kiến Trúc Mô Hình

**Cấu hình mặc định:**

- **Vocab size**: 50,000 tokens
- **Block size**: 32 tokens (sequence length)
- **Layers**: 8 Transformer blocks
- **Attention heads**: 8 heads
- **Embedding dimension**: 256
- **Dropout**: 0.1

#### 🔧 Thành Phần Chi Tiết

##### Causal Self-Attention

```python
class CausalSelfAttention:
    - Multi-head attention với causal masking
    - Đảm bảo mô hình chỉ nhìn thấy tokens trước đó
    - Attention dropout để regularization
```

##### MLP Block

```python
class MLP:
    - Linear layer: n_embd → 4*n_embd
    - GELU activation
    - Linear layer: 4*n_embd → n_embd
    - Dropout
```

##### Transformer Block

```python
class Block:
    - LayerNorm → Self-Attention → Residual Connection
    - LayerNorm → MLP → Residual Connection
```

### 2. Tokenizer System

**Vị trí**: `ml/tokenizer.py`

#### 🔤 Chức Năng Chính

1. **Word-to-ID mapping**: Chuyển đổi từ thành ID số
2. **Non-accented mapping**: Map từ có dấu → không dấu
3. **Frequency tracking**: Theo dõi tần suất xuất hiện của từ
4. **Special tokens**: `<pad>`, `<unk>`, `<sos>`, `<eos>`

#### 💾 Dữ Liệu Lưu Trữ

- `vocab.json`: Từ điển word → ID
- `word_to_non_accented.json`: Map từ có dấu → không dấu
- `non_accented_to_words.json`: Map không dấu → list từ có dấu
- `word_freq.json`: Tần suất xuất hiện của từ

### 3. Inference Engine

**Vị trí**: `ml/inference.py`

#### 🚀 Quy Trình Dự Đoán

##### Bước 1: Dự đoán dựa trên tần suất

```python
def _get_tokenizer_suggestions():
    - Tìm các từ có mapping với input không dấu
    - Sắp xếp theo tần suất xuất hiện
    - Trả về top candidates với confidence score
```

##### Bước 2: Dự đoán dựa trên ngữ cảnh (Neural)

```python
def _get_model_suggestions():
    - Encode ngữ cảnh (5 từ gần nhất)
    - Forward pass qua GPT model
    - Lấy top-k predictions
    - Filter theo non-accented input
    - Trả về candidates với probability scores
```

##### Bước 3: Kết hợp và xếp hạng

```python
def _combine_suggestions():
    - Tokenizer suggestions: weight = 0.7
    - Model suggestions: weight = 0.8
    - Combined suggestions: model*0.8 + tokenizer*0.2
    - Sort theo confidence score giảm dần
```

### 4. AI Recommender

**Vị trí**: `core/ai_recommender.py`

#### 🎛️ Chức Năng Chính

1. **Context management**: Quản lý ngữ cảnh 5 từ gần nhất
2. **Performance tracking**: Theo dõi hiệu suất dự đoán
3. **Fallback handling**: Xử lý khi mô hình không khả dụng
4. **Cache optimization**: Tối ưu hóa bằng cache

## ⚙️ Quy Trình Training

### 1. Data Preprocessing

**Vị trí**: `ml/data_preprocessor.py`

```python
Corpus text → Tokenization → Word mapping → Vocabulary building
                 ↓
Training pairs generation (input_sequence → target_word)
                 ↓
Save: vocab.json, word_mappings.json, training_pairs.csv
```

### 2. Model Training

**Vị trí**: `train_model.py`

```python
Load training pairs → Create DataLoader → Initialize GPT model
                 ↓
Training loop: Forward pass → Loss calculation → Backprop
                 ↓
Validation → Save best checkpoint
```

## 🧩 Thuật Toán Dự Đoán Chi Tiết

### Input Processing

1. **Text cleaning**: Loại bỏ ký tự đặc biệt, chuẩn hóa
2. **Non-accented conversion**: Chuyển input thành dạng không dấu
3. **Context extraction**: Lấy 5 từ gần nhất làm ngữ cảnh

### Prediction Pipeline

#### Phương pháp 1: Frequency-based

```
Input: "xinchao"
↓
Lookup trong non_accented_to_words mapping
↓
Kết quả: [("xin chào", 0.85), ("xin chao", 0.10), ...]
```

#### Phương pháp 2: Context-aware Neural

```
Context: ["tôi", "muốn", "nói"]
Input: "xinchao"
↓
Encode context: [245, 1023, 567] → Tensor(1, 3)
↓
GPT forward: context_tensor → logits(1, vocab_size)
↓
Top-k sampling: → [("xin chào", 0.92), ("xin chao", 0.05), ...]
↓
Filter by non-accented match với "xinchao"
```

### Combining Strategy

```python
final_score = {
    "frequency_only": freq_score * 0.7,
    "model_only": model_score * 1.0,
    "combined": model_score * 0.8 + freq_score * 0.2
}
```

## 📊 Tối Ưu Hóa Hiệu Suất

### 1. Caching System

- **Key**: `f"{input}_{context}_{max_suggestions}_{use_model}_{temperature}"`
- **Hit rate tracking**: Theo dõi tỷ lệ cache hit
- **Memory management**: Giới hạn kích thước cache

### 2. Model Optimization

- **Inference mode**: `model.eval()` và `torch.no_grad()`
- **Device optimization**: Auto-detect CUDA/CPU
- **Batch processing**: Xử lý context theo batch

### 3. Tokenizer Optimization

- **Pre-loaded mappings**: Load tất cả mappings vào memory
- **Fast lookup**: Sử dụng dictionary cho O(1) lookup
- **Frequency sorting**: Pre-sort theo frequency

## 🎯 Điểm Mạnh Của Kiến Trúc

### 1. Hybrid Approach

- **Reliability**: Fallback về frequency-based khi model fail
- **Accuracy**: Neural model cung cấp context-aware predictions
- **Speed**: Cache system giảm latency

### 2. Scalability

- **Modular design**: Các component độc lập, dễ thay thế
- **Configurable**: Dễ dàng điều chỉnh parameters
- **Extensible**: Có thể thêm more prediction methods

### 3. Vietnamese-specific Optimization

- **Non-accented mapping**: Tối ưu cho cách gõ tiếng Việt
- **Frequency weighting**: Ưu tiên từ phổ biến
- **Context awareness**: Hiểu ngữ cảnh tiếng Việt

## 🔄 Luồng Xử Lý Hoàn Chỉnh

```
User input: "toi muon an com"
                ↓
Text processing: clean + normalize
                ↓
Word segmentation: ["toi", "muon", "an", "com"]
                ↓
For each word:
    ┌─────────────────────────────────────┐
    │ Frequency predictions               │
    │ + Context neural predictions       │
    │ → Combine & rank                   │
    │ → Cache result                     │
    └─────────────────────────────────────┘
                ↓
Final suggestions: ["tôi muốn ăn cơm", "tôi muốn ăn com", ...]
                ↓
UI display với confidence scores
```

## 📈 Metrics & Monitoring

### Performance Metrics

- **Prediction accuracy**: % correct predictions trong top-k
- **Response latency**: Thời gian từ input → output
- **Cache hit rate**: Tỷ lệ cache hits
- **Model coverage**: % cases sử dụng neural model

### Quality Metrics

- **Relevance score**: Mức độ phù hợp với ngữ cảnh
- **Diversity score**: Đa dạng trong suggestions
- **User satisfaction**: Tracking user selections

---

_File này mô tả kiến trúc và cách hoạt động chi tiết của Vietnamese AI Keyboard. Để hiểu sâu hơn, hãy tham khảo source code trong các thư mục tương ứng._
