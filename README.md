# 🇻🇳 Vietnamese AI Keyboard - Enhanced Edition

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Accuracy](https://img.shields.io/badge/accuracy-96.22%25-green.svg)
![Processing](https://img.shields.io/badge/processing-<3ms-orange.svg)
![Vocabulary](https://img.shields.io/badge/vocabulary-44K+-red.svg)

**Bộ gõ tiếng Việt thông minh sử dụng AI với 44,000+ từ vựng và độ chính xác 96.22%**

---

## 🚀 **Tính năng nổi bật**

### ✨ **Core Features**

- **🎯 Độ chính xác cao**: 96.22% trên 9,083 test cases unseen
- **⚡ Tốc độ xử lý**: < 3ms per suggestion
- **📚 Từ vựng phong phú**: 44,102 từ từ Viet74K và corpus thực tế
- **🧠 AI-powered**: Systematic evaluation và automatic improvement
- **🎨 GUI hiện đại**: Real-time suggestions với confidence visualization

### 🔬 **Advanced Technologies**

- **Hybrid Processing**: Core patterns + Extended coverage
- **Multi-level Algorithms**: 7 different processing methods
- **Context-aware Suggestions**: N-gram patterns từ 19GB corpus
- **AI Error Learning**: Automatic improvement từ systematic analysis
- **Data-driven Optimization**: Train/test split methodology

---

## 📊 **Performance Metrics**

| Metric                    | Value        | Notes                            |
| ------------------------- | ------------ | -------------------------------- |
| **Accuracy**              | 96.22%       | Tested on 9,083 unseen cases     |
| **Processing Speed**      | 2.53ms       | Average per suggestion           |
| **Vocabulary Coverage**   | 44,102 words | Viet74K + Corpus patterns        |
| **Error Rate**            | 3.78%        | 343 systematic errors identified |
| **Segmentation Accuracy** | 97.99%       | Dynamic programming approach     |

---

## 🛠️ **Installation & Setup**

### **Prerequisites**

```bash
Python 3.7+
tkinter (usually included with Python)
```

### **Quick Installation**

```bash
# Clone repository
git clone https://github.com/your-repo/vietnamese-ai-keyboard.git
cd vietnamese-ai-keyboard/New_version

# Install dependencies
pip install -r requirements.txt

# Run application
python enhanced_launcher_gui.py
```

### **Data Setup**

```bash
# Data files cần thiết (đã included):
data/Viet74K.txt                    # 73,902 từ vựng tiếng Việt
data/processed_vietnamese_data.json  # Processed data cho AI system
```

---

## 🎮 **Usage Guide**

### **1. Basic Usage**

```bash
# Launch Enhanced GUI (Recommended)
python enhanced_launcher_gui.py

# Launch Simple GUI
python launcher_gui.py
```

### **2. GUI Interface**

- **Input Box**: Gõ văn bản không dấu (ví dụ: `toihocbai`)
- **Suggestions**: 5 gợi ý với confidence scores
- **Keyboard Shortcuts**: Phím 1-5 để chọn gợi ý nhanh
- **Copy Output**: Nút Copy để copy kết quả

### **3. Example Usage**

```
Input: "toihocbai"
Output: "tôi học bài" (95% confidence)

Input: "xinchao"
Output: "xin chào" (90% confidence)

Input: "homnaytoilam"
Output: "hôm nay tôi làm" (95% confidence)
```

---

## 🏗️ **System Architecture**

### **Core Components**

#### **1. Hybrid Vietnamese Processor**

```
ml/hybrid_vietnamese_processor.py
- Core dictionaries (118 proven patterns)
- Extended dictionaries (43,989 from Viet74K)
- Multi-level processing pipeline
- Context-aware suggestions
```

#### **2. Data Processing**

```
ml/data_processor.py
- Viet74K vocabulary processing
- Corpus pattern extraction
- N-gram analysis (bigrams, trigrams)
- Data categorization và filtering
```

#### **3. AI Enhancement System**

```
ml/systematic_evaluator.py      # Scientific evaluation framework
ml/ai_improvement_model.py      # AI learning từ errors
ml/auto_improvement_system.py   # Automated system enhancement
```

### **Processing Pipeline**

```
Input Text → Core Matching → Extended Matching → Context Analysis →
AI Segmentation → Confidence Ranking → Top Suggestions
```

---

## 🔬 **AI & Machine Learning**

### **Systematic Evaluation Framework**

- **Train/Test Split**: 80/20 split với 45,421 data points
- **Performance Evaluation**: Comprehensive metrics trên unseen data
- **Error Pattern Analysis**: AI-driven error categorization
- **Improvement Generation**: Automated enhancement recommendations

### **AI Learning Components**

1. **Segmentation Pattern Learning**: 19 learned patterns
2. **Vocabulary Gap Analysis**: 100 identified gaps with priorities
3. **Context Rule Extraction**: Data-driven context awareness
4. **Automated Improvement**: Self-improving system

### **Data Sources**

- **Viet74K**: 73,902 authoritative Vietnamese vocabulary
- **Corpus**: 19GB real Vietnamese text data
- **N-grams**: 556K+ bigrams, 1.5M+ trigrams
- **User Patterns**: Real-world usage statistics

---

## 📁 **Project Structure**

```
vietnamese-ai-keyboard/
├── New_version/                    # Main application directory
│   ├── enhanced_launcher_gui.py    # 🎨 Enhanced GUI (Recommended)
│   ├── launcher_gui.py            # 🎮 Simple GUI
│   │
│   ├── ml/                        # 🧠 AI & Processing Core
│   │   ├── hybrid_vietnamese_processor.py    # Main processing engine
│   │   ├── data_processor.py                # Data processing utilities
│   │   ├── systematic_evaluator.py          # Evaluation framework
│   │   ├── ai_improvement_model.py          # AI learning system
│   │   ├── auto_improvement_system.py       # Automated enhancement
│   │   └── simple_vietnamese_processor.py   # Lightweight processor
│   │
│   ├── data/                      # 📚 Data Assets
│   │   ├── Viet74K.txt           # Vietnamese vocabulary (73K words)
│   │   └── processed_vietnamese_data.json   # Processed AI data
│   │
│   ├── docs/                      # 📖 Documentation
│   │   ├── IMPROVEMENT_PLAN.md              # Development roadmap
│   │   ├── IMPROVEMENT_SUMMARY.md           # Project achievements
│   │   └── SYSTEMATIC_EVALUATION_REPORT.md  # AI evaluation results
│   │
│   ├── reports/                   # 📊 AI Reports
│   │   ├── systematic_evaluation_report.json
│   │   └── ai_improvements.json
│   │
│   └── requirements.txt           # Dependencies
```

---

## 🧪 **Testing & Evaluation**

### **Systematic Testing**

```bash
# Run comprehensive evaluation
python ml/systematic_evaluator.py

# Generate AI improvements
python ml/ai_improvement_model.py

# Apply automated improvements
python ml/auto_improvement_system.py
```

### **Performance Benchmarks**

- **Accuracy Test**: 96.22% trên 9,083 unseen cases
- **Speed Test**: 2.53ms average processing time
- **Memory Usage**: ~100MB với full dataset
- **Error Analysis**: 343 systematic errors categorized

### **Test Cases**

```python
# Core test cases được proven:
test_cases = [
    "toihocbai" → "tôi học bài",
    "toilasinhvien" → "tôi là sinh viên",
    "homnaytoilam" → "hôm nay tôi làm",
    "xinchao" → "xin chào",
    "camon" → "cảm ơn"
]
```

---

## 📈 **Performance Optimization**

### **Speed Optimization**

- **Async Processing**: Non-blocking UI operations
- **Efficient Data Structures**: Optimized lookups
- **Memory Management**: Smart caching strategies
- **Algorithm Optimization**: Multi-level fallback system

### **Accuracy Improvements**

- **Context Awareness**: N-gram pattern matching
- **Error Learning**: AI-driven improvement từ mistakes
- **Vocabulary Expansion**: Data-driven additions
- **Segmentation Enhancement**: Dynamic programming approach

---

## 🤝 **Contributing**

### **Development Setup**

```bash
# Development environment
git clone <repository>
cd New_version

# Install development dependencies
pip install -r requirements.txt

# Run tests
python ml/systematic_evaluator.py
```

### **Contributing Guidelines**

1. **Code Quality**: Follow Python PEP 8 standards
2. **Testing**: Add tests cho new features
3. **Documentation**: Update README cho changes
4. **Performance**: Maintain <3ms processing speed
5. **Accuracy**: Ensure >95% accuracy trên test sets

---

## 📄 **License**

MIT License - See LICENSE file for details

---

## 🙏 **Acknowledgments**

- **Viet74K Dataset**: Comprehensive Vietnamese vocabulary
- **Vietnamese Corpus**: Real-world text data cho training
- **Open Source Community**: Tools và libraries used
- **Vietnamese Language Research**: Academic foundations

---

## 📞 **Support & Contact**

### **Issues & Bug Reports**

- GitHub Issues: [Create Issue](https://github.com/your-repo/issues)
- Email: support@vietnamese-ai-keyboard.com

### **Documentation**

- **Detailed Docs**: See `docs/` directory
- **API Reference**: Code documentation in source files
- **Performance Reports**: `reports/` directory

### **Community**

- **Discord**: Vietnamese AI Keyboard Community
- **Forums**: Technical discussions và support

---

## 🔮 **Future Roadmap**

### **Version 2.1 (Planned)**

- **Neural Models**: Transformer-based segmentation
- **Voice Input**: Speech-to-text integration
- **Mobile Apps**: iOS/Android versions
- **Cloud Sync**: Multi-device synchronization

### **Version 3.0 (Research)**

- **Deep Learning**: BERT-based Vietnamese models
- **Real-time Learning**: Online adaptation from user input
- **Multi-modal**: Voice + text integration
- **Enterprise Features**: Team dictionaries, admin controls

---

**Vietnamese AI Keyboard - Revolutionizing Vietnamese Typing Experience! 🇻🇳🤖✨**

_Built with ❤️ for the Vietnamese community_
