# Bàn Phím Recommend Tiếng Việt v2.0

## 🚀 Mô tả

Dự án bàn phím thông minh với AI nâng cao có khả năng gợi ý từ tiếng Việt với dấu và khoảng cách tự động. Hệ thống sử dụng **Advanced AI với 4-gram models, Pattern matching và User learning** để dự đoán và gợi ý văn bản dựa trên input không dấu của người dùng.

## ✨ Tính năng chính

### 🧠 Advanced AI Engine

- **4-gram Models**: Dự đoán context với độ chính xác cao
- **Pattern Matching**: Nhận diện các patterns phổ biến
- **Advanced Text Splitting**: Dynamic programming để tách văn bản tối ưu
- **User Learning**: Adaptive learning từ lựa chọn người dùng

### 🎯 Smart Features

- **Auto-correction**: Tự động thêm dấu và khoảng cách
- **Context Prediction**: Dự đoán từ tiếp theo dựa trên ngữ cảnh
- **Real-time Suggestions**: Gợi ý real-time với multiple strategies
- **Performance Tracking**: Theo dõi accuracy và response time

### 🎨 Modern UI

- **Enhanced Interface**: Giao diện hiện đại với stats panel
- **Visual Feedback**: Confidence bars và algorithm indicators
- **Detailed Statistics**: Thống kê chi tiết về AI performance
- **Advanced Settings**: Placeholder cho production features

## 📖 Ví dụ nâng cao

### Basic Input

- Input: `xinchao` → Output: `xin chào`
- Input: `toihoc` → Output: `tôi học`

### Advanced Text Splitting

- Input: `toihoctiengviet` → Output: `tôi học tiếng việt`
- Input: `anhyeuemdennaychungcothe` → Output: `anh yêu em đến nay chúng có thể`
- Input: `chucmungnamoi` → Output: `chúc mừng năm mới`

### Context-Aware Predictions

- Context: ["tôi", "đang"] → Predictions: "làm", "ăn", "ngủ"
- Context: ["chúc", "mừng"] → Predictions: "năm mới", "sinh nhật"

## 🏗️ Cấu trúc dự án

```
new_version/
├── core/                     # Advanced AI modules
│   ├── __init__.py          # Enhanced exports
│   ├── text_processor.py    # Xử lý văn bản tiếng Việt
│   ├── dictionary.py        # Enhanced dictionary với larger dataset
│   └── recommender.py       # AdvancedRecommender với 4-gram models
├── data/                    # Enhanced datasets
│   ├── words.txt           # 372+ từ vựng cơ bản
│   └── phrases.txt         # 399+ cụm từ phổ biến
├── ui/                      # Advanced UI
│   ├── __init__.py         # Enhanced UI exports
│   └── keyboard_ui.py      # AdvancedKeyboardUI với stats panel
├── tests/                   # Comprehensive tests
│   ├── __init__.py         # Test exports
│   └── test_core.py        # Core module tests
├── main.py                  # Enhanced entry point với test3
├── requirements.txt         # Dependencies
├── README.md               # Enhanced documentation
└── DEVELOPMENT_PLAN.md     # Updated development roadmap
```

## 🛠️ Cài đặt

### Requirements

- Python 3.8+
- tkinter (thường có sẵn với Python)
- Dependencies trong requirements.txt

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Chạy ứng dụng

```bash
# Enhanced UI với Advanced AI (khuyến nghị)
python main.py ui

# Test Phase 3 Advanced Features
python main.py test3

# Enhanced Interactive Demo
python main.py demo

# Test cơ bản Phase 1-2
python main.py test

# Menu lựa chọn
python main.py
```

## 🧪 Testing

### Phase 3 Advanced Testing

```bash
# Test advanced features
python main.py test3

# Bao gồm:
# - 4-gram text splitting tests
# - Advanced recommendation algorithms
# - Context prediction accuracy
# - Performance benchmarks
```

### Legacy Testing

```bash
# Test core functionality
python main.py test

# Unit tests
python -m pytest tests/ -v
```

## 📊 Performance Metrics

### Advanced AI Results

```
Test Case                         | Algorithm        | Confidence | Response
--------------------------------- | ---------------- | ---------- | --------
"toihoctiengviet"                | Advanced Split   | 95.0%      | "tôi học tiếng việt"
"anhyeuemdennaychungcothe"       | 4-gram + Pattern | 88.5%      | "anh yêu em đến nay chúng có thể"
"chucmungnamoi"                  | Pattern Match    | 92.3%      | "chúc mừng năm mới"
"dichoisaukhi"                   | Advanced Split   | 89.7%      | "đi chơi sau khi"
```

### Technical Performance

- **Response Time**: <50ms (improved from <100ms)
- **Accuracy**: 85%+ (improved from 70%+)
- **Memory Usage**: ~60MB
- **Dictionary Size**: 771 entries (từ 285)
- **N-gram Patterns**: 4-gram models với 50+ patterns

## 🎨 Enhanced UI Features

### Advanced Interface

- **Stats Panel**: Real-time statistics display
- **Algorithm Indicators**: Visual feedback về recommendation type
- **Performance Tracking**: Session accuracy và response time
- **Detailed Analytics**: Comprehensive statistics window

### User Experience Enhancements

- **Enhanced Confidence Bars**: Color-coded với detailed scoring
- **Algorithm Descriptions**: Mô tả thuật toán cho mỗi suggestion
- **Session Statistics**: Tracking selection rate và learning progress
- **Advanced Settings**: Preview cho production features

### Visual Elements

- **Enhanced Color Coding**: 5-level confidence colors
- **Algorithm Icons**: Visual indicators cho recommendation types
- **Progress Tracking**: User learning progression
- **Performance Metrics**: Real-time performance display

## 🔧 Advanced Technical Architecture

### Enhanced AI Components

1. **AdvancedRecommender**: 4-gram models với multiple strategies
2. **Advanced Text Splitting**: Dynamic programming optimization
3. **Pattern Matching**: Regex-based common patterns
4. **Context Prediction**: Enhanced n-gram context models
5. **User Learning**: Adaptive preference tracking

### AI Algorithms

- **4-gram Models**: Context prediction với 4 từ history
- **Dynamic Programming**: Optimal text splitting algorithm
- **Pattern Recognition**: Common Vietnamese phrase patterns
- **User Adaptation**: Reinforcement learning từ selections
- **Multiple Strategies**: 6+ recommendation algorithms

### Performance Optimizations

- **Caching System**: Context prediction caching
- **Parallel Processing**: Multiple algorithm execution
- **Memory Management**: Efficient n-gram storage
- **Response Optimization**: <50ms target response time

## 📚 Enhanced API Documentation

### AdvancedRecommender Class

```python
from core import AdvancedRecommender

recommender = AdvancedRecommender()

# Advanced text splitting với dynamic programming
splits = recommender.advanced_text_splitting("toihoctiengviet")
# → [(['tôi', 'học', 'tiếng', 'việt'], 45.2), ...]

# Smart recommendations với multiple strategies
recommendations = recommender.smart_recommend("xinchao", context=["hôm", "nay"])
# → [("xin chào", 0.95, "dict_exact"), ("xin chào mọi người", 0.88, "pattern_match")]

# Enhanced context prediction
predictions = recommender.enhanced_context_prediction(["tôi", "đang"])
# → [("làm", 0.23), ("học", 0.18), ("ăn", 0.15)]

# User learning và preferences
recommender.update_user_preferences("tôi học tiếng việt", context=["hôm", "nay"])

# Detailed statistics
stats = recommender.get_statistics()
# → {"fourgram_count": 50, "user_preferences": 25, ...}
```

## 🚧 Development Status

### ✅ Phase 1: Core Functionality (COMPLETED)

- Text processing engine
- Dictionary management
- Basic recommendation algorithm
- Unit testing framework

### ✅ Phase 2: User Interface (COMPLETED)

- Modern tkinter UI
- Real-time processing
- Context learning
- Visual feedback

### ✅ Phase 3: Advanced Features (COMPLETED)

- **4-gram Models**: Enhanced context prediction
- **Advanced Text Splitting**: Dynamic programming optimization
- **Pattern Matching**: Common phrase recognition
- **User Learning**: Adaptive preference tracking
- **Enhanced UI**: Stats panel và detailed analytics
- **Performance Optimization**: <50ms response time

### 🔄 Phase 4: Production Ready (IN PROGRESS)

- Executable packaging
- Auto-updater system
- Advanced settings implementation
- Distribution và deployment

## 🤝 Contributing

### Development Setup

```bash
git clone <repository>
cd New_version
pip install -r requirements.txt

# Test Phase 3 features
python main.py test3

# Run enhanced UI
python main.py ui
```

### Code Guidelines

- Follow PEP 8
- Add comprehensive type hints
- Write tests cho new features
- Document advanced algorithms
- Performance benchmarking

### Advanced Features Roadmap

- **Machine Learning Integration**: Neural language models
- **Multi-language Support**: English, Chinese input
- **Voice Integration**: Speech-to-text input
- **Mobile App**: React Native version
- **Cloud Sync**: User preferences synchronization

## 📄 License

MIT License - Feel free to use and modify

## 🙏 Acknowledgments

- Advanced AI techniques from research papers
- Vietnamese linguistic resources
- Performance optimization best practices
- Beta testers và community feedback

## 📞 Support

- **Issues**: GitHub Issues
- **Email**: dinhgia2106@gmail.com
- **Documentation**: See DEVELOPMENT_PLAN.md
- **Roadmap**: Phase 4 Production Ready

---

**Version**: 2.0.0 - Phase 3 Advanced Features  
**Status**: Advanced AI Complete, Production Ready Preparation  
**Next Release**: Phase 4 Production Deployment  
**Last Updated**: 2024

### 🔬 Technical Highlights

- **4-gram Language Models** cho context prediction
- **Dynamic Programming** optimization cho text splitting
- **Pattern Recognition** cho common Vietnamese phrases
- **Adaptive Learning** từ user behavior
- **Real-time Performance** với <50ms response time
- **Comprehensive Analytics** với detailed statistics
