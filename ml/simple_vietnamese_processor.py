#!/usr/bin/env python3
"""
Simple Vietnamese Text Processor
Hiệu quả, đơn giản, chính xác cho bàn phím tiếng Việt
"""

from typing import List, Dict, Tuple


class SimpleVietnameseProcessor:
    """Hệ thống xử lý tiếng Việt đơn giản và hiệu quả"""

    def __init__(self):
        # Từ điển âm tiết cơ bản
        self.syllables = {
            # Đại từ
            'toi': 'tôi', 'ban': 'bạn', 'anh': 'anh', 'chi': 'chị', 'em': 'em',

            # Động từ thường dùng
            'la': 'là', 'co': 'có', 'hoc': 'học', 'lam': 'làm', 'di': 'đi', 've': 'về',
            'an': 'ăn', 'uong': 'uống', 'ngu': 'ngủ', 'xem': 'xem', 'choi': 'chơi',
            'doc': 'đọc', 'viet': 'viết', 'nghe': 'nghe', 'noi': 'nói',

            # Tính từ
            'tot': 'tốt', 'xau': 'xấu', 'dep': 'đẹp', 'lon': 'lớn', 'nho': 'nhỏ',
            'moi': 'mới', 'cu': 'cũ', 'hay': 'hay', 'te': 'tệ',

            # Thời gian
            'hom': 'hôm', 'nay': 'nay', 'qua': 'qua', 'mai': 'mai',
            'ngay': 'ngày', 'dem': 'đêm', 'sang': 'sáng', 'chieu': 'chiều', 'toi_time': 'tối',
            'gio': 'giờ', 'phut': 'phút', 'thang': 'tháng',

            # Giáo dục
            'sinh': 'sinh', 'vien': 'viên', 'bai': 'bài', 'tap': 'tập',
            'truong': 'trường', 'lop': 'lớp', 'giao': 'giáo', 'thi': 'thi',

            # Số đếm
            'mot': 'một', 'hai': 'hai', 'ba': 'ba', 'bon': 'bốn', 'nam': 'năm',
            'sau': 'sáu', 'bay': 'bảy', 'tam': 'tám', 'chin': 'chín',
            'nhat': 'nhất', 'nhi': 'nhì', 'thu': 'thứ',

            # Công nghệ
            'may': 'máy', 'tinh': 'tính', 'dien': 'điện', 'thoai': 'thoại',
            'bo': 'bộ', 'go': 'gõ', 'phan': 'phần', 'mem': 'mềm',

            # Gia đình
            'ba': 'ba', 'me': 'mẹ', 'con': 'con', 'gia': 'gia', 'dinh': 'đình',

            # Nơi chốn
            'nha': 'nhà', 'phong': 'phòng', 'cho': 'chợ',

            # Từ ngữ khác
            'rat': 'rất', 'qua_adv': 'quá', 'nhieu': 'nhiều', 'it': 'ít',
            'da': 'đã', 'se': 'sẽ', 'roi': 'rồi', 'chua': 'chưa',
            'vi': 'vì', 'ma': 'mà', 'neu': 'nếu', 'nhu': 'như',
            'cua': 'của', 'voi': 'với', 'trong': 'trong', 'ngoai': 'ngoài'
        }

        # Từ ghép và cụm từ phổ biến
        self.compounds = {
            # Cụm từ học tập
            'hocbai': 'học bài',
            'baitap': 'bài tập',
            'sinhvien': 'sinh viên',
            'giaovien': 'giáo viên',
            'truonghoc': 'trường học',
            'lophoc': 'lớp học',

            # Cụm từ thời gian
            'homnay': 'hôm nay',
            'homqua': 'hôm qua',
            'ngaymai': 'ngày mai',
            'namnay': 'năm nay',

            # Cụm từ công nghệ
            'maytinh': 'máy tính',
            'dienthoai': 'điện thoại',
            'bogo': 'bộ gõ',
            'phanmem': 'phần mềm',

            # Cụm từ gia đình
            'giadinh': 'gia đình',
            'bame': 'ba mẹ',

            # Hoạt động
            'xemphim': 'xem phim',
            'nghenhac': 'nghe nhạc',
            'choigame': 'chơi game',
            'ancom': 'ăn cơm',
            'dihoc': 'đi học',
            'dilam': 'đi làm',
            'venha': 'về nhà'
        }

        # Câu hoàn chỉnh thường gặp
        self.sentences = {
            # Với "tôi"
            'toila': 'tôi là',
            'toihoc': 'tôi học',
            'toilam': 'tôi làm',
            'toidi': 'tôi đi',
            'toive': 'tôi về',
            'toian': 'tôi ăn',
            'toixem': 'tôi xem',
            'toichoi': 'tôi chơi',
            'toinam': 'tôi năm',

            # Câu dài với "tôi học bài"
            'toihocbai': 'tôi học bài',
            'toilambaitap': 'tôi làm bài tập',

            # Câu với "tôi là sinh viên"
            'toilasinhvien': 'tôi là sinh viên',
            'toilagiaovien': 'tôi là giáo viên',

            # Câu với thời gian
            'homnaytoihoc': 'hôm nay tôi học',
            'homnaytoilam': 'hôm nay tôi làm',
            'homnaytoidi': 'hôm nay tôi đi',
            'homnaytoive': 'hôm nay tôi về',
            'homnaytoixem': 'hôm nay tôi xem',

            # Câu phức tạp
            'toihocbaihomnay': 'tôi học bài hôm nay',
            'toilambaitaphomnay': 'tôi làm bài tập hôm nay',
            'toiladihochomnay': 'tôi là đi học hôm nay',

            # Sinh viên năm
            'sinhviennamnhat': 'sinh viên năm nhất',
            'sinhviennamhai': 'sinh viên năm hai',
            'sinhviennamva': 'sinh viên năm ba',

            # Câu với động từ xem
            'xemphimhomnay': 'xem phim hôm nay',
            'toixemphimhomnay': 'tôi xem phim hôm nay',

            # Câu với đi học
            'dihochomnay': 'đi học hôm nay',
            'toidihochomnay': 'tôi đi học hôm nay',

            # Câu với ăn cơm
            'ancomroidi': 'ăn cơm rồi đi',
            'toiancomroidi': 'tôi ăn cơm rồi đi',

            # Câu với bài tập
            'baitaptoingay': 'bài tập tối ngày',
            'lambaitaptoingay': 'làm bài tập tối ngày'
        }

        print(
            f"✅ Loaded {len(self.syllables)} syllables, {len(self.compounds)} compounds, {len(self.sentences)} sentences")

    def process_text(self, input_text: str, max_suggestions: int = 3) -> List[Dict]:
        """
        Xử lý văn bản đầu vào và trả về các gợi ý

        Args:
            input_text: Văn bản không dấu
            max_suggestions: Số gợi ý tối đa

        Returns:
            Danh sách gợi ý với confidence
        """
        if not input_text:
            return []

        input_text = input_text.lower().strip()
        suggestions = []

        # Phương pháp 1: Kiểm tra câu hoàn chỉnh
        if input_text in self.sentences:
            suggestions.append({
                'vietnamese_text': self.sentences[input_text],
                'confidence': 95,
                'method': 'complete_sentence'
            })

        # Phương pháp 2: Kiểm tra từ ghép
        if input_text in self.compounds:
            if len(suggestions) == 0 or suggestions[0]['vietnamese_text'] != self.compounds[input_text]:
                suggestions.append({
                    'vietnamese_text': self.compounds[input_text],
                    'confidence': 88,
                    'method': 'compound_word'
                })

        # Phương pháp 3: Phân tách theo âm tiết
        syllable_result = self._segment_by_syllables(input_text)
        if syllable_result and len(suggestions) < max_suggestions:
            suggestions.append({
                'vietnamese_text': syllable_result,
                'confidence': 75,
                'method': 'syllable_segmentation'
            })

        # Phương pháp 4: Tìm kiếm thông minh
        smart_results = self._smart_search(input_text)
        for result in smart_results:
            if len(suggestions) < max_suggestions:
                # Kiểm tra không trùng lặp
                exists = any(s['vietnamese_text'] ==
                             result['vietnamese_text'] for s in suggestions)
                if not exists:
                    suggestions.append(result)

        return suggestions[:max_suggestions]

    def _segment_by_syllables(self, text: str) -> str:
        """Phân tách text thành các âm tiết"""
        result = []
        i = 0

        while i < len(text):
            found = False

            # Thử từ dài nhất trước
            for length in range(min(6, len(text) - i), 0, -1):
                substr = text[i:i + length]
                if substr in self.syllables:
                    result.append(self.syllables[substr])
                    i += length
                    found = True
                    break

            if not found:
                # Thêm ký tự không nhận diện được
                result.append(text[i])
                i += 1

        return ' '.join(result)

    def _smart_search(self, text: str) -> List[Dict]:
        """Tìm kiếm thông minh với các biến thể"""
        results = []

        # Tìm các câu chứa từ khóa
        for sentence_key, sentence_value in self.sentences.items():
            if text in sentence_key and len(text) >= 4:
                similarity = len(text) / len(sentence_key)
                if similarity > 0.6:  # Độ tương đồng > 60%
                    confidence = int(similarity * 85)
                    results.append({
                        'vietnamese_text': sentence_value,
                        'confidence': confidence,
                        'method': 'smart_match'
                    })

        # Tìm từ ghép tương tự
        for compound_key, compound_value in self.compounds.items():
            if text in compound_key and len(text) >= 3:
                similarity = len(text) / len(compound_key)
                if similarity > 0.7:  # Độ tương đồng > 70%
                    confidence = int(similarity * 80)
                    results.append({
                        'vietnamese_text': compound_value,
                        'confidence': confidence,
                        'method': 'smart_compound'
                    })

        # Sắp xếp theo confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:2]  # Chỉ lấy 2 kết quả tốt nhất

    def get_best_suggestion(self, input_text: str) -> str:
        """Lấy gợi ý tốt nhất"""
        suggestions = self.process_text(input_text, 1)
        if suggestions:
            return suggestions[0]['vietnamese_text']
        return input_text
