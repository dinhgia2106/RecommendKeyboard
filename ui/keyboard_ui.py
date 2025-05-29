"""
Keyboard UI với Tkinter - Giao diện bàn phím recommend tiếng Việt
Version 2.0 - Enhanced với advanced features
"""

import tkinter as tk
from tkinter import ttk, font, messagebox
import sys
import os
import time
from typing import List, Tuple

# Thêm đường dẫn để import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AdvancedRecommender


class AdvancedKeyboardUI:
    def __init__(self):
        self.root = tk.Tk()
        self.recommender = AdvancedRecommender()
        self.context = []
        
        # Performance tracking
        self.session_stats = {
            "suggestions_shown": 0,
            "selections_made": 0,
            "session_start": time.time(),
            "response_times": []
        }
        
        self.setup_ui()
        self.setup_styles()
        
    def setup_ui(self):
        """Thiết lập giao diện"""
        self.root.title("🚀 Bàn Phím Recommend Tiếng Việt v2.0 - Advanced")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Set icon (nếu có)
        try:
            self.root.iconbitmap('assets/icon.ico')
        except:
            pass
        
        # Header
        self.create_header()
        
        # Stats panel (new!)
        self.create_stats_panel()
        
        # Input section
        self.create_input_section()
        
        # Suggestions section
        self.create_suggestions_section()
        
        # Output section
        self.create_output_section()
        
        # Control buttons
        self.create_control_section()
        
        # Status bar
        self.create_status_bar()
        
    def setup_styles(self):
        """Thiết lập styles và themes"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Custom styles
        self.style.configure('Header.TLabel', 
                           font=('Segoe UI', 18, 'bold'),
                           background='#2c3e50',
                           foreground='white',
                           padding=10)
        
        self.style.configure('Stats.TLabel',
                           font=('Segoe UI', 9),
                           background='#3498db',
                           foreground='white',
                           padding=5)
        
    def create_header(self):
        """Tạo header"""
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="🚀 Bàn Phím Recommend Tiếng Việt v2.0",
                              font=('Segoe UI', 16, 'bold'),
                              bg='#2c3e50',
                              fg='white')
        title_label.pack(pady=15)
        
        subtitle_label = tk.Label(header_frame,
                                 text="Advanced AI với 4-gram models & Pattern matching",
                                 font=('Segoe UI', 9),
                                 bg='#2c3e50',
                                 fg='#bdc3c7')
        subtitle_label.pack()
        
    def create_stats_panel(self):
        """Tạo panel hiển thị statistics (New!)"""
        stats_frame = tk.Frame(self.root, bg='#3498db', height=60)
        stats_frame.pack(fill='x', padx=10, pady=(0, 10))
        stats_frame.pack_propagate(False)
        
        # Create stats labels
        stats_container = tk.Frame(stats_frame, bg='#3498db')
        stats_container.pack(expand=True, fill='both')
        
        # Dictionary stats
        dict_stats = self.recommender.get_statistics()
        
        self.stats_labels = {}
        stats_info = [
            ("Từ vựng", f"{dict_stats['word_count']} words"),
            ("4-grams", f"{dict_stats['fourgram_count']} patterns"),
            ("User prefs", f"{dict_stats['user_preferences']} learned"),
            ("Session", "0 selections"),
            ("Accuracy", "N/A")
        ]
        
        for i, (label, value) in enumerate(stats_info):
            stat_frame = tk.Frame(stats_container, bg='#3498db')
            stat_frame.pack(side='left', expand=True, fill='both', padx=5, pady=10)
            
            tk.Label(stat_frame, text=label, font=('Segoe UI', 8, 'bold'), 
                    bg='#3498db', fg='white').pack()
            
            value_label = tk.Label(stat_frame, text=value, font=('Segoe UI', 10), 
                                  bg='#3498db', fg='#ecf0f1')
            value_label.pack()
            
            self.stats_labels[label.lower().replace(" ", "_")] = value_label
        
    def create_input_section(self):
        """Tạo phần input"""
        input_frame = tk.Frame(self.root, bg='#f0f0f0')
        input_frame.pack(fill='x', padx=20, pady=10)
        
        input_label = tk.Label(input_frame,
                              text="📝 Nhập text:",
                              font=('Segoe UI', 12, 'bold'),
                              bg='#f0f0f0')
        input_label.pack(anchor='w', pady=(0, 5))
        
        # Input entry với style đẹp
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(input_frame,
                                   textvariable=self.input_var,
                                   font=('Segoe UI', 14),
                                   relief='flat',
                                   bd=2,
                                   highlightthickness=2,
                                   highlightcolor='#3498db')
        self.input_entry.pack(fill='x', pady=(0, 10), ipady=8)
        
        # Bind events
        self.input_var.trace('w', self.on_input_change)
        self.input_entry.bind('<Return>', self.on_enter_pressed)
        
        # Advanced examples
        example_label = tk.Label(input_frame,
                                text="💡 Thử nghiệm: toihoctiengviet, anhyeuemdennaychungcothe, chucmungnamoi",
                                font=('Segoe UI', 9, 'italic'),
                                bg='#f0f0f0',
                                fg='#7f8c8d')
        example_label.pack(anchor='w')
        
    def create_suggestions_section(self):
        """Tạo phần suggestions"""
        suggestions_frame = tk.Frame(self.root, bg='#f0f0f0')
        suggestions_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        suggestions_label = tk.Label(suggestions_frame,
                                    text="💡 Gợi ý thông minh:",
                                    font=('Segoe UI', 12, 'bold'),
                                    bg='#f0f0f0')
        suggestions_label.pack(anchor='w', pady=(0, 10))
        
        # Canvas với scrollbar cho suggestions
        canvas_frame = tk.Frame(suggestions_frame, bg='white', relief='solid', bd=1)
        canvas_frame.pack(fill='both', expand=True)
        
        self.suggestions_canvas = tk.Canvas(canvas_frame, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical', command=self.suggestions_canvas.yview)
        self.suggestions_frame = tk.Frame(self.suggestions_canvas, bg='white')
        
        self.suggestions_frame.bind('<Configure>',
                                   lambda e: self.suggestions_canvas.configure(scrollregion=self.suggestions_canvas.bbox("all")))
        
        self.suggestions_canvas.create_window((0, 0), window=self.suggestions_frame, anchor="nw")
        self.suggestions_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.suggestions_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Placeholder
        self.show_placeholder()
        
    def create_output_section(self):
        """Tạo phần output"""
        output_frame = tk.Frame(self.root, bg='#f0f0f0')
        output_frame.pack(fill='x', padx=20, pady=10)
        
        output_label = tk.Label(output_frame,
                               text="✅ Kết quả:",
                               font=('Segoe UI', 12, 'bold'),
                               bg='#f0f0f0')
        output_label.pack(anchor='w', pady=(0, 5))
        
        # Text widget để hiển thị kết quả
        self.output_text = tk.Text(output_frame,
                                  height=3,
                                  font=('Segoe UI', 12),
                                  relief='flat',
                                  bd=2,
                                  highlightthickness=2,
                                  highlightcolor='#27ae60',
                                  wrap='word')
        self.output_text.pack(fill='x', pady=(0, 10))
        
    def create_control_section(self):
        """Tạo phần controls"""
        control_frame = tk.Frame(self.root, bg='#f0f0f0')
        control_frame.pack(fill='x', padx=20, pady=10)
        
        # Buttons frame
        buttons_frame = tk.Frame(control_frame, bg='#f0f0f0')
        buttons_frame.pack()
        
        # Clear button
        clear_btn = tk.Button(buttons_frame,
                             text="🗑️ Xóa",
                             font=('Segoe UI', 10),
                             bg='#e74c3c',
                             fg='white',
                             relief='flat',
                             padx=20,
                             pady=8,
                             command=self.clear_all)
        clear_btn.pack(side='left', padx=(0, 10))
        
        # Copy button
        copy_btn = tk.Button(buttons_frame,
                            text="📋 Copy",
                            font=('Segoe UI', 10),
                            bg='#95a5a6',
                            fg='white',
                            relief='flat',
                            padx=20,
                            pady=8,
                            command=self.copy_result)
        copy_btn.pack(side='left', padx=(0, 10))
        
        # Stats button (New!)
        stats_btn = tk.Button(buttons_frame,
                             text="📊 Thống kê",
                             font=('Segoe UI', 10),
                             bg='#3498db',
                             fg='white',
                             relief='flat',
                             padx=20,
                             pady=8,
                             command=self.show_detailed_stats)
        stats_btn.pack(side='left', padx=(0, 10))
        
        # Settings button
        settings_btn = tk.Button(buttons_frame,
                                text="⚙️ Cài đặt",
                                font=('Segoe UI', 10),
                                bg='#34495e',
                                fg='white',
                                relief='flat',
                                padx=20,
                                pady=8,
                                command=self.show_settings)
        settings_btn.pack(side='left')
        
    def create_status_bar(self):
        """Tạo status bar"""
        dict_stats = self.recommender.get_statistics()
        total_entries = dict_stats['word_count'] + len(self.recommender.dictionary.phrases)
        
        self.status_bar = tk.Label(self.root,
                                  text=f"Sẵn sàng | Enhanced AI với {total_entries} entries | 4-gram patterns: {dict_stats['fourgram_count']}",
                                  font=('Segoe UI', 9),
                                  bg='#34495e',
                                  fg='white',
                                  anchor='w',
                                  padx=10)
        self.status_bar.pack(fill='x', side='bottom')
        
    def show_placeholder(self):
        """Hiển thị placeholder cho suggestions"""
        placeholder = tk.Label(self.suggestions_frame,
                              text="Nhập text để xem gợi ý AI thông minh...\n🧠 4-gram models • 🎯 Pattern matching • 📈 User learning",
                              font=('Segoe UI', 12, 'italic'),
                              bg='white',
                              fg='#bdc3c7',
                              pady=50)
        placeholder.pack(fill='both', expand=True)
        
    def on_input_change(self, *args):
        """Xử lý khi input thay đổi"""
        user_input = self.input_var.get().strip()
        
        if not user_input:
            self.clear_suggestions()
            self.show_placeholder()
            return
        
        # Debounce - chỉ search khi có ít nhất 2 ký tự
        if len(user_input) >= 2:
            start_time = time.time()
            self.update_suggestions(user_input)
            response_time = time.time() - start_time
            self.session_stats["response_times"].append(response_time)
        
    def update_suggestions(self, user_input: str):
        """Cập nhật suggestions với enhanced features"""
        self.clear_suggestions()
        
        try:
            # Lấy enhanced recommendations
            recommendations = self.recommender.smart_recommend(
                user_input, 
                self.context, 
                max_suggestions=8
            )
            
            if not recommendations:
                no_result = tk.Label(self.suggestions_frame,
                                    text="❌ Không tìm thấy gợi ý phù hợp\n💡 Thử nghiệm với: 'toihoc', 'xinchao', 'chucmung'",
                                    font=('Segoe UI', 11),
                                    bg='white',
                                    fg='#e74c3c',
                                    pady=20)
                no_result.pack()
                return
            
            # Hiển thị suggestions với advanced info
            for i, (text, confidence, rec_type) in enumerate(recommendations):
                self.create_advanced_suggestion_button(i + 1, text, confidence, rec_type)
            
            # Update session stats
            self.session_stats["suggestions_shown"] += len(recommendations)
            self.update_session_stats_display()
            
            # Update status
            avg_confidence = sum(conf for _, conf, _ in recommendations) / len(recommendations)
            self.update_status(f"Tìm thấy {len(recommendations)} gợi ý | Độ tin cậy TB: {avg_confidence:.2f}")
            
        except Exception as e:
            error_label = tk.Label(self.suggestions_frame,
                                  text=f"❌ Lỗi: {str(e)}",
                                  font=('Segoe UI', 11),
                                  bg='white',
                                  fg='#e74c3c',
                                  pady=20)
            error_label.pack()
            
    def create_advanced_suggestion_button(self, index: int, text: str, confidence: float, rec_type: str):
        """Tạo button cho suggestion với advanced info"""
        suggestion_frame = tk.Frame(self.suggestions_frame, bg='white')
        suggestion_frame.pack(fill='x', padx=10, pady=2)
        
        # Main button với enhanced styling
        btn_text = f"{index}. {text}"
        suggestion_btn = tk.Button(suggestion_frame,
                                  text=btn_text,
                                  font=('Segoe UI', 11, 'bold' if confidence > 0.8 else 'normal'),
                                  bg='white',
                                  fg='#2c3e50',
                                  relief='solid',
                                  bd=1,
                                  anchor='w',
                                  padx=15,
                                  pady=8,
                                  command=lambda: self.select_advanced_suggestion(text, confidence, rec_type))
        suggestion_btn.pack(fill='x', pady=1)
        
        # Advanced info frame
        info_frame = tk.Frame(suggestion_frame, bg='white')
        info_frame.pack(fill='x', padx=15)
        
        # Confidence bar visual
        confidence_width = int(confidence * 200)
        confidence_color = self.get_confidence_color(confidence)
        
        confidence_frame = tk.Frame(info_frame, bg='#ecf0f1', height=4)
        confidence_frame.pack(fill='x', pady=(2, 0))
        
        confidence_bar = tk.Frame(confidence_frame, bg=confidence_color, height=4)
        confidence_bar.place(x=0, y=0, width=confidence_width)
        
        # Enhanced info text với algorithm type
        algo_info = self.get_algorithm_description(rec_type)
        info_text = f"Độ tin cậy: {confidence:.3f} | {algo_info}"
        info_label = tk.Label(info_frame,
                             text=info_text,
                             font=('Segoe UI', 8),
                             bg='white',
                             fg='#7f8c8d')
        info_label.pack(anchor='w', pady=(2, 5))
        
        # Enhanced hover effects
        def on_enter(e):
            suggestion_btn.configure(bg='#ecf0f1', relief='raised')
            
        def on_leave(e):
            suggestion_btn.configure(bg='white', relief='solid')
            
        suggestion_btn.bind('<Enter>', on_enter)
        suggestion_btn.bind('<Leave>', on_leave)
    
    def get_algorithm_description(self, rec_type: str) -> str:
        """Lấy mô tả thuật toán"""
        descriptions = {
            "dict_exact": "🎯 Khớp chính xác",
            "dict_prefix": "🔍 Khớp tiền tố",
            "dict_fuzzy": "🧩 Khớp gần đúng",
            "dict_contains": "📝 Chứa chuỗi con",
            "advanced_split": "🧠 AI Text Splitting",
            "pattern_match": "🎨 Pattern Matching",
            "context_extend": "📈 Context Extension",
            "context_prepend": "⬅️ Context Prepend",
            "context_predict": "🔮 Context Prediction"
        }
        return descriptions.get(rec_type, f"❓ {rec_type}")
        
    def get_confidence_color(self, confidence: float) -> str:
        """Lấy màu dựa trên confidence"""
        if confidence >= 0.9:
            return '#27ae60'  # Dark green
        elif confidence >= 0.8:
            return '#2ecc71'  # Green
        elif confidence >= 0.6:
            return '#f39c12'  # Orange
        elif confidence >= 0.4:
            return '#e67e22'  # Dark orange
        else:
            return '#e74c3c'  # Red
            
    def select_advanced_suggestion(self, text: str, confidence: float, rec_type: str):
        """Xử lý khi user chọn suggestion với advanced tracking"""
        # Cập nhật output
        current_output = self.output_text.get('1.0', 'end-1c')
        if current_output:
            new_output = current_output + " " + text
        else:
            new_output = text
            
        self.output_text.delete('1.0', 'end')
        self.output_text.insert('1.0', new_output)
        
        # Clear input
        self.input_var.set("")
        
        # Update context và enhanced learning
        self.context.extend(text.split())
        if len(self.context) > 15:  # Increased context window
            self.context = self.context[-15:]
            
        self.recommender.update_user_preferences(text, self.context)
        
        # Update session stats
        self.session_stats["selections_made"] += 1
        self.update_session_stats_display()
        
        # Clear suggestions
        self.clear_suggestions()
        self.show_placeholder()
        
        # Enhanced status update
        algo_desc = self.get_algorithm_description(rec_type)
        self.update_status(f"✅ Đã chọn: '{text}' | {algo_desc} | Confidence: {confidence:.3f}")
        
        # Focus lại input
        self.input_entry.focus_set()
        
    def update_session_stats_display(self):
        """Cập nhật hiển thị session stats"""
        # Update selection count
        self.stats_labels["session"].configure(text=f"{self.session_stats['selections_made']} selections")
        
        # Calculate and update accuracy
        if self.session_stats["selections_made"] > 0:
            accuracy = (self.session_stats["selections_made"] / max(self.session_stats["suggestions_shown"], 1)) * 100
            self.stats_labels["accuracy"].configure(text=f"{accuracy:.1f}%")
        
        # Update user preferences count
        dict_stats = self.recommender.get_statistics()
        self.stats_labels["user_prefs"].configure(text=f"{dict_stats['user_preferences']} learned")
        
    def show_detailed_stats(self):
        """Hiển thị thống kê chi tiết"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("📊 Thống kê chi tiết")
        stats_window.geometry("600x500")
        stats_window.configure(bg='#f0f0f0')
        
        # Header
        header_label = tk.Label(stats_window,
                               text="📊 Thống kê chi tiết AI Engine",
                               font=('Segoe UI', 16, 'bold'),
                               bg='#f0f0f0')
        header_label.pack(pady=20)
        
        # Stats text widget
        stats_text = tk.Text(stats_window,
                            font=('Segoe UI', 10),
                            bg='white',
                            relief='solid',
                            bd=1,
                            wrap='word')
        stats_text.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Generate detailed stats
        dict_stats = self.recommender.get_statistics()
        session_time = time.time() - self.session_stats["session_start"]
        avg_response_time = sum(self.session_stats["response_times"]) / max(len(self.session_stats["response_times"]), 1)
        
        stats_content = f"""
🧠 AI ENGINE STATISTICS
{'='*50}

📚 Dictionary:
  • Từ vựng: {dict_stats['word_count']:,} words
  • Cụm từ: {len(self.recommender.dictionary.phrases):,} phrases
  • Tổng entries: {dict_stats['word_count'] + len(self.recommender.dictionary.phrases):,}

🔗 N-gram Models:
  • Bigrams: {dict_stats['bigram_count']:,}
  • Trigrams: {dict_stats['trigram_count']:,} 
  • 4-grams: {dict_stats['fourgram_count']:,}

👤 User Learning:
  • Preferences learned: {dict_stats['user_preferences']}
  • Context cache: {dict_stats['cache_size']} entries

📈 Session Performance:
  • Session time: {session_time/60:.1f} minutes
  • Suggestions shown: {self.session_stats['suggestions_shown']}
  • Selections made: {self.session_stats['selections_made']}
  • Selection rate: {(self.session_stats['selections_made']/max(self.session_stats['suggestions_shown'], 1)*100):.1f}%
  • Avg response time: {avg_response_time*1000:.1f}ms

🏆 Top Words:
"""
        
        for i, (word, freq) in enumerate(dict_stats['top_words'][:5], 1):
            stats_content += f"  {i}. {word}: {freq} times\n"
        
        if dict_stats['top_preferences']:
            stats_content += "\n🎯 Top User Preferences:\n"
            for i, (word, pref) in enumerate(dict_stats['top_preferences'][:5], 1):
                stats_content += f"  {i}. {word}: {pref:.2f} score\n"
        
        stats_text.insert('1.0', stats_content)
        stats_text.configure(state='disabled')
        
    def clear_suggestions(self):
        """Xóa tất cả suggestions"""
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()
            
    def on_enter_pressed(self, event):
        """Xử lý khi nhấn Enter"""
        user_input = self.input_var.get().strip()
        if user_input:
            # Lấy suggestion đầu tiên nếu có
            try:
                recommendations = self.recommender.smart_recommend(user_input, self.context, max_suggestions=1)
                if recommendations:
                    text, confidence, rec_type = recommendations[0]
                    self.select_advanced_suggestion(text, confidence, rec_type)
            except:
                pass
                
    def clear_all(self):
        """Xóa tất cả"""
        self.input_var.set("")
        self.output_text.delete('1.0', 'end')
        self.context.clear()
        self.clear_suggestions()
        self.show_placeholder()
        self.update_status("Đã xóa tất cả - Session reset")
        
    def copy_result(self):
        """Copy kết quả"""
        result = self.output_text.get('1.0', 'end-1c')
        if result:
            self.root.clipboard_clear()
            self.root.clipboard_append(result)
            self.update_status("✅ Đã copy kết quả vào clipboard")
        else:
            self.update_status("❌ Không có gì để copy")
            
    def show_settings(self):
        """Hiển thị cài đặt advanced"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("⚙️ Cài đặt Advanced")
        settings_window.geometry("500x400")
        settings_window.configure(bg='#f0f0f0')
        
        settings_label = tk.Label(settings_window,
                                 text="⚙️ Cài đặt Advanced",
                                 font=('Segoe UI', 16, 'bold'),
                                 bg='#f0f0f0')
        settings_label.pack(pady=20)
        
        # Settings options (placeholder for Phase 4)
        settings_info = [
            "🧠 AI Model Configuration",
            "📚 Dictionary Management", 
            "🎯 User Preference Tuning",
            "⚡ Performance Optimization",
            "🎨 UI Themes & Appearance",
            "📊 Export/Import Settings",
            "🔄 Auto-update Configuration"
        ]
        
        for setting in settings_info:
            setting_label = tk.Label(settings_window,
                                   text=f"• {setting}",
                                   font=('Segoe UI', 11),
                                   bg='#f0f0f0',
                                   fg='#7f8c8d',
                                   anchor='w')
            setting_label.pack(fill='x', padx=40, pady=5)
        
        info_label = tk.Label(settings_window,
                             text="🚧 Advanced settings sẽ được hoàn thiện trong Phase 4\n📅 Production Ready",
                             font=('Segoe UI', 11, 'italic'),
                             bg='#f0f0f0',
                             fg='#3498db')
        info_label.pack(pady=30)
        
    def update_status(self, message: str):
        """Cập nhật status bar"""
        self.status_bar.configure(text=message)
        
    def run(self):
        """Chạy ứng dụng"""
        self.root.mainloop()


# Backward compatibility
class KeyboardUI(AdvancedKeyboardUI):
    """Backward compatibility class"""
    pass


def main():
    """Main function để chạy UI"""
    app = AdvancedKeyboardUI()
    app.run()


if __name__ == "__main__":
    main() 