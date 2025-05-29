"""
Keyboard UI với Tkinter - Giao diện bàn phím recommend tiếng Việt
Version 2.1 - Production Ready với Performance Optimization
"""

import tkinter as tk
from tkinter import ttk, font, messagebox
import sys
import os
import time
import threading
from typing import List, Tuple

# Thêm đường dẫn để import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import AdvancedRecommender


class AdvancedKeyboardUI:
    def __init__(self):
        self.root = tk.Tk()
        self.recommender = AdvancedRecommender()
        self.context = []
        
        # Performance optimization features
        self.debounce_timer = None
        self.debounce_delay = 150  # 150ms debounce
        self.is_processing = False
        self.last_input = ""
        self.suggestion_thread = None
        
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
        self.root.title("🚀 Bàn Phím Recommend Tiếng Việt v2.1 - Production Ready")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # Set icon (nếu có)
        try:
            self.root.iconbitmap('assets/icon.ico')
        except:
            pass
        
        # Header
        self.create_header()
        
        # Stats panel
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
                              text="🚀 Bàn Phím Recommend Tiếng Việt v2.1",
                              font=('Segoe UI', 16, 'bold'),
                              bg='#2c3e50',
                              fg='white')
        title_label.pack(pady=15)
        
        subtitle_label = tk.Label(header_frame,
                                 text="Production Ready • Optimized Performance • Smart Caching",
                                 font=('Segoe UI', 9),
                                 bg='#2c3e50',
                                 fg='#bdc3c7')
        subtitle_label.pack()
        
    def create_stats_panel(self):
        """Tạo panel hiển thị statistics với performance metrics"""
        stats_frame = tk.Frame(self.root, bg='#3498db', height=60)
        stats_frame.pack(fill='x', padx=10, pady=(0, 10))
        stats_frame.pack_propagate(False)
        
        # Create stats labels
        stats_container = tk.Frame(stats_frame, bg='#3498db')
        stats_container.pack(expand=True, fill='both')
        
        # Enhanced dictionary stats
        dict_stats = self.recommender.get_statistics()
        perf_stats = self.recommender.get_performance_stats()
        
        self.stats_labels = {}
        stats_info = [
            ("Từ vựng", f"{dict_stats['word_count']} words"),
            ("Cache", f"{perf_stats['cache_sizes']['recommendations']} cached"),
            ("Avg Response", f"{perf_stats['avg_response_time_ms']:.1f}ms"),
            ("Session", "0 selections"),
            ("Status", "Ready")
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
        """Tạo phần input với optimized event handling"""
        input_frame = tk.Frame(self.root, bg='#f0f0f0')
        input_frame.pack(fill='x', padx=20, pady=10)
        
        input_label = tk.Label(input_frame,
                              text="📝 Nhập text (Optimized typing):",
                              font=('Segoe UI', 12, 'bold'),
                              bg='#f0f0f0')
        input_label.pack(anchor='w', pady=(0, 5))
        
        # Input entry với optimized debouncing
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(input_frame,
                                   textvariable=self.input_var,
                                   font=('Segoe UI', 14),
                                   relief='flat',
                                   bd=2,
                                   highlightthickness=2,
                                   highlightcolor='#3498db')
        self.input_entry.pack(fill='x', pady=(0, 10), ipady=8)
        
        # Optimized event binding
        self.input_var.trace('w', self.on_input_change_debounced)
        self.input_entry.bind('<Return>', self.on_enter_pressed)
        self.input_entry.bind('<KeyPress>', self.on_key_press)
        
        # Performance tips
        example_label = tk.Label(input_frame,
                                text="⚡ Optimized: Debounced input • Smart caching • Background processing",
                                font=('Segoe UI', 9, 'italic'),
                                bg='#f0f0f0',
                                fg='#7f8c8d')
        example_label.pack(anchor='w')
        
    def create_suggestions_section(self):
        """Tạo phần suggestions với loading indicator"""
        suggestions_frame = tk.Frame(self.root, bg='#f0f0f0')
        suggestions_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Header với loading indicator
        header_frame = tk.Frame(suggestions_frame, bg='#f0f0f0')
        header_frame.pack(fill='x', pady=(0, 10))
        
        suggestions_label = tk.Label(header_frame,
                                    text="💡 Gợi ý thông minh:",
                                    font=('Segoe UI', 12, 'bold'),
                                    bg='#f0f0f0')
        suggestions_label.pack(side='left')
        
        # Loading indicator
        self.loading_label = tk.Label(header_frame,
                                     text="",
                                     font=('Segoe UI', 10),
                                     bg='#f0f0f0',
                                     fg='#3498db')
        self.loading_label.pack(side='right')
        
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
        
        # Performance stats button
        perf_btn = tk.Button(buttons_frame,
                            text="⚡ Performance",
                            font=('Segoe UI', 10),
                            bg='#f39c12',
                            fg='white',
                            relief='flat',
                            padx=20,
                            pady=8,
                            command=self.show_performance_stats)
        perf_btn.pack(side='left', padx=(0, 10))
        
        # Stats button
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
        """Tạo status bar với performance info"""
        perf_stats = self.recommender.get_performance_stats()
        dict_stats = self.recommender.get_statistics()
        total_entries = dict_stats['word_count'] + len(self.recommender.dictionary.phrases)
        
        self.status_bar = tk.Label(self.root,
                                  text=f"Production Ready | {total_entries} entries | Avg: {perf_stats['avg_response_time_ms']:.1f}ms | Cache: {perf_stats['cache_sizes']['recommendations']}",
                                  font=('Segoe UI', 9),
                                  bg='#34495e',
                                  fg='white',
                                  anchor='w',
                                  padx=10)
        self.status_bar.pack(fill='x', side='bottom')
        
    def show_placeholder(self):
        """Hiển thị placeholder cho suggestions"""
        placeholder = tk.Label(self.suggestions_frame,
                              text="Nhập text để xem gợi ý AI thông minh...\n⚡ Optimized Performance • 🧠 Smart Caching • 📈 Real-time Processing",
                              font=('Segoe UI', 12, 'italic'),
                              bg='white',
                              fg='#bdc3c7',
                              pady=50)
        placeholder.pack(fill='both', expand=True)
    
    def on_key_press(self, event):
        """Handle immediate key press for responsiveness"""
        # Update status immediately for responsiveness
        self.update_status_immediately("Typing...")
    
    def on_input_change_debounced(self, *args):
        """Debounced input change handler để giảm lag"""
        # Cancel previous timer
        if self.debounce_timer:
            self.root.after_cancel(self.debounce_timer)
        
        # Set new timer
        self.debounce_timer = self.root.after(self.debounce_delay, self.process_input_change)
    
    def process_input_change(self):
        """Process input change after debounce"""
        user_input = self.input_var.get().strip()
        
        # Skip if same as last input
        if user_input == self.last_input:
            return
        
        self.last_input = user_input
        
        if not user_input:
            self.clear_suggestions()
            self.show_placeholder()
            self.update_status_immediately("Ready")
            return
        
        # Only process if input has minimum length
        if len(user_input) >= 2:
            # Cancel previous thread if running
            if self.suggestion_thread and self.suggestion_thread.is_alive():
                # Note: We can't cancel thread in Python easily, so we use a flag
                self.is_processing = False
            
            # Start new background processing
            self.is_processing = True
            self.show_loading()
            self.suggestion_thread = threading.Thread(
                target=self.update_suggestions_background, 
                args=(user_input,),
                daemon=True
            )
            self.suggestion_thread.start()
    
    def show_loading(self):
        """Show loading indicator"""
        self.loading_label.configure(text="⏳ Processing...")
        self.update_status_immediately("Processing suggestions...")
    
    def hide_loading(self):
        """Hide loading indicator"""
        self.loading_label.configure(text="")
    
    def update_suggestions_background(self, user_input: str):
        """Update suggestions trong background thread"""
        if not self.is_processing:
            return
        
        try:
            start_time = time.time()
            
            # Get recommendations
            recommendations = self.recommender.smart_recommend(
                user_input, 
                self.context, 
                max_suggestions=8
            )
            
            response_time = time.time() - start_time
            
            # Only update UI if still processing this input
            if self.is_processing and user_input == self.last_input:
                # Schedule UI update on main thread
                self.root.after(0, self.update_ui_with_suggestions, recommendations, response_time, user_input)
                
        except Exception as e:
            if self.is_processing:
                self.root.after(0, self.handle_suggestion_error, str(e))
    
    def update_ui_with_suggestions(self, recommendations: List[Tuple[str, float, str]], response_time: float, original_input: str):
        """Update UI với suggestions trên main thread"""
        # Double check we're still processing the same input
        if original_input != self.last_input:
            return
        
        self.hide_loading()
        self.clear_suggestions()
        
        if not recommendations:
            no_result = tk.Label(self.suggestions_frame,
                                text="❌ Không tìm thấy gợi ý phù hợp\n💡 Thử nghiệm với: 'toihoc', 'xinchao', 'chucmung'",
                                font=('Segoe UI', 11),
                                bg='white',
                                fg='#e74c3c',
                                pady=20)
            no_result.pack()
            return
        
        # Hiển thị suggestions
        for i, (text, confidence, rec_type) in enumerate(recommendations):
            self.create_advanced_suggestion_button(i + 1, text, confidence, rec_type)
        
        # Update session stats
        self.session_stats["suggestions_shown"] += len(recommendations)
        self.session_stats["response_times"].append(response_time)
        self.update_session_stats_display()
        
        # Update status với performance info
        avg_confidence = sum(conf for _, conf, _ in recommendations) / len(recommendations)
        self.update_status(f"Found {len(recommendations)} suggestions in {response_time*1000:.1f}ms | Avg confidence: {avg_confidence:.2f}")
    
    def handle_suggestion_error(self, error_message: str):
        """Handle error trong suggestion processing"""
        self.hide_loading()
        self.clear_suggestions()
        
        error_label = tk.Label(self.suggestions_frame,
                              text=f"❌ Lỗi: {error_message}",
                              font=('Segoe UI', 11),
                              bg='white',
                              fg='#e74c3c',
                              pady=20)
        error_label.pack()
        
        self.update_status(f"Error: {error_message}")
    
    def update_status_immediately(self, message: str):
        """Update status ngay lập tức để responsive"""
        self.stats_labels["status"].configure(text=message)
    
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
        info_text = f"Confidence: {confidence:.3f} | {algo_info}"
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
            "dict_exact": "🎯 Exact Match",
            "dict_prefix": "🔍 Prefix Match", 
            "dict_fuzzy": "🧩 Fuzzy Match",
            "dict_contains": "📝 Contains",
            "advanced_split": "🧠 AI Split",
            "pattern_match": "🎨 Pattern",
            "context_extend": "📈 Context",
            "context_prepend": "⬅️ Prepend",
            "context_predict": "🔮 Predict"
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
        """Xử lý khi user chọn suggestion"""
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
        self.last_input = ""
        
        # Update context
        self.context.extend(text.split())
        if len(self.context) > 15:
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
        self.update_status(f"✅ Selected: '{text}' | {algo_desc} | Confidence: {confidence:.3f}")
        
        # Focus lại input
        self.input_entry.focus_set()
        
    def update_session_stats_display(self):
        """Cập nhật hiển thị session stats"""
        # Update selection count
        self.stats_labels["session"].configure(text=f"{self.session_stats['selections_made']} selections")
        
        # Update performance stats
        if self.session_stats["response_times"]:
            avg_response = sum(self.session_stats["response_times"]) / len(self.session_stats["response_times"])
            self.stats_labels["avg_response"].configure(text=f"{avg_response*1000:.1f}ms")
        
        # Update cache info
        perf_stats = self.recommender.get_performance_stats()
        self.stats_labels["cache"].configure(text=f"{perf_stats['cache_sizes']['recommendations']} cached")
        
        # Update status
        self.stats_labels["status"].configure(text="Active")
    
    def show_performance_stats(self):
        """Hiển thị thống kê performance chi tiết"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("⚡ Performance Statistics")
        stats_window.geometry("600x500")
        stats_window.configure(bg='#f0f0f0')
        
        # Header
        header_label = tk.Label(stats_window,
                               text="⚡ Performance Statistics",
                               font=('Segoe UI', 16, 'bold'),
                               bg='#f0f0f0')
        header_label.pack(pady=20)
        
        # Stats text widget
        stats_text = tk.Text(stats_window,
                            font=('Consolas', 10),
                            bg='white',
                            relief='solid',
                            bd=1,
                            wrap='word')
        stats_text.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Generate performance stats
        perf_stats = self.recommender.get_performance_stats()
        dict_stats = self.recommender.get_statistics()
        session_time = time.time() - self.session_stats["session_start"]
        
        if self.session_stats["response_times"]:
            session_avg_response = sum(self.session_stats["response_times"]) / len(self.session_stats["response_times"])
            min_response = min(self.session_stats["response_times"])
            max_response = max(self.session_stats["response_times"])
        else:
            session_avg_response = 0
            min_response = 0
            max_response = 0
        
        stats_content = f"""
⚡ PERFORMANCE STATISTICS
{'='*50}

📊 Response Times:
  • Average (AI Engine): {perf_stats['avg_response_time_ms']:.1f}ms
  • Average (Session): {session_avg_response*1000:.1f}ms
  • Minimum: {min_response*1000:.1f}ms
  • Maximum: {max_response*1000:.1f}ms
  • Samples: {len(self.session_stats['response_times'])}

🧠 Memory & Cache:
  • Recommendation Cache: {perf_stats['cache_sizes']['recommendations']} entries
  • Text Split Cache: {perf_stats['cache_sizes']['splits']} entries
  • Context Cache: {perf_stats['cache_sizes']['context']} entries
  • Total Dictionary: {dict_stats['word_count'] + len(self.recommender.dictionary.phrases):,} entries

⚙️ Performance Settings:
  • Max Processing Time: {perf_stats['performance_settings']['max_processing_time_ms']:.1f}ms
  • Debounce Delay: {self.debounce_delay}ms
  • Cache Timeout: {perf_stats['performance_settings']['cache_timeout_s']}s

📈 Session Statistics:
  • Session Duration: {session_time/60:.1f} minutes
  • Suggestions Shown: {self.session_stats['suggestions_shown']}
  • Selections Made: {self.session_stats['selections_made']}
  • Selection Rate: {(self.session_stats['selections_made']/max(self.session_stats['suggestions_shown'], 1)*100):.1f}%
  • User Preferences: {len(self.recommender.user_preferences)} learned

🔧 Optimization Features:
  • Background Processing: ✅ Enabled
  • Input Debouncing: ✅ {self.debounce_delay}ms
  • Smart Caching: ✅ Multi-level
  • Performance Monitoring: ✅ Real-time
"""
        
        stats_text.insert('1.0', stats_content)
        stats_text.configure(state='disabled')
    
    def show_detailed_stats(self):
        """Hiển thị thống kê chi tiết"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("📊 Detailed Statistics")
        stats_window.geometry("700x600")
        stats_window.configure(bg='#f0f0f0')
        
        # Header
        header_label = tk.Label(stats_window,
                               text="📊 Advanced AI Statistics",
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
        perf_stats = self.recommender.get_performance_stats()
        session_time = time.time() - self.session_stats["session_start"]
        
        stats_content = f"""
🧠 AI ENGINE STATISTICS (Production Ready)
{'='*60}

📚 Dictionary & Data:
  • Words: {dict_stats['word_count']:,} entries
  • Phrases: {len(self.recommender.dictionary.phrases):,} entries
  • Total Vocabulary: {dict_stats['word_count'] + len(self.recommender.dictionary.phrases):,} entries

🔗 N-gram Language Models:
  • Bigrams: {dict_stats['bigram_count']:,} patterns
  • Trigrams: {dict_stats['trigram_count']:,} patterns
  • 4-grams: {dict_stats['fourgram_count']:,} patterns

👤 Machine Learning:
  • User Preferences: {dict_stats['user_preferences']} learned
  • Adaptive Weights: Dynamic scoring
  • Context Window: 15 words maximum

📈 Session Analytics:
  • Duration: {session_time/60:.1f} minutes
  • Total Suggestions: {self.session_stats['suggestions_shown']}
  • User Selections: {self.session_stats['selections_made']}
  • Success Rate: {(self.session_stats['selections_made']/max(self.session_stats['suggestions_shown'], 1)*100):.1f}%

⚡ Performance Metrics:
  • Avg Response Time: {perf_stats['avg_response_time_ms']:.1f}ms
  • Cache Hit Rate: Optimized
  • Memory Usage: Efficient
  • Background Processing: Active

🏆 Top Learned Words:
"""
        
        if dict_stats['top_preferences']:
            for i, (word, pref) in enumerate(dict_stats['top_preferences'][:7], 1):
                stats_content += f"  {i}. {word}: {pref:.2f} weight\n"
        else:
            stats_content += "  (No preferences learned yet)\n"
        
        stats_content += f"""
🎯 Most Frequent Words:
"""
        for i, (word, freq) in enumerate(dict_stats['top_words'][:7], 1):
            stats_content += f"  {i}. {word}: {freq} occurrences\n"
        
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
            # Cancel any pending processing
            self.is_processing = False
            
            # Get first suggestion if available
            try:
                recommendations = self.recommender.smart_recommend(user_input, self.context, max_suggestions=1)
                if recommendations:
                    text, confidence, rec_type = recommendations[0]
                    self.select_advanced_suggestion(text, confidence, rec_type)
            except:
                pass
                
    def clear_all(self):
        """Xóa tất cả"""
        self.is_processing = False  # Stop any background processing
        self.input_var.set("")
        self.last_input = ""
        self.output_text.delete('1.0', 'end')
        self.context.clear()
        self.clear_suggestions()
        self.show_placeholder()
        self.hide_loading()
        self.update_status("Cleared - Ready for new input")
        
    def copy_result(self):
        """Copy kết quả"""
        result = self.output_text.get('1.0', 'end-1c')
        if result:
            self.root.clipboard_clear()
            self.root.clipboard_append(result)
            self.update_status("✅ Copied to clipboard")
        else:
            self.update_status("❌ Nothing to copy")
            
    def show_settings(self):
        """Hiển thị cài đặt production ready"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("⚙️ Production Settings")
        settings_window.geometry("600x500")
        settings_window.configure(bg='#f0f0f0')
        
        settings_label = tk.Label(settings_window,
                                 text="⚙️ Production Ready Settings",
                                 font=('Segoe UI', 16, 'bold'),
                                 bg='#f0f0f0')
        settings_label.pack(pady=20)
        
        # Performance settings frame
        perf_frame = tk.LabelFrame(settings_window, text="⚡ Performance Settings", 
                                  font=('Segoe UI', 12, 'bold'),
                                  bg='#f0f0f0', padx=20, pady=10)
        perf_frame.pack(fill='x', padx=20, pady=10)
        
        # Current performance stats
        perf_stats = self.recommender.get_performance_stats()
        
        perf_info = [
            f"🎯 Debounce Delay: {self.debounce_delay}ms",
            f"⏱️ Max Processing: {perf_stats['performance_settings']['max_processing_time_ms']:.0f}ms",
            f"🗂️ Cache Timeout: {perf_stats['performance_settings']['cache_timeout_s']}s",
            f"📦 Cache Size: {perf_stats['cache_sizes']['recommendations']} entries"
        ]
        
        for info in perf_info:
            info_label = tk.Label(perf_frame, text=info,
                                 font=('Segoe UI', 11),
                                 bg='#f0f0f0', anchor='w')
            info_label.pack(fill='x', pady=2)
        
        # Future features frame
        future_frame = tk.LabelFrame(settings_window, text="🚀 Coming in Phase 4",
                                    font=('Segoe UI', 12, 'bold'),
                                    bg='#f0f0f0', padx=20, pady=10)
        future_frame.pack(fill='x', padx=20, pady=10)
        
        future_features = [
            "🎛️ Adjustable Performance Settings",
            "🎨 UI Themes & Customization",
            "📊 Advanced Analytics Dashboard",
            "🔄 Auto-update System",
            "💾 Export/Import Configurations",
            "🌐 Multi-language Support",
            "📱 Mobile App Integration"
        ]
        
        for feature in future_features:
            feature_label = tk.Label(future_frame, text=f"• {feature}",
                                   font=('Segoe UI', 11),
                                   bg='#f0f0f0', fg='#7f8c8d',
                                   anchor='w')
            feature_label.pack(fill='x', pady=2)
        
        # Current status
        status_label = tk.Label(settings_window,
                               text="✅ Current: Production Ready v2.1\n🚧 Next: Deployment & Distribution",
                               font=('Segoe UI', 11, 'italic'),
                               bg='#f0f0f0', fg='#27ae60',
                               justify='center')
        status_label.pack(pady=20)
        
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