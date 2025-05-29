"""
Keyboard UI với Tkinter - Giao diện bàn phím recommend tiếng Việt
"""

import tkinter as tk
from tkinter import ttk, font
import sys
import os
from typing import List, Tuple

# Thêm đường dẫn để import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Recommender


class KeyboardUI:
    def __init__(self):
        self.root = tk.Tk()
        self.recommender = Recommender()
        self.context = []
        
        self.setup_ui()
        self.setup_styles()
        
    def setup_ui(self):
        """Thiết lập giao diện"""
        self.root.title("🚀 Bàn Phím Recommend Tiếng Việt")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Set icon (nếu có)
        try:
            self.root.iconbitmap('assets/icon.ico')
        except:
            pass
        
        # Header
        self.create_header()
        
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
        
        self.style.configure('Suggestion.TButton',
                           font=('Segoe UI', 11),
                           padding=(10, 5))
        
        self.style.configure('Primary.TButton',
                           font=('Segoe UI', 11, 'bold'),
                           padding=(15, 8))
        
    def create_header(self):
        """Tạo header"""
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="🚀 Bàn Phím Recommend Tiếng Việt",
                              font=('Segoe UI', 18, 'bold'),
                              bg='#2c3e50',
                              fg='white')
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(header_frame,
                                 text="Nhập text không dấu để nhận gợi ý thông minh",
                                 font=('Segoe UI', 10),
                                 bg='#2c3e50',
                                 fg='#bdc3c7')
        subtitle_label.pack()
        
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
        
        # Ví dụ
        example_label = tk.Label(input_frame,
                                text="💡 Ví dụ: xinchao, toihoc, chucmung, etc.",
                                font=('Segoe UI', 9, 'italic'),
                                bg='#f0f0f0',
                                fg='#7f8c8d')
        example_label.pack(anchor='w')
        
    def create_suggestions_section(self):
        """Tạo phần suggestions"""
        suggestions_frame = tk.Frame(self.root, bg='#f0f0f0')
        suggestions_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        suggestions_label = tk.Label(suggestions_frame,
                                    text="💡 Gợi ý:",
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
        self.status_bar = tk.Label(self.root,
                                  text="Sẵn sàng | Đã load từ điển với 285 entries",
                                  font=('Segoe UI', 9),
                                  bg='#34495e',
                                  fg='white',
                                  anchor='w',
                                  padx=10)
        self.status_bar.pack(fill='x', side='bottom')
        
    def show_placeholder(self):
        """Hiển thị placeholder cho suggestions"""
        placeholder = tk.Label(self.suggestions_frame,
                              text="Nhập text để xem gợi ý...",
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
            self.update_suggestions(user_input)
        
    def update_suggestions(self, user_input: str):
        """Cập nhật suggestions"""
        self.clear_suggestions()
        
        try:
            # Lấy recommendations
            recommendations = self.recommender.recommend_smart(
                user_input, 
                self.context, 
                max_suggestions=8
            )
            
            if not recommendations:
                no_result = tk.Label(self.suggestions_frame,
                                    text="❌ Không tìm thấy gợi ý phù hợp",
                                    font=('Segoe UI', 11),
                                    bg='white',
                                    fg='#e74c3c',
                                    pady=20)
                no_result.pack()
                return
            
            # Hiển thị suggestions
            for i, (text, confidence, rec_type) in enumerate(recommendations):
                self.create_suggestion_button(i + 1, text, confidence, rec_type)
                
            self.update_status(f"Tìm thấy {len(recommendations)} gợi ý")
            
        except Exception as e:
            error_label = tk.Label(self.suggestions_frame,
                                  text=f"❌ Lỗi: {str(e)}",
                                  font=('Segoe UI', 11),
                                  bg='white',
                                  fg='#e74c3c',
                                  pady=20)
            error_label.pack()
            
    def create_suggestion_button(self, index: int, text: str, confidence: float, rec_type: str):
        """Tạo button cho suggestion"""
        suggestion_frame = tk.Frame(self.suggestions_frame, bg='white')
        suggestion_frame.pack(fill='x', padx=10, pady=2)
        
        # Confidence bar
        confidence_width = int(confidence * 200)
        confidence_color = self.get_confidence_color(confidence)
        
        # Button chính
        btn_text = f"{index}. {text}"
        suggestion_btn = tk.Button(suggestion_frame,
                                  text=btn_text,
                                  font=('Segoe UI', 11),
                                  bg='white',
                                  fg='#2c3e50',
                                  relief='solid',
                                  bd=1,
                                  anchor='w',
                                  padx=15,
                                  pady=8,
                                  command=lambda: self.select_suggestion(text))
        suggestion_btn.pack(fill='x', pady=1)
        
        # Confidence và type info
        info_frame = tk.Frame(suggestion_frame, bg='white')
        info_frame.pack(fill='x', padx=15)
        
        # Confidence bar visual
        confidence_frame = tk.Frame(info_frame, bg='#ecf0f1', height=4)
        confidence_frame.pack(fill='x', pady=(2, 0))
        
        confidence_bar = tk.Frame(confidence_frame, bg=confidence_color, height=4)
        confidence_bar.place(x=0, y=0, width=confidence_width)
        
        # Info text
        info_text = f"Độ tin cậy: {confidence:.2f} | Loại: {rec_type}"
        info_label = tk.Label(info_frame,
                             text=info_text,
                             font=('Segoe UI', 8),
                             bg='white',
                             fg='#7f8c8d')
        info_label.pack(anchor='w', pady=(2, 5))
        
        # Hover effects
        def on_enter(e):
            suggestion_btn.configure(bg='#ecf0f1')
            
        def on_leave(e):
            suggestion_btn.configure(bg='white')
            
        suggestion_btn.bind('<Enter>', on_enter)
        suggestion_btn.bind('<Leave>', on_leave)
        
    def get_confidence_color(self, confidence: float) -> str:
        """Lấy màu dựa trên confidence"""
        if confidence >= 0.8:
            return '#27ae60'  # Green
        elif confidence >= 0.6:
            return '#f39c12'  # Orange
        else:
            return '#e74c3c'  # Red
            
    def select_suggestion(self, text: str):
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
        
        # Update context và learning
        self.context.extend(text.split())
        if len(self.context) > 10:  # Giới hạn context
            self.context = self.context[-10:]
            
        self.recommender.update_user_choice(text, self.context)
        
        # Clear suggestions
        self.clear_suggestions()
        self.show_placeholder()
        
        # Update status
        self.update_status(f"Đã chọn: '{text}'")
        
        # Focus lại input
        self.input_entry.focus_set()
        
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
                recommendations = self.recommender.recommend_smart(user_input, self.context, max_suggestions=1)
                if recommendations:
                    self.select_suggestion(recommendations[0][0])
            except:
                pass
                
    def clear_all(self):
        """Xóa tất cả"""
        self.input_var.set("")
        self.output_text.delete('1.0', 'end')
        self.context.clear()
        self.clear_suggestions()
        self.show_placeholder()
        self.update_status("Đã xóa tất cả")
        
    def copy_result(self):
        """Copy kết quả"""
        result = self.output_text.get('1.0', 'end-1c')
        if result:
            self.root.clipboard_clear()
            self.root.clipboard_append(result)
            self.update_status("Đã copy kết quả")
        else:
            self.update_status("Không có gì để copy")
            
    def show_settings(self):
        """Hiển thị cài đặt"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("⚙️ Cài đặt")
        settings_window.geometry("400x300")
        settings_window.configure(bg='#f0f0f0')
        
        settings_label = tk.Label(settings_window,
                                 text="⚙️ Cài đặt",
                                 font=('Segoe UI', 16, 'bold'),
                                 bg='#f0f0f0')
        settings_label.pack(pady=20)
        
        # Placeholder cho settings
        info_label = tk.Label(settings_window,
                             text="Tính năng cài đặt sẽ được phát triển trong Phase 3",
                             font=('Segoe UI', 11),
                             bg='#f0f0f0',
                             fg='#7f8c8d')
        info_label.pack(pady=20)
        
    def update_status(self, message: str):
        """Cập nhật status bar"""
        self.status_bar.configure(text=message)
        
    def run(self):
        """Chạy ứng dụng"""
        self.root.mainloop()


def main():
    """Main function để chạy UI"""
    app = KeyboardUI()
    app.run()


if __name__ == "__main__":
    main() 