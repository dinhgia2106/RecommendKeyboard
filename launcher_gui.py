#!/usr/bin/env python3
"""
Vietnamese Keyboard Launcher with GUI Interface
Complete Vietnamese text input system with intelligent suggestions
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from typing import List, Dict

# Import our simple Vietnamese processor
try:
    from ml.simple_vietnamese_processor import SimpleVietnameseProcessor
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import modules: {e}")
    MODULES_AVAILABLE = False


class VietnameseKeyboardGUI:
    """Complete Vietnamese Keyboard GUI with intelligent text processing"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🇻🇳 Vietnamese AI Keyboard - Simple & Effective")
        self.root.geometry("900x750")
        self.root.configure(bg='#f0f8ff')

        # Initialize AI system
        self.processor = None

        if MODULES_AVAILABLE:
            try:
                print("🚀 Initializing Simple Vietnamese Processor...")
                self.processor = SimpleVietnameseProcessor()
                print("✅ System ready!")
            except Exception as e:
                print(f"❌ Error initializing system: {e}")

        # GUI state
        self.current_suggestions = []

        self.setup_gui()
        self.bind_events()

    def setup_gui(self):
        """Setup the complete GUI interface"""

        # Title
        title_frame = tk.Frame(self.root, bg='#f0f8ff')
        title_frame.pack(fill='x', padx=10, pady=5)

        title_label = tk.Label(
            title_frame,
            text="🇻🇳 VIETNAMESE AI KEYBOARD - SIMPLE & EFFECTIVE",
            font=('Arial', 16, 'bold'),
            bg='#f0f8ff',
            fg='#2c3e50'
        )
        title_label.pack()

        # Input section
        input_frame = tk.Frame(self.root, bg='#f0f8ff')
        input_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(
            input_frame,
            text="✏️ Nhập văn bản (không dấu):",
            font=('Arial', 12, 'bold'),
            bg='#f0f8ff',
            fg='#34495e'
        ).pack(anchor='w')

        # Input entry with auto-processing
        self.input_entry = tk.Entry(
            input_frame,
            font=('Arial', 14),
            bg='white',
            fg='#2c3e50',
            relief='solid',
            borderwidth=2
        )
        self.input_entry.pack(fill='x', pady=5)

        # Suggestions section
        suggestions_frame = tk.Frame(self.root, bg='#f0f8ff')
        suggestions_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(
            suggestions_frame,
            text="💡 Gợi ý (Click hoặc nhấn 1,2,3):",
            font=('Arial', 12, 'bold'),
            bg='#f0f8ff',
            fg='#34495e'
        ).pack(anchor='w')

        # Suggestions display frame
        self.suggestions_frame = tk.Frame(suggestions_frame, bg='#f0f8ff')
        self.suggestions_frame.pack(fill='x', pady=5)

        # Main output section
        output_frame = tk.Frame(self.root, bg='#f0f8ff')
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)

        tk.Label(
            output_frame,
            text="📄 Văn bản đã xử lý:",
            font=('Arial', 12, 'bold'),
            bg='#f0f8ff',
            fg='#34495e'
        ).pack(anchor='w')

        # Main text output with scrollbar
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            font=('Arial', 14),
            bg='white',
            fg='#2c3e50',
            relief='solid',
            borderwidth=2,
            wrap='word',
            height=15
        )
        self.output_text.pack(fill='both', expand=True, pady=5)

        # Control buttons
        controls_frame = tk.Frame(self.root, bg='#f0f8ff')
        controls_frame.pack(fill='x', padx=10, pady=5)

        # Clear button
        self.clear_btn = tk.Button(
            controls_frame,
            text="🗑️ Xóa",
            font=('Arial', 12),
            bg='#e74c3c',
            fg='white',
            relief='flat',
            borderwidth=0,
            command=self.clear_output
        )
        self.clear_btn.pack(side='left', padx=5)

        # Convert button (manual trigger)
        self.convert_btn = tk.Button(
            controls_frame,
            text="🔄 Chuyển đổi",
            font=('Arial', 12),
            bg='#3498db',
            fg='white',
            relief='flat',
            borderwidth=0,
            command=self.manual_convert
        )
        self.convert_btn.pack(side='left', padx=5)

        # Status info
        self.status_label = tk.Label(
            controls_frame,
            text="✅ Hệ thống sẵn sàng",
            font=('Arial', 10),
            bg='#f0f8ff',
            fg='#27ae60'
        )
        self.status_label.pack(side='right', padx=5)

    def bind_events(self):
        """Bind keyboard and GUI events"""
        # Auto-process on typing
        self.input_entry.bind('<KeyRelease>', self.on_input_change)
        self.input_entry.bind('<Return>', self.on_enter_pressed)

        # Keyboard shortcuts for suggestions
        self.root.bind('<Key-1>', lambda e: self.apply_suggestion(0))
        self.root.bind('<Key-2>', lambda e: self.apply_suggestion(1))
        self.root.bind('<Key-3>', lambda e: self.apply_suggestion(2))

        # Focus management
        self.root.bind('<Button-1>', self.on_root_click)

    def on_input_change(self, event):
        """Handle input text changes with intelligent processing"""
        input_text = self.input_entry.get().strip()

        if not input_text:
            self.clear_suggestions()
            return

        self.process_text(input_text)

    def process_text(self, input_text):
        """Process input using our simple Vietnamese processor"""
        if not self.processor:
            self.show_error("Hệ thống xử lý chưa sẵn sàng")
            return

        try:
            # Get suggestions from processor
            results = self.processor.process_text(
                input_text, max_suggestions=3)

            if results:
                self.current_suggestions = []

                for i, result in enumerate(results):
                    method_icon = self.get_method_icon(result['method'])

                    suggestion = {
                        'text': result['vietnamese_text'],
                        'confidence': result['confidence'],
                        'method': result['method'],
                        'icon': method_icon
                    }
                    self.current_suggestions.append(suggestion)

                self.display_suggestions()
                self.update_status(
                    f"✨ Tìm thấy {len(results)} gợi ý chất lượng")
            else:
                self.clear_suggestions()
                self.update_status("❌ Không có gợi ý phù hợp")

        except Exception as e:
            self.show_error(f"Lỗi xử lý: {e}")

    def get_method_icon(self, method):
        """Get icon for processing method"""
        method_icons = {
            'complete_sentence': '🎯',
            'compound_word': '🔗',
            'syllable_segmentation': '📚',
            'smart_match': '🧠',
            'smart_compound': '⚡'
        }
        return method_icons.get(method, '⭐')

    def display_suggestions(self):
        """Display current suggestions in the GUI"""
        # Clear previous suggestions
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

        # Display new suggestions
        for i, suggestion in enumerate(self.current_suggestions):
            # Create suggestion button
            btn_text = f"{i+1}. {suggestion['text']} {suggestion['icon']}({suggestion['confidence']}%)"

            suggestion_btn = tk.Button(
                self.suggestions_frame,
                text=btn_text,
                font=('Arial', 12),
                bg='#ecf0f1',
                fg='#2c3e50',
                relief='solid',
                borderwidth=1,
                command=lambda idx=i: self.apply_suggestion(idx),
                cursor='hand2'
            )
            suggestion_btn.pack(fill='x', pady=2)

    def clear_suggestions(self):
        """Clear all suggestions"""
        self.current_suggestions = []
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

    def apply_suggestion(self, index):
        """Apply selected suggestion to output"""
        if 0 <= index < len(self.current_suggestions):
            suggestion = self.current_suggestions[index]

            # Add to output text
            current_output = self.output_text.get('1.0', 'end-1c')
            if current_output and not current_output.endswith(' '):
                new_text = current_output + ' ' + suggestion['text']
            else:
                new_text = current_output + suggestion['text']

            self.output_text.delete('1.0', 'end')
            self.output_text.insert('1.0', new_text)

            # Clear input and suggestions
            self.input_entry.delete(0, 'end')
            self.clear_suggestions()

            # Focus back to input
            self.input_entry.focus_set()

            self.update_status(f"✅ Đã áp dụng: {suggestion['text']}")

    def on_enter_pressed(self, event):
        """Handle Enter key press"""
        if self.current_suggestions:
            self.apply_suggestion(0)  # Apply first suggestion
        else:
            self.manual_convert()

    def manual_convert(self):
        """Manual conversion trigger"""
        input_text = self.input_entry.get().strip()
        if not input_text:
            return

        if self.processor:
            best_result = self.processor.get_best_suggestion(input_text)
            if best_result:
                # Add to output
                current_output = self.output_text.get('1.0', 'end-1c')
                if current_output and not current_output.endswith(' '):
                    new_text = current_output + ' ' + best_result
                else:
                    new_text = current_output + best_result

                self.output_text.delete('1.0', 'end')
                self.output_text.insert('1.0', new_text)

                self.input_entry.delete(0, 'end')
                self.clear_suggestions()
                self.update_status(f"✅ Đã chuyển đổi: {best_result}")

    def clear_output(self):
        """Clear the output text"""
        self.output_text.delete('1.0', 'end')
        self.input_entry.delete(0, 'end')
        self.clear_suggestions()
        self.update_status("🗑️ Đã xóa nội dung")

    def on_root_click(self, event):
        """Handle clicks on root window"""
        # Keep focus on input entry
        self.input_entry.focus_set()

    def update_status(self, message):
        """Update status message"""
        self.status_label.config(text=message)

        # Auto-clear status after 3 seconds
        def clear_status():
            time.sleep(3)
            self.status_label.config(text="✅ Hệ thống sẵn sàng")

        threading.Thread(target=clear_status, daemon=True).start()

    def show_error(self, error_msg):
        """Show error message"""
        self.status_label.config(text=f"❌ {error_msg}", fg='#e74c3c')

    def run(self):
        """Run the GUI application"""
        print("🚀 Starting Vietnamese AI Keyboard GUI...")
        self.input_entry.focus_set()
        self.root.mainloop()


def main():
    """Main function"""
    if not MODULES_AVAILABLE:
        print("❌ Critical modules missing. Please check installation.")
        return

    try:
        app = VietnameseKeyboardGUI()
        app.run()
    except Exception as e:
        print(f"❌ Error starting application: {e}")


if __name__ == "__main__":
    main()
