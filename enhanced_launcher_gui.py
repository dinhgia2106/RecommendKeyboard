#!/usr/bin/env python3
"""
Enhanced Vietnamese Keyboard Launcher - Powered by Hybrid Processor
Bộ gõ tiếng Việt nâng cao với 44K+ từ vựng từ Viet74K và corpus
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
from typing import List, Dict

# Import hybrid processor
try:
    from ml.hybrid_vietnamese_processor import HybridVietnameseProcessor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import hybrid processor: {e}")
    PROCESSOR_AVAILABLE = False


class EnhancedVietnameseKeyboardGUI:
    """Enhanced Vietnamese Keyboard GUI với Hybrid Processor"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🇻🇳 Vietnamese AI Keyboard - Enhanced with 44K+ Words")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f5f6fa')

        # Initialize AI system
        self.processor = None
        self.current_suggestions = []

        if PROCESSOR_AVAILABLE:
            try:
                print("🚀 Initializing Enhanced Hybrid Processor...")
                self.processor = HybridVietnameseProcessor()
                stats = self.processor.get_statistics()
                print(
                    f"✅ System ready with {stats['total_dictionaries']:,} words!")
            except Exception as e:
                print(f"❌ Error initializing processor: {e}")

        self.setup_enhanced_gui()
        self.bind_events()

    def setup_enhanced_gui(self):
        """Setup enhanced GUI với design hiện đại"""

        # Header với statistics
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="🇻🇳 VIETNAMESE AI KEYBOARD - ENHANCED EDITION",
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title_label.pack(pady=15)

        # Statistics info
        if self.processor:
            stats = self.processor.get_statistics()
            stats_label = tk.Label(
                header_frame,
                text=f"📊 Core: {stats['core_count']:,} | Extended: {stats['extended_count']:,} | Total: {stats['total_dictionaries']:,} words",
                font=('Arial', 10),
                bg='#2c3e50',
                fg='#bdc3c7'
            )
            stats_label.pack()

        # Main content area
        main_frame = tk.Frame(self.root, bg='#f5f6fa')
        main_frame.pack(fill='both', expand=True, padx=15, pady=15)

        # Input section with enhanced styling
        input_section = tk.LabelFrame(
            main_frame,
            text="✏️ Nhập văn bản (không dấu)",
            font=('Arial', 12, 'bold'),
            bg='#f5f6fa',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        input_section.pack(fill='x', pady=(0, 15))

        self.input_entry = tk.Entry(
            input_section,
            font=('Arial', 16),
            bg='white',
            fg='#2c3e50',
            relief='solid',
            borderwidth=2,
            insertwidth=3
        )
        self.input_entry.pack(fill='x', pady=5)

        # Quick examples
        examples_label = tk.Label(
            input_section,
            text="💡 Thử: toihocbai, toilasinhvien, homnaytoilam, xemphimhomnay",
            font=('Arial', 9, 'italic'),
            bg='#f5f6fa',
            fg='#7f8c8d'
        )
        examples_label.pack()

        # Suggestions section với enhanced styling
        suggestions_section = tk.LabelFrame(
            main_frame,
            text="🎯 Gợi ý thông minh (Click hoặc nhấn 1,2,3,4,5)",
            font=('Arial', 12, 'bold'),
            bg='#f5f6fa',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        suggestions_section.pack(fill='x', pady=(0, 15))

        self.suggestions_frame = tk.Frame(suggestions_section, bg='#f5f6fa')
        self.suggestions_frame.pack(fill='x', pady=5)

        # Output section với enhanced styling
        output_section = tk.LabelFrame(
            main_frame,
            text="📄 Văn bản đã xử lý",
            font=('Arial', 12, 'bold'),
            bg='#f5f6fa',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        output_section.pack(fill='both', expand=True, pady=(0, 15))

        self.output_text = scrolledtext.ScrolledText(
            output_section,
            font=('Arial', 14),
            bg='white',
            fg='#2c3e50',
            relief='solid',
            borderwidth=2,
            wrap='word',
            height=12,
            insertwidth=3
        )
        self.output_text.pack(fill='both', expand=True, pady=5)

        # Control panel với enhanced styling
        control_panel = tk.Frame(main_frame, bg='#f5f6fa')
        control_panel.pack(fill='x')

        # Left controls
        left_controls = tk.Frame(control_panel, bg='#f5f6fa')
        left_controls.pack(side='left')

        self.clear_btn = tk.Button(
            left_controls,
            text="🗑️ Xóa tất cả",
            font=('Arial', 11, 'bold'),
            bg='#e74c3c',
            fg='white',
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=8,
            command=self.clear_output,
            cursor='hand2'
        )
        self.clear_btn.pack(side='left', padx=(0, 10))

        self.convert_btn = tk.Button(
            left_controls,
            text="🔄 Chuyển đổi",
            font=('Arial', 11, 'bold'),
            bg='#3498db',
            fg='white',
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=8,
            command=self.manual_convert,
            cursor='hand2'
        )
        self.convert_btn.pack(side='left', padx=(0, 10))

        # Copy button
        self.copy_btn = tk.Button(
            left_controls,
            text="📋 Copy",
            font=('Arial', 11, 'bold'),
            bg='#27ae60',
            fg='white',
            relief='flat',
            borderwidth=0,
            padx=20,
            pady=8,
            command=self.copy_output,
            cursor='hand2'
        )
        self.copy_btn.pack(side='left')

        # Right status
        right_status = tk.Frame(control_panel, bg='#f5f6fa')
        right_status.pack(side='right')

        self.status_label = tk.Label(
            right_status,
            text="✅ Enhanced system ready",
            font=('Arial', 10),
            bg='#f5f6fa',
            fg='#27ae60'
        )
        self.status_label.pack(side='right', padx=10)

        # Performance info
        if self.processor:
            self.performance_label = tk.Label(
                right_status,
                text="⚡ Hybrid Processor",
                font=('Arial', 9),
                bg='#f5f6fa',
                fg='#8e44ad'
            )
            self.performance_label.pack(side='right', padx=(0, 20))

    def bind_events(self):
        """Bind keyboard and GUI events"""
        # Auto-process on typing
        self.input_entry.bind('<KeyRelease>', self.on_input_change)
        self.input_entry.bind('<Return>', self.on_enter_pressed)

        # Keyboard shortcuts cho suggestions (1-5)
        for i in range(1, 6):
            self.root.bind(f'<Key-{i}>', lambda e,
                           idx=i-1: self.apply_suggestion(idx))

        # Focus management
        self.root.bind('<Button-1>', self.on_root_click)

        # Keyboard shortcuts
        self.root.bind('<Control-l>', lambda e: self.clear_output())
        self.root.bind('<Control-Return>', lambda e: self.manual_convert())

    def on_input_change(self, event):
        """Handle input text changes"""
        input_text = self.input_entry.get().strip()

        if not input_text:
            self.clear_suggestions()
            return

        self.process_text_async(input_text)

    def process_text_async(self, input_text):
        """Process text asynchronously để tránh block UI"""
        if not self.processor:
            self.show_error("Processor not available")
            return

        def process():
            try:
                start_time = time.time()
                results = self.processor.process_text(
                    input_text, max_suggestions=10)
                processing_time = time.time() - start_time

                # Update UI in main thread
                self.root.after(0, lambda: self.display_results(
                    results, processing_time))

            except Exception as e:
                self.root.after(0, lambda: self.show_error(
                    f"Processing error: {e}"))

        # Run in background thread
        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def display_results(self, results: List[Dict], processing_time: float):
        """Display processing results"""
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

            # Update status với performance info
            self.update_status(
                f"✨ {len(results)} suggestions in {processing_time*1000:.1f}ms",
                color='#27ae60'
            )
        else:
            self.clear_suggestions()
            self.update_status("❌ No suggestions found", color='#e74c3c')

    def get_method_icon(self, method: str) -> str:
        """Get icon for processing method"""
        method_icons = {
            'core_sentence': '🎯',
            'core_compound': '🔗',
            'corpus_trigram': '⭐',
            'corpus_bigram': '💫',
            'extended_compound': '📦',
            'extended_word': '📚',
            'hybrid_segmentation': '🧠',
            'segmentation_variation': '🧠',
            'partial_match': '🤔',
            'aggressive_segmentation': '⚡',
            'conservative_segmentation': '🏛️',
            'simple_segmentation': '📝',
            'partial_compound': '🔗',
            'partial_extended': '📦'
        }
        return method_icons.get(method, '⚡')

    def display_suggestions(self):
        """Display suggestions với enhanced styling"""
        # Clear previous suggestions
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

        # Display new suggestions
        for i, suggestion in enumerate(self.current_suggestions):
            # Create suggestion frame
            suggestion_frame = tk.Frame(self.suggestions_frame, bg='#f5f6fa')
            suggestion_frame.pack(fill='x', pady=2)

            # Suggestion button với color coding
            confidence = suggestion['confidence']
            if confidence >= 90:
                bg_color = '#27ae60'  # Green for high confidence
            elif confidence >= 80:
                bg_color = '#f39c12'  # Orange for medium confidence
            else:
                bg_color = '#3498db'  # Blue for lower confidence

            btn_text = f"{i+1}. {suggestion['text']} {suggestion['icon']} ({confidence}%)"

            suggestion_btn = tk.Button(
                suggestion_frame,
                text=btn_text,
                font=('Arial', 12, 'bold'),
                bg=bg_color,
                fg='white',
                relief='flat',
                borderwidth=0,
                command=lambda idx=i: self.apply_suggestion(idx),
                cursor='hand2',
                padx=15,
                pady=8
            )
            suggestion_btn.pack(fill='x')

            # Hover effects
            def on_enter(e, btn=suggestion_btn, original_bg=bg_color):
                btn.config(bg=self.darken_color(original_bg))

            def on_leave(e, btn=suggestion_btn, original_bg=bg_color):
                btn.config(bg=original_bg)

            suggestion_btn.bind("<Enter>", on_enter)
            suggestion_btn.bind("<Leave>", on_leave)

    def darken_color(self, color: str) -> str:
        """Darken color for hover effect"""
        color_map = {
            '#27ae60': '#229954',
            '#f39c12': '#e67e22',
            '#3498db': '#2980b9'
        }
        return color_map.get(color, color)

    def clear_suggestions(self):
        """Clear all suggestions"""
        self.current_suggestions = []
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

    def apply_suggestion(self, index: int):
        """Apply selected suggestion"""
        if 0 <= index < len(self.current_suggestions):
            suggestion = self.current_suggestions[index]

            # Add to output với space handling
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

            self.update_status(
                f"✅ Applied: {suggestion['text']}", color='#27ae60')

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
                self.update_status(
                    f"🔄 Converted: {best_result}", color='#3498db')

    def clear_output(self):
        """Clear the output text"""
        self.output_text.delete('1.0', 'end')
        self.input_entry.delete(0, 'end')
        self.clear_suggestions()
        self.update_status("🗑️ Cleared", color='#e74c3c')

    def copy_output(self):
        """Copy output to clipboard"""
        output_text = self.output_text.get('1.0', 'end-1c')
        if output_text.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(output_text)
            self.update_status("📋 Copied to clipboard", color='#27ae60')
        else:
            self.update_status("❌ Nothing to copy", color='#e74c3c')

    def on_root_click(self, event):
        """Handle clicks on root window"""
        # Keep focus on input entry
        self.input_entry.focus_set()

    def update_status(self, message: str, color: str = '#27ae60'):
        """Update status message với color"""
        self.status_label.config(text=message, fg=color)

        # Auto-clear status after 3 seconds
        def clear_status():
            time.sleep(3)
            self.status_label.config(
                text="✅ Enhanced system ready", fg='#27ae60')

        threading.Thread(target=clear_status, daemon=True).start()

    def show_error(self, error_msg: str):
        """Show error message"""
        self.status_label.config(text=f"❌ {error_msg}", fg='#e74c3c')

    def run(self):
        """Run the Enhanced GUI application"""
        print("🚀 Starting Enhanced Vietnamese AI Keyboard GUI...")
        self.input_entry.focus_set()

        # Show welcome message
        if self.processor:
            stats = self.processor.get_statistics()
            welcome_text = f"Chào mừng đến với bộ gõ tiếng Việt nâng cao!\n\nHệ thống đã sẵn sàng với {stats['total_dictionaries']:,} từ vựng.\nBắt đầu gõ để trải nghiệm..."
            self.output_text.insert('1.0', welcome_text)

        self.root.mainloop()


def main():
    """Main function"""
    if not PROCESSOR_AVAILABLE:
        print("❌ Hybrid processor not available. Please check installation.")
        return

    try:
        app = EnhancedVietnameseKeyboardGUI()
        app.run()
    except Exception as e:
        print(f"❌ Error starting enhanced application: {e}")


if __name__ == "__main__":
    main()
