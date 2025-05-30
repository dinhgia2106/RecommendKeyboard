"""
Vietnamese AI Keyboard UI - GPT Edition
Modern UI with GPT-powered Vietnamese non-accented predictions
"""

from core.text_processor import TextProcessor
from core.ai_recommender import get_ai_recommender, AIRecommender
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AIKeyboardUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Vietnamese AI Keyboard - GPT Edition 🤖")
        self.root.geometry("1200x800")
        self.root.configure(bg="#0f0f0f")

        # Initialize AI recommender
        self.recommender = None
        self.text_processor = TextProcessor()
        self.init_ai_recommender()

        # UI State
        self.current_suggestions = []
        self.selected_index = 0
        self.context_words = []
        self.last_update_time = 0
        self.debounce_delay = 0.2  # 200ms debounce for better responsiveness

        # Session Statistics
        self.session_stats = {
            'total_predictions': 0,
            'ai_predictions': 0,
            'fallback_predictions': 0,
            'selections_made': 0,
            'avg_confidence': 0.0,
            'start_time': time.time()
        }

        self.create_modern_ui()
        self.setup_bindings()

    def init_ai_recommender(self):
        """Initialize AI Recommender in background"""
        def load_ai_recommender():
            try:
                self.recommender = get_ai_recommender()
                self.root.after(0, self.on_ai_recommender_loaded)
            except Exception as e:
                self.root.after(0, lambda: self.show_error(
                    f"Error loading AI Recommender: {e}"))

        # Show loading screen
        self.show_loading_screen()

        # Load in background thread
        thread = threading.Thread(target=load_ai_recommender, daemon=True)
        thread.start()

    def show_loading_screen(self):
        """Show loading screen while initializing AI"""
        self.loading_frame = tk.Frame(self.root, bg="#0f0f0f")
        self.loading_frame.pack(fill="both", expand=True)

        # AI Loading animation
        self.loading_label = tk.Label(
            self.loading_frame,
            text="🤖 Loading Vietnamese AI Keyboard...",
            font=("Segoe UI", 20, "bold"),
            fg="#00ff88",
            bg="#0f0f0f"
        )
        self.loading_label.pack(expand=True)

        # Progress animation
        self.progress_label = tk.Label(
            self.loading_frame,
            text="Initializing GPT model and tokenizer...",
            font=("Segoe UI", 12),
            fg="#888888",
            bg="#0f0f0f"
        )
        self.progress_label.pack()

        # Animate loading
        self.animate_loading()

    def animate_loading(self):
        """Animate loading text with AI theme"""
        texts = [
            "🤖 Loading Vietnamese AI Keyboard",
            "🤖 Loading Vietnamese AI Keyboard ●",
            "🤖 Loading Vietnamese AI Keyboard ●●",
            "🤖 Loading Vietnamese AI Keyboard ●●●"
        ]

        if hasattr(self, 'loading_label') and self.loading_label.winfo_exists():
            current_text = self.loading_label.cget("text")
            try:
                next_index = (texts.index(current_text) + 1) % len(texts)
            except ValueError:
                next_index = 0

            self.loading_label.configure(text=texts[next_index])
            self.root.after(400, self.animate_loading)

    def on_ai_recommender_loaded(self):
        """Called when AI recommender is loaded"""
        self.loading_frame.destroy()
        self.show_ready_message()

    def show_ready_message(self):
        """Show ready message with AI stats"""
        if self.recommender:
            stats = self.recommender.get_statistics()

            # Create status message
            message = "🚀 AI Keyboard Ready!\n\n"

            if stats.get('ai_engine_available'):
                message += "✅ GPT Model: Loaded\n"
                message += f"📊 Vocabulary: {stats.get('ai_vocab_size', 0):,} words\n"
                message += f"🧠 Device: {stats.get('ai_device', 'Unknown')}\n"
                message += f"⚡ Cache: Active\n\n"
                message += "Ready for intelligent Vietnamese typing!"
            else:
                message += "⚠️  GPT Model: Not available\n"
                message += "📖 Fallback: Simple recommendations\n\n"
                message += "Basic Vietnamese typing ready."

            messagebox.showinfo("AI Keyboard Ready", message)

    def show_error(self, error_message):
        """Show error message"""
        messagebox.showerror("Error", error_message)
        # Continue with fallback mode
        self.on_ai_recommender_loaded()

    def create_modern_ui(self):
        """Create modern AI-themed UI"""
        # Configure styles
        style = ttk.Style()
        style.theme_use("clam")

        # Configure AI dark theme colors
        style.configure("AI.TFrame", background="#0f0f0f")
        style.configure("AI.TLabel", background="#0f0f0f",
                        foreground="#ffffff")
        style.configure("AI.TButton", background="#333333",
                        foreground="#ffffff")
        style.map("AI.TButton", background=[("active", "#555555")])

        # Main container
        main_frame = ttk.Frame(self.root, style="AI.TFrame")
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # Title with AI branding
        title_frame = tk.Frame(main_frame, bg="#0f0f0f")
        title_frame.pack(fill="x", pady=(0, 25))

        title_label = tk.Label(
            title_frame,
            text="Vietnamese AI Keyboard",
            font=("Segoe UI", 24, "bold"),
            fg="#00ff88",
            bg="#0f0f0f"
        )
        title_label.pack(side="left")

        subtitle_label = tk.Label(
            title_frame,
            text="🤖 GPT Edition",
            font=("Segoe UI", 16),
            fg="#ffaa00",
            bg="#0f0f0f"
        )
        subtitle_label.pack(side="right")

        # Input section
        self.create_input_section(main_frame)

        # Suggestions section
        self.create_suggestions_section(main_frame)

        # Statistics section
        self.create_statistics_section(main_frame)

        # Control buttons
        self.create_control_buttons(main_frame)

    def create_input_section(self, parent):
        """Create enhanced input section"""
        input_frame = tk.Frame(parent, bg="#0f0f0f")
        input_frame.pack(fill="x", pady=(0, 20))

        # Input label with example
        label_frame = tk.Frame(input_frame, bg="#0f0f0f")
        label_frame.pack(fill="x")

        input_label = tk.Label(
            label_frame,
            text="Nhập non-accented (VD: xinchao → xin chào):",
            font=("Segoe UI", 12, "bold"),
            fg="#ffffff",
            bg="#0f0f0f"
        )
        input_label.pack(side="left")

        # Mode indicator
        self.mode_label = tk.Label(
            label_frame,
            text="🤖 AI Mode",
            font=("Segoe UI", 10, "bold"),
            fg="#00ff88",
            bg="#0f0f0f"
        )
        self.mode_label.pack(side="right")

        # Input entry with better styling
        self.input_var = tk.StringVar()
        self.input_entry = tk.Entry(
            input_frame,
            textvariable=self.input_var,
            font=("Segoe UI", 16),
            bg="#1a1a1a",
            fg="#ffffff",
            insertbackground="#00ff88",
            relief="flat",
            bd=0,
            highlightthickness=2,
            highlightcolor="#00ff88",
            highlightbackground="#333333"
        )
        self.input_entry.pack(fill="x", pady=(8, 0), ipady=12)

        # Context display with better formatting
        self.context_label = tk.Label(
            input_frame,
            text="📝 Context: (none)",
            font=("Segoe UI", 10),
            fg="#aaaaaa",
            bg="#0f0f0f"
        )
        self.context_label.pack(anchor="w", pady=(8, 0))

    def create_suggestions_section(self, parent):
        """Create enhanced suggestions section"""
        suggestions_frame = tk.Frame(parent, bg="#0f0f0f")
        suggestions_frame.pack(fill="both", expand=True, pady=(0, 20))

        # Suggestions header
        header_frame = tk.Frame(suggestions_frame, bg="#0f0f0f")
        header_frame.pack(fill="x")

        suggestions_label = tk.Label(
            header_frame,
            text="🎯 AI Predictions:",
            font=("Segoe UI", 12, "bold"),
            fg="#ffffff",
            bg="#0f0f0f"
        )
        suggestions_label.pack(side="left")

        # Method indicator
        self.method_label = tk.Label(
            header_frame,
            text="",
            font=("Segoe UI", 9),
            fg="#888888",
            bg="#0f0f0f"
        )
        self.method_label.pack(side="right")

        # Suggestions container with modern styling
        suggestions_container = tk.Frame(
            suggestions_frame, bg="#1a1a1a", relief="flat", bd=1)
        suggestions_container.pack(fill="both", expand=True, pady=(8, 0))

        # Scrollbar
        scrollbar = tk.Scrollbar(
            suggestions_container, bg="#444444", troughcolor="#1a1a1a")
        scrollbar.pack(side="right", fill="y")

        # Listbox with modern styling
        self.suggestions_listbox = tk.Listbox(
            suggestions_container,
            font=("Segoe UI", 12),
            bg="#1a1a1a",
            fg="#ffffff",
            selectbackground="#00ff88",
            selectforeground="#000000",
            relief="flat",
            bd=0,
            highlightthickness=0,
            yscrollcommand=scrollbar.set
        )
        self.suggestions_listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.suggestions_listbox.yview)

        # Detailed info frame
        self.info_frame = tk.Frame(suggestions_frame, bg="#0f0f0f")
        self.info_frame.pack(fill="x", pady=(15, 0))

        self.info_label = tk.Label(
            self.info_frame,
            text="💡 Select a suggestion to see details",
            font=("Segoe UI", 10),
            fg="#888888",
            bg="#0f0f0f",
            anchor="w",
            justify="left"
        )
        self.info_label.pack(fill="x")

    def create_statistics_section(self, parent):
        """Create enhanced statistics section"""
        stats_frame = tk.Frame(parent, bg="#0f0f0f")
        stats_frame.pack(fill="x", pady=(0, 20))

        # Stats header
        stats_header = tk.Frame(stats_frame, bg="#0f0f0f")
        stats_header.pack(fill="x")

        stats_label = tk.Label(
            stats_header,
            text="📊 Session Statistics:",
            font=("Segoe UI", 12, "bold"),
            fg="#ffffff",
            bg="#0f0f0f"
        )
        stats_label.pack(side="left")

        # Performance indicator
        self.performance_label = tk.Label(
            stats_header,
            text="⚡ Real-time",
            font=("Segoe UI", 9),
            fg="#00ff88",
            bg="#0f0f0f"
        )
        self.performance_label.pack(side="right")

        # Stats display with modern styling
        self.stats_text = scrolledtext.ScrolledText(
            stats_frame,
            height=8,
            font=("Consolas", 10),
            bg="#1a1a1a",
            fg="#00ff88",
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightcolor="#333333"
        )
        self.stats_text.pack(fill="x", pady=(8, 0))
        self.update_statistics_display()

    def create_control_buttons(self, parent):
        """Create enhanced control buttons"""
        button_frame = tk.Frame(parent, bg="#0f0f0f")
        button_frame.pack(fill="x")

        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear All",
            font=("Segoe UI", 11, "bold"),
            bg="#ff4444",
            fg="#ffffff",
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            command=self.clear_all
        )
        clear_btn.pack(side="left", padx=(0, 15))

        # Benchmark button
        benchmark_btn = tk.Button(
            button_frame,
            text="⚡ Benchmark",
            font=("Segoe UI", 11, "bold"),
            bg="#4444ff",
            fg="#ffffff",
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            command=self.run_benchmark
        )
        benchmark_btn.pack(side="left", padx=(0, 15))

        # AI Statistics button
        ai_stats_btn = tk.Button(
            button_frame,
            text="🤖 AI Stats",
            font=("Segoe UI", 11, "bold"),
            bg="#ff8800",
            fg="#ffffff",
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            command=self.show_ai_statistics
        )
        ai_stats_btn.pack(side="left", padx=(0, 15))

        # Context button
        context_btn = tk.Button(
            button_frame,
            text="📝 Clear Context",
            font=("Segoe UI", 11, "bold"),
            bg="#8844ff",
            fg="#ffffff",
            relief="flat",
            bd=0,
            padx=20,
            pady=8,
            command=self.clear_context
        )
        context_btn.pack(side="left")

    def setup_bindings(self):
        """Setup event bindings"""
        # Input change binding
        self.input_var.trace("w", self.on_input_change)

        # Enter key binding
        self.input_entry.bind("<Return>", self.on_enter_pressed)

        # Number key bindings for suggestions (1-9)
        for i in range(1, 10):
            self.root.bind(f"<Key-{i}>", lambda e,
                           idx=i-1: self.select_suggestion(idx))

        # Arrow key bindings
        self.root.bind("<Up>", self.navigate_up)
        self.root.bind("<Down>", self.navigate_down)

        # Listbox bindings
        self.suggestions_listbox.bind(
            "<<ListboxSelect>>", self.on_suggestion_select)
        self.suggestions_listbox.bind(
            "<Double-Button-1>", self.on_suggestion_double_click)

        # Focus on input
        self.input_entry.focus_set()

    def on_input_change(self, *args):
        """Handle input change with debouncing"""
        current_time = time.time()
        self.last_update_time = current_time

        # Debounce: wait before processing
        self.root.after(int(self.debounce_delay * 1000),
                        lambda: self.process_input_change(current_time))

    def process_input_change(self, trigger_time):
        """Process input change if it's the latest one"""
        if trigger_time == self.last_update_time:
            self.update_suggestions()

    def update_suggestions(self):
        """Update suggestions using AI recommender"""
        if not self.recommender:
            return

        user_input = self.input_var.get().strip()
        if not user_input:
            self.clear_suggestions()
            return

        try:
            # Get AI recommendations
            recommendations = self.recommender.smart_recommend(
                user_input=user_input,
                context=self.context_words,
                max_suggestions=8
            )

            self.current_suggestions = recommendations
            self.display_suggestions(recommendations)

            # Update session stats
            self.session_stats['total_predictions'] += 1
            if recommendations:
                methods = [method for _, _, method in recommendations]
                if any('model' in method or 'combined' in method for method in methods):
                    self.session_stats['ai_predictions'] += 1
                else:
                    self.session_stats['fallback_predictions'] += 1

                # Update mode indicator
                top_method = recommendations[0][2]
                if 'model' in top_method or 'combined' in top_method:
                    self.mode_label.configure(text="🤖 AI Mode", fg="#00ff88")
                else:
                    self.mode_label.configure(text="📖 Fallback", fg="#ffaa00")

        except Exception as e:
            print(f"Error getting AI suggestions: {e}")
            self.clear_suggestions()

    def display_suggestions(self, suggestions):
        """Display suggestions with enhanced formatting"""
        self.suggestions_listbox.delete(0, tk.END)

        for i, (text, confidence, method) in enumerate(suggestions):
            # Format display text with method icons
            method_icon = {
                'model': '🤖',
                'combined': '🔗',
                'frequency': '📊',
                'simple': '📖'
            }.get(method, '❓')

            display_text = f"{i+1}. {text}"
            confidence_text = f" ({confidence:.1%})"
            method_text = f" {method_icon}"

            full_text = display_text + confidence_text + method_text
            self.suggestions_listbox.insert(tk.END, full_text)

        # Select first suggestion
        if suggestions:
            self.suggestions_listbox.selection_set(0)
            self.selected_index = 0
            self.update_suggestion_info(0)

            # Update method indicator
            methods_count = {}
            for _, _, method in suggestions:
                methods_count[method] = methods_count.get(method, 0) + 1

            method_summary = ", ".join(
                [f"{method}:{count}" for method, count in methods_count.items()])
            self.method_label.configure(text=f"Methods: {method_summary}")

    def clear_suggestions(self):
        """Clear all suggestions"""
        self.suggestions_listbox.delete(0, tk.END)
        self.current_suggestions = []
        self.selected_index = 0
        self.info_label.configure(text="💡 No suggestions available")
        self.method_label.configure(text="")

    def on_suggestion_select(self, event):
        """Handle suggestion selection"""
        selection = self.suggestions_listbox.curselection()
        if selection:
            index = selection[0]
            self.selected_index = index
            self.update_suggestion_info(index)

    def on_suggestion_double_click(self, event):
        """Handle suggestion double click"""
        selection = self.suggestions_listbox.curselection()
        if selection:
            self.select_suggestion(selection[0])

    def update_suggestion_info(self, index):
        """Update detailed suggestion info"""
        if 0 <= index < len(self.current_suggestions):
            text, confidence, method = self.current_suggestions[index]

            # Enhanced info display
            info_text = f"📝 Text: {text}\n"
            info_text += f"📊 Confidence: {confidence:.2%}\n"
            info_text += f"🔧 Method: {method}\n"
            info_text += f"📏 Length: {len(text)} chars\n"
            info_text += f"🔤 Words: {len(text.split())} word(s)"

            self.info_label.configure(text=info_text)
        else:
            self.info_label.configure(text="💡 No selection")

    def select_suggestion(self, index):
        """Select and apply a suggestion"""
        if 0 <= index < len(self.current_suggestions):
            selected_text, confidence, method = self.current_suggestions[index]

            # Update context in recommender
            self.recommender.update_context(selected_text)

            # Update local context for display
            words = self.text_processor.tokenize(selected_text)
            self.context_words.extend(words)
            if len(self.context_words) > 5:  # Keep only last 5 words
                self.context_words = self.context_words[-5:]

            # Update context display
            if self.context_words:
                context_display = " → ".join(
                    self.context_words[-3:])  # Show last 3 words
                self.context_label.configure(
                    text=f"📝 Context: {context_display}")
            else:
                self.context_label.configure(text="📝 Context: (none)")

            # Clear input and update stats
            self.input_var.set("")
            self.session_stats['selections_made'] += 1

            # Update average confidence
            if self.session_stats['selections_made'] > 0:
                self.session_stats['avg_confidence'] = (
                    (self.session_stats['avg_confidence'] *
                     (self.session_stats['selections_made'] - 1) + confidence)
                    / self.session_stats['selections_made']
                )

            # Show selection feedback
            self.show_selection_feedback(selected_text, confidence, method)

            # Update statistics
            self.update_statistics_display()

    def show_selection_feedback(self, text, confidence, method):
        """Show visual feedback for selection"""
        # Flash effect
        if hasattr(self, 'suggestions_listbox'):
            original_bg = self.suggestions_listbox.cget('selectbackground')
            self.suggestions_listbox.configure(selectbackground="#ffff00")
            self.root.after(150, lambda: self.suggestions_listbox.configure(
                selectbackground=original_bg))

    def navigate_up(self, event):
        """Navigate up in suggestions"""
        if self.current_suggestions and self.selected_index > 0:
            self.selected_index -= 1
            self.suggestions_listbox.selection_clear(0, tk.END)
            self.suggestions_listbox.selection_set(self.selected_index)
            self.suggestions_listbox.see(self.selected_index)
            self.update_suggestion_info(self.selected_index)

    def navigate_down(self, event):
        """Navigate down in suggestions"""
        if self.current_suggestions and self.selected_index < len(self.current_suggestions) - 1:
            self.selected_index += 1
            self.suggestions_listbox.selection_clear(0, tk.END)
            self.suggestions_listbox.selection_set(self.selected_index)
            self.suggestions_listbox.see(self.selected_index)
            self.update_suggestion_info(self.selected_index)

    def on_enter_pressed(self, event):
        """Handle Enter key press"""
        if self.current_suggestions:
            self.select_suggestion(self.selected_index)

    def clear_all(self):
        """Clear all data"""
        self.input_var.set("")
        self.context_words = []
        self.context_label.configure(text="📝 Context: (none)")
        self.clear_suggestions()

        # Clear context in recommender
        if self.recommender:
            self.recommender.clear_context()

        # Reset session stats
        self.session_stats = {
            'total_predictions': 0,
            'ai_predictions': 0,
            'fallback_predictions': 0,
            'selections_made': 0,
            'avg_confidence': 0.0,
            'start_time': time.time()
        }
        self.update_statistics_display()

    def clear_context(self):
        """Clear only context"""
        self.context_words = []
        self.context_label.configure(text="📝 Context: (none)")
        if self.recommender:
            self.recommender.clear_context()

    def run_benchmark(self):
        """Run AI recommender benchmark"""
        if not self.recommender:
            messagebox.showwarning("Warning", "AI Recommender not available")
            return

        # Run benchmark in background to avoid UI freeze
        def benchmark_thread():
            try:
                self.recommender.benchmark_performance()
            except Exception as e:
                print(f"Benchmark error: {e}")

        messagebox.showinfo(
            "Benchmark", "Running AI benchmark... Check console for results.")
        thread = threading.Thread(target=benchmark_thread, daemon=True)
        thread.start()

    def show_ai_statistics(self):
        """Show detailed AI statistics"""
        if not self.recommender:
            messagebox.showwarning("Warning", "AI Recommender not available")
            return

        stats = self.recommender.get_statistics()

        # Create statistics window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("AI Statistics")
        stats_window.geometry("700x600")
        stats_window.configure(bg="#0f0f0f")

        # Statistics text
        stats_text = scrolledtext.ScrolledText(
            stats_window,
            font=("Consolas", 10),
            bg="#1a1a1a",
            fg="#00ff88",
            relief="flat",
            bd=0
        )
        stats_text.pack(fill="both", expand=True, padx=15, pady=15)

        # Format statistics
        stats_content = "🤖 VIETNAMESE AI KEYBOARD STATISTICS\n"
        stats_content += "=" * 60 + "\n\n"

        # AI Engine Stats
        stats_content += "🧠 AI Engine:\n"
        stats_content += f"  Model loaded: {stats.get('ai_engine_available', False)}\n"
        stats_content += f"  ML module: {stats.get('ml_module_available', False)}\n"
        stats_content += f"  Vocabulary size: {stats.get('ai_vocab_size', 0):,}\n"
        stats_content += f"  Device: {stats.get('ai_device', 'Unknown')}\n"

        # Performance Stats
        stats_content += f"\n⚡ Performance:\n"
        stats_content += f"  Recommendations served: {stats.get('recommendations_served', 0)}\n"
        stats_content += f"  Average response time: {stats.get('avg_response_time', 0)*1000:.2f}ms\n"
        stats_content += f"  Prediction count: {stats.get('ai_prediction_count', 0)}\n"
        stats_content += f"  Cache hits: {stats.get('ai_cache_hits', 0)}\n"
        stats_content += f"  Cache hit rate: {stats.get('ai_cache_hit_rate', 0):.1%}\n"
        stats_content += f"  Cache size: {stats.get('ai_cache_size', 0)}\n"

        # Context Stats
        stats_content += f"\n📝 Context:\n"
        stats_content += f"  Current length: {stats.get('context_length', 0)}\n"
        stats_content += f"  Current words: {stats.get('current_context', [])}\n"

        # Session Stats
        session_time = time.time() - self.session_stats['start_time']
        stats_content += f"\n📊 Session:\n"
        stats_content += f"  Session time: {session_time:.1f}s\n"
        stats_content += f"  Total predictions: {self.session_stats['total_predictions']}\n"
        stats_content += f"  AI predictions: {self.session_stats['ai_predictions']}\n"
        stats_content += f"  Fallback predictions: {self.session_stats['fallback_predictions']}\n"
        stats_content += f"  Selections made: {self.session_stats['selections_made']}\n"
        stats_content += f"  Average confidence: {self.session_stats['avg_confidence']:.1%}\n"

        if self.session_stats['total_predictions'] > 0:
            ai_ratio = self.session_stats['ai_predictions'] / \
                self.session_stats['total_predictions']
            stats_content += f"  AI usage ratio: {ai_ratio:.1%}\n"

        stats_text.insert("1.0", stats_content)
        stats_text.configure(state="disabled")

    def update_statistics_display(self):
        """Update session statistics display"""
        session_time = time.time() - self.session_stats['start_time']

        stats_text = "🤖 AI KEYBOARD SESSION STATISTICS\n"
        stats_text += "=" * 40 + "\n"
        stats_text += f"⏰ Session time: {session_time:.1f}s\n"
        stats_text += f"🎯 Total predictions: {self.session_stats['total_predictions']}\n"
        stats_text += f"🤖 AI predictions: {self.session_stats['ai_predictions']}\n"
        stats_text += f"📖 Fallback predictions: {self.session_stats['fallback_predictions']}\n"
        stats_text += f"✅ Selections made: {self.session_stats['selections_made']}\n"

        if self.session_stats['selections_made'] > 0:
            stats_text += f"📊 Avg confidence: {self.session_stats['avg_confidence']:.1%}\n"

        if self.session_stats['total_predictions'] > 0:
            ai_ratio = self.session_stats['ai_predictions'] / \
                self.session_stats['total_predictions']
            stats_text += f"🧠 AI usage: {ai_ratio:.1%}\n"

        # Context info
        if self.context_words:
            stats_text += f"\n📝 Current context:\n"
            stats_text += f"  {' → '.join(self.context_words)}\n"
        else:
            stats_text += f"\n📝 No context set\n"

        # Recommender stats
        if self.recommender:
            ai_stats = self.recommender.get_statistics()
            stats_text += f"\n🤖 AI Engine Performance:\n"
            stats_text += f"  Response time: {ai_stats.get('avg_response_time', 0)*1000:.1f}ms\n"
            stats_text += f"  Cache hit rate: {ai_stats.get('ai_cache_hit_rate', 0):.1%}\n"

        # Update display
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", stats_text)

    def run(self):
        """Start the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = AIKeyboardUI()
    app.run()
