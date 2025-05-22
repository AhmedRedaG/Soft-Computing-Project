import numpy as np
import pandas as pd
import pickle
import logging
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime
from anfis_model import ANFIS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/anfis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_preprocessors():
    """Load saved model, scaler, encoder, and selected features"""
    try:
        with open('savedmodel/anfis_model.pkl', 'rb') as f:
            model_params = pickle.load(f)
        with open('savedmodel/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('savedmodel/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        with open('savedmodel/selected_features.pkl', 'rb') as f:
            selected_features = pickle.load(f)
        logger.info("Model, scaler, encoder, and selected features loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    anfis = ANFIS(
        n_membership=model_params['n_membership'],
        learning_rate=model_params['learning_rate'],
        epochs=model_params['epochs'],
        max_rules=model_params['max_rules'],
        n_classes=model_params['n_classes']
    )
    anfis.centers = model_params['centers']
    anfis.spreads = model_params['spreads']
    anfis.consequent_params = model_params['consequent_params']

    return anfis, scaler, encoder, selected_features

class ANFISGUI:
    def __init__(self, root, anfis, scaler, encoder, selected_features):
        self.root = root
        self.anfis = anfis
        self.scaler = scaler
        self.encoder = encoder
        self.selected_features = selected_features
        self.attack_types = encoder.categories_[0]
        self.feature_names = ['id', 'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
                             'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
                             'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
                             'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
                             'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd',
                             'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']
        self.feature_descriptions = {
            'id': 'Record identifier',
            'dur': 'Duration of the connection (seconds)',
            'proto': 'Protocol (e.g., tcp, udp)',
            'service': 'Service type (e.g., http, ftp)',
            'state': 'Connection state (e.g., FIN, INT)',
            'spkts': 'Source packets',
            'dpkts': 'Destination packets',
            'sbytes': 'Source bytes',
            'dbytes': 'Destination bytes',
            'rate': 'Packet rate (packets/second)',
            'sttl': 'Source time-to-live',
            'dttl': 'Destination time-to-live',
            'sload': 'Source bits per second',
            'dload': 'Destination bits per second',
            'sloss': 'Source packets retransmitted or dropped',
            'dloss': 'Destination packets retransmitted or dropped',
            'sinpkt': 'Source inter-packet arrival time (ms)',
            'dinpkt': 'Destination inter-packet arrival time (ms)',
            'sjit': 'Source jitter (ms)',
            'djit': 'Destination jitter (ms)',
            'swin': 'Source TCP window size',
            'stcpb': 'Source TCP base sequence number',
            'dtcpb': 'Destination TCP base sequence number',
            'dwin': 'Destination TCP window size',
            'tcprtt': 'TCP round-trip time',
            'synack': 'Time between SYN and ACK',
            'ackdat': 'Time between ACK and data',
            'smean': 'Mean of source packet sizes',
            'dmean': 'Mean of destination packet sizes',
            'trans_depth': 'HTTP transaction depth',
            'response_body_len': 'Response body length',
            'ct_srv_src': 'Connections to same service from source',
            'ct_state_ttl': 'Connections with same state and TTL',
            'ct_dst_ltm': 'Connections to destination in last 2 minutes',
            'ct_src_dport_ltm': 'Connections from source to destination port',
            'ct_dst_sport_ltm': 'Connections to destination from source port',
            'ct_dst_src_ltm': 'Connections between source and destination',
            'is_ftp_login': '1 if FTP login, 0 otherwise',
            'ct_ftp_cmd': 'Number of FTP commands',
            'ct_flw_http_mthd': 'Number of HTTP methods',
            'ct_src_ltm': 'Connections from source in last 2 minutes',
            'ct_srv_dst': 'Connections to same service and destination',
            'is_sm_ips_ports': '1 if same IP and port, 0 otherwise'
        }
        
        self.root.title("ANFIS Attack Predictor")
        self.root.geometry("900x700")
        self.root.configure(bg="#E8ECEF")  # Soft white-gray background
        logger.info("GUI launched")

        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        # Button style with hover effect
        style.configure("TButton", padding=8, font=("Helvetica", 12), background="#2ECC71", foreground="white", relief="flat")
        style.map("TButton", background=[('active', '#27AE60'), ('!disabled', '#2ECC71')])
        # Label style
        style.configure("TLabel", font=("Helvetica", 12), background="#E8ECEF", foreground="#333333")
        # Entry style with focus effect
        style.configure("TEntry", font=("Helvetica", 12), fieldbackground="white", foreground="black")
        style.map("TEntry", fieldbackground=[('focus', 'white')], selectbackground=[('focus', '#1A73E8')], selectforeground=[('focus', 'white')])
        # Frame style with shadow effect
        style.configure("TFrame", background="#E8ECEF")
        style.configure("TLabelframe", background="#E8ECEF", relief="groove", borderwidth=2)
        style.configure("TLabelframe.Label", background="#E8ECEF", foreground="#333333", font=("Helvetica", 11, "bold"))

        # Main frame
        main_frame = ttk.Frame(self.root, padding=10, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="ANFIS Attack Predictor", font=("Helvetica", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # Instructions frame
        instructions_frame = ttk.LabelFrame(main_frame, text="Instructions", padding=5, style="TLabelframe")
        instructions_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        instructions = (
            "Enter a UNSW-NB15 record (43 fields: id,dur,proto,service,state,...,is_sm_ips_ports)\n"
            "Format: comma-separated values\n"
            f"Model uses: {', '.join(selected_features)}\n"
            "Example: 1,0.1,tcp,http,FIN,10,8,500,400,100,1,1,1000,800,2,1,0.01,0.02,50,20,255,0,0,255,"
            "0.001,0.0005,0.0005,50,40,0,0,2,1,2,2,2,2,0,0,0,2,2,0"
        )
        instructions_label = ttk.Label(instructions_frame, text=instructions, wraplength=850, justify=tk.LEFT, font=("Helvetica", 10), foreground="#666666")
        instructions_label.pack(fill=tk.X)

        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Data", padding=5, style="TLabelframe")
        input_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        ttk.Label(input_frame, text="Input Record (43 fields):", font=("Helvetica", 12)).grid(row=0, column=0, sticky="w")
        self.input_entry = ttk.Entry(input_frame, width=80, style="TEntry")
        self.input_entry.grid(row=1, column=0, sticky="ew", pady=3)

        # Button frame
        button_frame = ttk.LabelFrame(main_frame, text="Actions", padding=5, style="TLabelframe")
        button_frame.grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        ttk.Button(button_frame, text="Predict", command=self.predict, width=15, style="TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Load Example", command=self.load_example, width=15, style="TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Clear", command=self.clear, width=15, style="TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Help", command=self.show_help, width=15, style="TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Exit", command=self.exit, width=15, style="TButton").pack(side=tk.LEFT, padx=3)

        # Separator
        ttk.Separator(main_frame, orient="horizontal").grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)

        # Output label
        self.output_label = ttk.Label(main_frame, text="Prediction: None", font=("Helvetica", 14, "bold"), foreground="#E74C3C")
        self.output_label.grid(row=5, column=0, columnspan=3, pady=10)

        # History frame
        history_frame = ttk.LabelFrame(main_frame, text="Prediction History", padding=5, style="TLabelframe")
        history_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.history_text = scrolledtext.ScrolledText(history_frame, width=80, height=8, font=("Courier", 10), bg="white", fg="black", relief="sunken", borderwidth=1)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        self.history_text.config(state='disabled')
        history_button_frame = ttk.Frame(history_frame)
        history_button_frame.pack(pady=3)
        ttk.Button(history_button_frame, text="Save History", command=self.save_history, width=12, style="TButton").pack(side=tk.LEFT, padx=3)
        ttk.Button(history_button_frame, text="Clear History", command=self.clear_history, width=12, style="TButton").pack(side=tk.LEFT, padx=3)

        # Add context menus for copy-paste
        self.make_context_menu(self.input_entry)
        self.make_context_menu(self.history_text)

    def make_context_menu(self, widget):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Copy", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Paste", command=lambda: widget.event_generate("<<Paste>>"))
        widget.bind("<Button-3>", lambda event: menu.tk_popup(event.x_root, event.y_root))

    def predict(self):
        """Process input and predict attack type"""
        user_input = self.input_entry.get().strip()
        if not user_input:
            logger.error("No input provided")
            messagebox.showerror("Error", "Please enter a record.")
            return
        values = [x.strip() for x in user_input.split(',')]
        
        try:
            # Validate input
            if len(values) != 43:
                logger.error(f"Expected 43 fields, got {len(values)}")
                messagebox.showerror("Error", f"Expected 43 fields, got {len(values)}.")
                return

            logger.info(f"User input received: {values}")

            # Create DataFrame for preprocessing
            input_df = pd.DataFrame([values], columns=self.feature_names)
            
            # Handle categorical features
            categorical_cols = ['proto', 'service', 'state']
            input_df[categorical_cols] = input_df[categorical_cols].astype(str)
            input_df = pd.get_dummies(input_df, columns=categorical_cols)
            logger.info("Categorical features encoded")

            # Ensure all selected features are present
            missing_cols = set(self.selected_features) - set(input_df.columns)
            for col in missing_cols:
                input_df[col] = 0
            X_input = input_df[self.selected_features]

            # Convert to numeric and scale
            X_input = X_input.astype(float)
            X_input_scaled = self.scaler.transform(X_input)
            logger.info("Input features scaled")

            # Predict
            pred = self.anfis.predict(X_input_scaled)
            attack_type = self.attack_types[pred[0]]
            output = "Normal" if attack_type.lower() == "normal" else attack_type
            logger.info(f"Predicted attack type: {output}")
            self.output_label.config(text=f"Prediction: {output}")

            # Update history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_entry = f"{timestamp} | Input: {user_input}\n{timestamp} | Prediction: {output}\n{'-'*80}\n"
            self.history_text.config(state='normal')
            self.history_text.insert(tk.END, history_entry)
            self.history_text.config(state='disabled')
            self.history_text.see(tk.END)

        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            messagebox.showerror("Error", f"Invalid input: Please ensure all feature values are valid (numeric where required).")
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            messagebox.showerror("Error", f"Error: {e}")

    def load_example(self):
        """Load example UNSW-NB15 record (43 fields)"""
        example = (
            "1,0.1,tcp,http,FIN,10,8,500,400,100,1,1,1000,800,2,1,0.01,0.02,50,20,255,0,0,255,"
            "0.001,0.0005,0.0005,50,40,0,0,2,1,2,2,2,2,0,0,0,2,2,0"
        )
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, example)
        logger.info("Example record loaded")

    def show_help(self):
        """Show feature descriptions"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Feature Descriptions")
        help_window.geometry("600x400")
        help_window.configure(bg="#E8ECEF")

        help_frame = ttk.Frame(help_window, padding=10, style="TFrame")
        help_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(help_frame, bg="#E8ECEF")
        scrollbar = ttk.Scrollbar(help_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        for i, (feature, desc) in enumerate(self.feature_descriptions.items()):
            ttk.Label(scrollable_frame, text=f"{feature}: {desc}", wraplength=500, justify=tk.LEFT, font=("Helvetica", 10), foreground="#333333").grid(row=i, column=0, sticky="w", pady=1)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(help_frame, text="Close", command=help_window.destroy, width=12, style="TButton").pack(pady=5)
        logger.info("Help window opened")

    def save_history(self):
        """Save prediction history to file"""
        try:
            with open('results/prediction_history.txt', 'w') as f:
                f.write(self.history_text.get("1.0", tk.END))
            logger.info("Prediction history saved to prediction_history.txt")
            messagebox.showinfo("Success", "Prediction history saved to prediction_history.txt")
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            messagebox.showerror("Error", f"Error saving history: {e}")

    def clear(self):
        """Clear input field and reset output"""
        self.input_entry.delete(0, tk.END)
        self.output_label.config(text="Prediction: None")
        logger.info("Input field cleared")

    def clear_history(self):
        """Clear prediction history"""
        self.history_text.config(state='normal')
        self.history_text.delete("1.0", tk.END)
        self.history_text.config(state='disabled')
        logger.info("Prediction history cleared")

    def exit(self):
        """Close GUI"""
        logger.info("GUI closed")
        self.root.quit()
        self.root.destroy()

def main():
    """Main function to run GUI prediction"""
    logger.info("Starting prediction process")
    anfis, scaler, encoder, selected_features = load_model_and_preprocessors()
    logger.info("Launching GUI for custom input prediction")
    root = tk.Tk()
    app = ANFISGUI(root, anfis, scaler, encoder, selected_features)
    root.mainloop()
    logger.info("Prediction process completed")

if __name__ == "__main__":
    main()